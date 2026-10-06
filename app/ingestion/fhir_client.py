"""Reusable async FHIR R4 client for the SMART Health IT public sandbox.

Endpoint: ``https://r4.smarthealthit.org`` (synthetic data, no auth).

Handled failure modes:

* connection errors and timeouts  -> retryable
* HTTP 429 / 5xx                  -> retryable
* HTTP 4xx (other)                -> non-retryable, surfaced immediately
* malformed JSON body             -> retryable (bounded)
* malformed / missing FHIR fields -> that single resource is skipped, the
  rest of the batch continues
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import httpx

from app.utils.logger import get_logger
from app.utils.retry import (
    NonRetryableError,
    RetryPolicy,
    RetryStats,
    RetryableError,
    retry_async,
)

logger = get_logger("ingestion.fhir")

DEFAULT_BASE_URL = "https://r4.smarthealthit.org"
MAX_BUNDLE_PAGES = 6


class FHIRClientError(RuntimeError):
    """Raised when the FHIR source cannot deliver the requested resources."""


class FHIRClient:
    """Small, dependency-light async FHIR R4 read client."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 20.0,
        policy: RetryPolicy | None = None,
        stats: RetryStats | None = None,
        client: httpx.AsyncClient | None = None,
        simulated_failures: int = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.policy = policy or RetryPolicy()
        self.stats = stats or RetryStats()
        self._client = client
        self._owns_client = client is None
        self._simulated_failures = simulated_failures

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def __aenter__(self) -> "FHIRClient":
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Accept": "application/fhir+json, application/json"},
            )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def fetch_patients(self, count: int) -> list[dict[str, Any]]:
        """Fetch up to ``count`` synthetic ``Patient`` resources (raw JSON)."""
        patients: list[dict[str, Any]] = []
        next_url: str | None = f"/Patient?_count={count}"
        page = 0

        while next_url and len(patients) < count and page < MAX_BUNDLE_PAGES:
            page += 1
            bundle = await self._get(next_url)
            entries = bundle.get("entry")
            if not isinstance(entries, list):
                logger.warning("FHIR bundle page=%s contained no entry array", page)
                break

            for entry in entries:
                if len(patients) >= count:
                    break
                resource = entry.get("resource") if isinstance(entry, dict) else None
                validated = self._validate_patient(resource)
                if validated is not None:
                    patients.append(validated)

            next_url = self._next_link(bundle)

        if not patients:
            raise FHIRClientError(
                f"No usable Patient resources returned by {self.base_url}"
            )
        logger.info("Fetched %s FHIR patients (pages=%s)", len(patients), page)
        return patients

    async def fetch_conditions(self, patient_id: str) -> list[dict[str, Any]]:
        """Fetch ``Condition`` resources for a patient (optional enrichment)."""
        try:
            bundle = await self._get(
                "/Condition", params={"patient": patient_id, "_count": "20"}
            )
        except NonRetryableError as exc:
            logger.warning("Condition lookup skipped patient=%s error=%s", patient_id, exc)
            return []

        conditions: list[dict[str, Any]] = []
        for entry in bundle.get("entry") or []:
            resource = entry.get("resource") if isinstance(entry, dict) else None
            if isinstance(resource, dict) and resource.get("resourceType") == "Condition":
                conditions.append(resource)
        return conditions

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _next_link(self, bundle: dict[str, Any]) -> str | None:
        for link in bundle.get("link") or []:
            if isinstance(link, dict) and link.get("relation") == "next":
                url = link.get("url")
                if isinstance(url, str) and url:
                    # Keep same-host relative URLs so base_url applies.
                    if url.startswith(self.base_url):
                        return url[len(self.base_url):]
                    return url
        return None

    def _validate_patient(self, resource: Any) -> dict[str, Any] | None:
        if not isinstance(resource, dict):
            logger.error("Skipping malformed FHIR resource: not a JSON object")
            return None
        if resource.get("resourceType") != "Patient":
            logger.error(
                "Skipping unexpected FHIR resourceType=%r", resource.get("resourceType")
            )
            return None
        if not resource.get("id"):
            logger.error("Skipping FHIR Patient with missing id")
            return None
        return resource

    async def _get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        """GET with bounded retry; returns the decoded JSON body."""

        async def attempt() -> dict[str, Any]:
            if self._simulated_failures > 0:
                self._simulated_failures -= 1
                raise RetryableError("simulated FHIR 503 response (demo failure mode)")
            assert self._client is not None, "FHIRClient must be used as an async context manager"
            try:
                response = await self._client.get(path, params=params)
            except httpx.TimeoutException as exc:
                raise RetryableError(f"FHIR request timed out: {path}") from exc
            except httpx.TransportError as exc:
                raise RetryableError(f"FHIR connection error: {exc}") from exc

            if response.status_code == 429 or response.status_code >= 500:
                raise RetryableError(
                    f"FHIR HTTP {response.status_code} for {path}"
                )
            if response.status_code >= 400:
                raise NonRetryableError(
                    f"FHIR HTTP {response.status_code} for {path}: {response.text[:200]}"
                )
            try:
                body = response.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise RetryableError(f"FHIR returned malformed JSON for {path}") from exc
            if not isinstance(body, dict):
                raise RetryableError(
                    f"FHIR returned unexpected JSON type {type(body).__name__} for {path}"
                )
            return body

        return await retry_async(
            attempt,
            policy=self.policy,
            description="FHIR request",
            stats=self.stats,
        )


def iter_patient_names(resources: Iterable[dict[str, Any]]) -> list[str]:
    """Tiny helper used by smoke tests / debugging."""
    names: list[str] = []
    for resource in resources:
        for name in resource.get("name") or []:
            given = " ".join(name.get("given") or [])
            family = name.get("family") or ""
            full = f"{given} {family}".strip()
            if full:
                names.append(full)
                break
    return names
