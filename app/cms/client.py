"""Downstream integration boundary: real HTTP calls to the mock CMS.

The pipeline never imports the CMS application - it talks to it over HTTP with
timeouts, bounded retries and explicit error reporting, exactly like it would
against a real vendor endpoint.
"""

from __future__ import annotations

import httpx

from app.models.agent_result import AgentResult
from app.utils.logger import get_logger
from app.utils.retry import (
    NonRetryableError,
    RetryPolicy,
    RetryStats,
    RetryableError,
    retry_async,
)

logger = get_logger("cms.client")

#: Header understood by the mock CMS for controlled failure demos.
SIMULATE_HEADER = "X-Simulate-Error"


class CMSDeliveryError(RuntimeError):
    """Raised when a record could not be delivered to the CMS."""


class CMSClient:
    """Async client for ``POST /records`` and friends."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 15.0,
        policy: RetryPolicy | None = None,
        stats: RetryStats | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.policy = policy or RetryPolicy()
        self.stats = stats or RetryStats()
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout
        )
        self._owns_client = client is None

    async def __aenter__(self) -> "CMSClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> dict:
        """GET /health - used as a pre-flight check by the pipeline."""
        try:
            response = await self._client.get("/health", timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise CMSDeliveryError(f"CMS health check timed out: {self.base_url}") from exc
        except httpx.TransportError as exc:
            raise CMSDeliveryError(
                f"CMS unreachable at {self.base_url}: {exc}. "
                "Start it with: uvicorn mock_cms.main:app --reload --port 8000"
            ) from exc
        if response.status_code >= 400:
            raise CMSDeliveryError(f"CMS health check returned HTTP {response.status_code}")
        return response.json()

    async def deliver(
        self,
        result: AgentResult,
        *,
        simulate: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        """POST a structured AgentResult to the CMS (retried on 5xx/timeouts)."""
        payload = result.model_dump(mode="json")
        headers = {SIMULATE_HEADER: simulate} if simulate else None

        async def attempt() -> dict:
            try:
                response = await self._client.post(
                    "/records",
                    json=payload,
                    headers=headers,
                    timeout=timeout or self.timeout,
                )
            except httpx.TimeoutException as exc:
                raise RetryableError(
                    f"CMS request timed out for record_id={result.record_id}"
                ) from exc
            except httpx.TransportError as exc:
                raise RetryableError(
                    f"CMS connection error for record_id={result.record_id}: {exc}"
                ) from exc

            if response.status_code == 429 or response.status_code >= 500:
                raise RetryableError(
                    f"CMS HTTP {response.status_code} for record_id={result.record_id}: "
                    f"{response.text[:200]}"
                )
            if response.status_code >= 400:
                raise NonRetryableError(
                    f"CMS rejected payload HTTP {response.status_code} "
                    f"record_id={result.record_id}: {response.text[:300]}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise RetryableError("CMS returned malformed JSON") from exc

        try:
            body = await retry_async(
                attempt,
                policy=self.policy,
                description="CMS delivery",
                stats=self.stats,
            )
        except RetryableError as exc:
            raise CMSDeliveryError(
                f"CMS delivery failed for record_id={result.record_id} "
                f"after {self.policy.attempts} attempt(s): {exc}"
            ) from exc
        except NonRetryableError as exc:
            raise CMSDeliveryError(str(exc)) from exc

        logger.info(
            "CMS delivery successful record_id=%s status=%s",
            result.record_id,
            body.get("status", "stored"),
        )
        return body
