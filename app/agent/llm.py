"""LLM provider abstraction.

Everything Gemini-specific lives in this module so the rest of the application
depends only on :class:`BaseLLMProvider`. Adding an Ollama (or any other)
provider means implementing one class and registering it in
:func:`build_llm_provider` - no other module changes.

Two providers ship with the project:

``gemini``
    Calls the public Gemini ``generateContent`` REST API over HTTP. This is the
    default and the only provider that performs real AI inference.

``offline``
    A deterministic, rule-based stand-in that requires **no API key**. It exists
    so the pipeline, the mock CMS and the tests can be exercised without a key
    or network access. It is explicitly *not* an LLM and is announced loudly in
    the logs. It is never selected unless ``LLM_PROVIDER=offline`` is set.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import ConfigError, Settings
from app.utils.logger import get_logger
from app.utils.retry import RetryPolicy, RetryStats, RetryableError, retry_async

logger = get_logger("agent.llm")

GEMINI_API_BASE = "https://generativelanguage.googleapis.com"

#: 429 response shapes emitted by the Gemini API.
_RETRY_DELAY_JSON_RE = re.compile(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"')
_RETRY_IN_TEXT_RE = re.compile(r"Please retry in (\d+)h(\d+)m(\d+(?:\.\d+)?)s")
_HARD_QUOTA_RE = re.compile(
    r"exceeded your current quota|PerDay|free_tier_requests", re.IGNORECASE
)


def _parse_retry_hint(body: str) -> float | None:
    """Extract the server's retry delay (seconds) from a Gemini 429 body."""
    match = _RETRY_DELAY_JSON_RE.search(body)
    if match:
        return float(match.group(1))
    match = _RETRY_IN_TEXT_RE.search(body)
    if match:
        return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    return None


def _format_wait(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h{minutes}m{secs}s" if hours else f"{minutes}m{secs}s"


class LLMError(RuntimeError):
    """Base class for LLM provider failures."""


class LLMRetryableError(LLMError):
    """Transient LLM failure (timeout, 5xx, short-hint 429) - safe to retry."""


class LLMFatalError(LLMError):
    """Permanent LLM failure (bad key, bad model, invalid request)."""


class LLMRateLimitError(LLMRetryableError):
    """Transient 429 rate limit.

    Carries the server's ``retry_after`` hint (seconds) so the retry helper can
    honour it instead of blindly hammering the API with short sleeps.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMQuotaError(LLMError):
    """Provider quota exhausted for the current window (free-tier daily limit).

    Deliberately NOT an :class:`LLMRetryableError`: retrying a multi-hour daily
    quota inside one run is pointless and wastes API calls. The provider also
    uses this error to open a circuit - once raised, every subsequent request
    fails fast without touching the network.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        circuit_open: bool = False,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.circuit_open = circuit_open


#: Server retry hints longer than this are hard quota exhaustion (free-tier
#: daily limits are measured in hours), not a transient per-minute rate limit.
QUOTA_HINT_THRESHOLD_SECONDS = 60.0


@dataclass
class LLMRequest:
    """Provider-agnostic completion request."""

    system_prompt: str
    user_prompt: str
    record: dict[str, Any] = field(default_factory=dict)
    is_repair: bool = False


class BaseLLMProvider(ABC):
    """Minimal interface every provider must implement."""

    name: str = "base"

    @abstractmethod
    async def complete(self, request: LLMRequest) -> str:
        """Return the raw model output for ``request``."""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        """Release any network resources held by the provider."""


# --------------------------------------------------------------------------- #
# Gemini (default, real inference)
# --------------------------------------------------------------------------- #

class GeminiProvider(BaseLLMProvider):
    """Gemini ``generateContent`` REST client built on ``httpx``."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-flash-latest",
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        policy: RetryPolicy | None = None,
        stats: RetryStats | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ConfigError("GeminiProvider requires an API key (GEMINI_API_KEY)")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.policy = policy or RetryPolicy()
        self.stats = stats or RetryStats()
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        #: Set after the first hard-quota failure; fails fast, no HTTP calls.
        self._quota_error: LLMQuotaError | None = None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(self, request: LLMRequest) -> str:
        if self._quota_error is not None:
            raise LLMQuotaError(
                str(self._quota_error),
                retry_after=self._quota_error.retry_after,
                circuit_open=True,
            )
        payload = {
            "system_instruction": {"parts": [{"text": request.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": request.user_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
            },
        }
        url = f"{GEMINI_API_BASE}/v1beta/models/{self.model}:generateContent"

        async def attempt() -> str:
            try:
                response = await self._client.post(
                    url,
                    json=payload,
                    headers={"x-goog-api-key": self.api_key},
                    timeout=self.timeout,
                )
            except httpx.TimeoutException as exc:
                raise LLMRetryableError(
                    f"Gemini request timed out after {self.timeout}s"
                ) from exc
            except httpx.TransportError as exc:
                raise LLMRetryableError(f"Gemini connection error: {exc}") from exc

            return self._handle_response(response)

        try:
            return await retry_async(
                attempt,
                policy=self.policy,
                description="Gemini request",
                stats=self.stats,
                retryable=(RetryableError, LLMRetryableError),
            )
        except LLMQuotaError as exc:
            if self._quota_error is None:
                self._quota_error = exc
                logger.error(
                    "Gemini quota exhausted - circuit opened, no further API calls this "
                    "run (retry after %s)",
                    _format_wait(exc.retry_after),
                )
            raise

    def _handle_response(self, response: httpx.Response) -> str:
        status = response.status_code
        if status == 429:
            raise self._classify_rate_limit(response.text)
        if status >= 500:
            raise LLMRetryableError(f"Gemini HTTP {status}: {response.text[:200]}")
        if status in (400, 403, 404):
            raise LLMFatalError(self._friendly_client_error(status, response.text))
        if status >= 400:
            raise LLMFatalError(f"Gemini HTTP {status}: {response.text[:300]}")

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMRetryableError("Gemini returned malformed JSON") from exc

        candidates = body.get("candidates") or []
        if not candidates:
            feedback = body.get("promptFeedback") or {}
            block = feedback.get("blockReason") or body.get("error", {}).get("message") or "empty"
            raise LLMFatalError(f"Gemini returned no candidates (blockReason={block})")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text.strip():
            finish = candidates[0].get("finishReason", "unknown")
            raise LLMRetryableError(f"Gemini returned empty text (finishReason={finish})")
        return text

    @staticmethod
    def _classify_rate_limit(body: str) -> LLMQuotaError | LLMRateLimitError:
        """Split a 429 into hard quota exhaustion vs a transient rate limit.

        * Long server hint (hours) or daily-quota markers  -> :class:`LLMQuotaError`
          (fail fast, opens the provider circuit).
        * Short hint or no hint                            -> :class:`LLMRateLimitError`
          (retried with exponential backoff, honouring the server hint).
        """
        hint = _parse_retry_hint(body)
        if hint is not None:
            if hint > QUOTA_HINT_THRESHOLD_SECONDS:
                return LLMQuotaError(
                    f"Gemini free-tier quota exhausted: {body[:400]}",
                    retry_after=hint,
                )
            return LLMRateLimitError(
                f"Gemini HTTP 429 (rate limited, server asks to retry in "
                f"{_format_wait(hint)}): {body[:200]}",
                retry_after=hint,
            )
        if _HARD_QUOTA_RE.search(body):
            return LLMQuotaError(f"Gemini free-tier quota exhausted: {body[:400]}")
        return LLMRateLimitError(f"Gemini HTTP 429 (rate limited): {body[:200]}")

    @staticmethod
    def _friendly_client_error(status: int, body: str) -> str:
        lowered = body.lower()
        if status in (401, 403) or "api key" in lowered or "unauthenticated" in lowered:
            return (
                f"Gemini rejected the credentials (HTTP {status}). "
                "Check GEMINI_API_KEY in your .env file."
            )
        if status == 404 or "not found" in lowered:
            return (
                f"Gemini model not found (HTTP 404). Current GEMINI_MODEL may be "
                f"unavailable for your key - try GEMINI_MODEL=gemini-flash-latest or "
                f"list valid ids with GET /v1beta/models. Detail: {body[:200]}"
            )
        return f"Gemini rejected the request (HTTP {status}): {body[:300]}"


# --------------------------------------------------------------------------- #
# Offline rule-based provider (explicit opt-in, no key required)
# --------------------------------------------------------------------------- #

_INTENT_RE = re.compile(
    r"\b(requests?|requested)\b.{0,30}\b(demo|pricing|quote|info|call|meeting|consult)\b"
    r"|\bpricing info\b|\bbook a (call|demo)\b",
    re.IGNORECASE,
)
_ENGAGEMENT_WORDS = (
    "liked", "commented", "shared", "attended", "posted", "connected",
    "mutual", "warm", "engaged", "webinar", "clicked",
)


class OfflineProvider(BaseLLMProvider):
    """Deterministic local rules. NOT a language model.

    Selected only when ``LLM_PROVIDER=offline``. Used for demos/tests without an
    API key; the README documents exactly what it is and is not.
    """

    name = "offline"

    def __init__(self) -> None:
        logger.warning(
            "LLM provider 'offline' active - deterministic rule-based decisions, "
            "NOT an LLM. Set LLM_PROVIDER=gemini and GEMINI_API_KEY to use real AI."
        )

    async def complete(self, request: LLMRequest) -> str:
        record = request.record
        if record.get("record_type") == "patient":
            decision = self._decide_patient(record)
        else:
            decision = self._decide_lead(record)
        return json.dumps(decision, ensure_ascii=False)

    # --- leads ---------------------------------------------------------- #
    @staticmethod
    def _decide_lead(record: dict[str, Any]) -> dict[str, Any]:
        person = record.get("person", {})
        context = record.get("context", {})
        facts = context.get("facts", {})
        tags = set(context.get("tags", []))
        text = " ".join(
            [
                str(context.get("summary", "")),
                str(facts.get("recent_activity", "")),
                str(facts.get("connection_note", "")),
                str(record.get("priority_signal", "")),
            ]
        ).lower()

        name = person.get("name", "there")
        role = person.get("role", "")
        company = person.get("organization", "your clinic")
        activity = str(facts.get("recent_activity") or facts.get("connection_note") or "")

        if "intent" in tags or _INTENT_RE.search(text):
            classification, action, priority, tone = (
                "HOT", "SEND_OUTREACH", "HIGH", "friendly",
            )
            rationale = (
                f"High priority because the record shows explicit intent: "
                f"{activity or record.get('priority_signal')}."
            )
        elif tags & {"engaged", "warm-intro"} or any(word in text for word in _ENGAGEMENT_WORDS):
            classification, action, priority, tone = (
                "WARM", "FOLLOW_UP", "MEDIUM", "professional",
            )
            rationale = (
                f"Warm because the lead recently engaged with relevant content "
                f"('{activity or 'engagement noted'}')."
            )
        else:
            classification, action, priority, tone = (
                "COLD", "NURTURE", "LOW", "neutral",
            )
            rationale = (
                "Cold because the record shows no recent engagement "
                f"({activity or facts.get('recent_activity') or 'no activity recorded'})."
            )

        if action == "SEND_OUTREACH":
            message = (
                f"Hi {name} - saw your note about "
                f"\"{(facts.get('recent_activity') or 'your recent activity').rstrip('.')}\". "
                f"We help {role or 'teams'} at {company} automate intake and follow-up. "
                f"Open to a short conversation this week?"
            )
        elif action == "FOLLOW_UP":
            message = (
                f"Follow up with {name} ({role}, {company}) referencing their recent "
                f"engagement: {(facts.get('recent_activity') or 'content interaction')}."
            )
        elif action == "NURTURE":
            message = (
                f"Add {name} of {company} to the nurture track; no immediate outreach. "
                f"Re-evaluate when new engagement appears."
            )
        else:
            message = f"No action for {name}; record does not meet outreach thresholds."

        return {
            "record_id": record.get("id", ""),
            "source": record.get("source", "linkedin"),
            "record_type": "lead",
            "classification": classification,
            "priority": priority,
            "action": action,
            "channel": "linkedin" if action != "NO_ACTION" else "none",
            "tone": tone,
            "message": message[:800],
            "rationale": rationale[:500],
        }

    # --- patients -------------------------------------------------------- #
    @staticmethod
    def _decide_patient(record: dict[str, Any]) -> dict[str, Any]:
        person = record.get("person", {})
        context = record.get("context", {})
        facts = context.get("facts", {})
        contact = record.get("contact_info", {})
        name = person.get("name", "patient")
        conditions = int(facts.get("condition_count", 0) or 0)
        deceased = bool(facts.get("deceased"))
        has_phone = bool(contact.get("phone"))
        has_email = bool(contact.get("email"))

        if deceased:
            decision = {
                "classification": "LOW_PRIORITY",
                "priority": "LOW",
                "action": "NO_ACTION",
                "channel": "none",
                "tone": "neutral",
                "message": f"No outreach. Record {record.get('id')} is flagged deceased; close workflow item.",
                "rationale": "Administrative only: record is flagged deceased, so no contact is appropriate.",
            }
        elif conditions and (has_phone or has_email):
            decision = {
                "classification": "HIGH_PRIORITY",
                "priority": "HIGH",
                "action": "SCHEDULE_OUTREACH",
                "channel": "phone" if has_phone else "email",
                "tone": "professional",
                "message": (
                    f"Administrative next step for {name}: schedule an outreach call to "
                    f"confirm contact preferences and open items ({conditions} condition(s) on file). "
                    f"No clinical guidance required."
                ),
                "rationale": (
                    f"High administrative priority: {conditions} condition record(s) on file and "
                    f"reachable contact details are present."
                ),
            }
        elif has_phone or has_email:
            decision = {
                "classification": "STANDARD",
                "priority": "MEDIUM",
                "action": "FOLLOW_UP",
                "channel": "phone" if has_phone else "email",
                "tone": "professional",
                "message": (
                    f"Administrative follow-up for {name}: confirm demographics and contact "
                    f"preferences through the available channel."
                ),
                "rationale": "Standard priority: active record with reachable contact details.",
            }
        else:
            decision = {
                "classification": "STANDARD",
                "priority": "MEDIUM",
                "action": "ADMIN_REVIEW",
                "channel": "internal_note",
                "tone": "neutral",
                "message": (
                    f"Internal review for {name}: no contact details on file - queue for "
                    f"manual data-quality check."
                ),
                "rationale": "No contact details in the record, so an internal review is required.",
            }

        return {
            "record_id": record.get("id", ""),
            "source": record.get("source", "fhir"),
            "record_type": "patient",
            **decision,
        }


# --------------------------------------------------------------------------- #
# Demo fault injection
# --------------------------------------------------------------------------- #

class FaultInjectionProvider(BaseLLMProvider):
    """Returns malformed model output for the first N calls (deliberate demo).

    Exercises the "invalid LLM output -> repair retry" path without touching the
    underlying provider.
    """

    def __init__(self, inner: BaseLLMProvider, failures: int = 1) -> None:
        self.inner = inner
        self.name = inner.name
        self._failures = failures

    async def complete(self, request: LLMRequest) -> str:
        if self._failures > 0:
            self._failures -= 1
            logger.error(
                "Simulated invalid LLM output injected (demo failure mode) record=%s",
                request.record.get("id"),
            )
            return '{"classification": "VERY_HOT", "action": "SEND_IT_NOW", }'  # invalid JSON
        return await self.inner.complete(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def build_llm_provider(
    settings: Settings, stats: RetryStats | None = None
) -> BaseLLMProvider:
    """Create the configured provider; raises :class:`ConfigError` if unusable."""
    if settings.llm_provider == "gemini":
        settings.require_llm_credentials()
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.http_timeout_seconds,
            policy=RetryPolicy(
                attempts=settings.max_retries,
                base_delay=settings.retry_backoff_seconds,
            ),
            stats=stats,
        )
    if settings.llm_provider == "offline":
        return OfflineProvider()
    raise ConfigError(f"Unknown LLM provider: {settings.llm_provider}")
