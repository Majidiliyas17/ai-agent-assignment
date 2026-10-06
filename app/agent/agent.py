"""The agent processing loop.

For every :class:`UnifiedRecord` the agent:

1. prepares the relevant context (prompts.build_user_prompt),
2. calls the LLM provider,
3. decodes the structured JSON,
4. validates it (schema + record-type contract),
5. repairs invalid output with a bounded follow-up call,
6. logs the final decision and rationale,
7. returns an :class:`AgentOutcome`.

It never invents a result: if the model cannot produce valid output after the
bounded repair attempts, the record is marked failed and the pipeline
continues with the remaining records.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.agent.llm import BaseLLMProvider, LLMError, LLMQuotaError, LLMRequest
from app.agent.prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt, record_payload
from app.models.agent_result import AgentResult, ResultValidationError, validate_agent_result
from app.models.unified_record import UnifiedRecord
from app.utils.logger import get_logger
from app.utils.retry import RetryStats

logger = get_logger("agent")

#: Total LLM calls allowed per record = 1 initial + MAX_REPAIR_ATTEMPTS repairs.
MAX_REPAIR_ATTEMPTS = 2


@dataclass
class AgentOutcome:
    """Result of processing one record through the agent."""

    record: UnifiedRecord
    result: AgentResult | None = None
    error: str | None = None
    llm_calls: int = 0
    #: True when the record could not be processed because the LLM quota ran out
    #: (distinct from invalid output / network failures - never an offline fill).
    quota_blocked: bool = False

    @property
    def ok(self) -> bool:
        return self.result is not None


def extract_json(raw: str) -> dict[str, Any]:
    """Decode the model output into a dict, tolerating markdown fences.

    Raises :class:`ValueError` when nothing decodable is found.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fall back to the first balanced {...} block.
    start = text.find("{")
    if start != -1:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    raise ValueError("no JSON object found in model output")


class IntakeAgent:
    """Stateful agent wrapper around an LLM provider."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        max_repairs: int = MAX_REPAIR_ATTEMPTS,
        stats: RetryStats | None = None,
    ) -> None:
        self.provider = provider
        self.max_repairs = max_repairs
        self.stats = stats if stats is not None else RetryStats()

    async def process(self, record: UnifiedRecord) -> AgentOutcome:
        """Run the full reasoning loop for a single record."""
        logger.info(
            "Processing record record_id=%s source=%s type=%s",
            record.id, record.source.value, record.record_type.value,
        )

        user_prompt = build_user_prompt(record)
        payload_context = record_payload(record)
        raw_output = ""
        last_error = "model did not respond"
        calls = 0

        for attempt in range(1 + self.max_repairs):
            prompt = (
                user_prompt
                if attempt == 0
                else build_repair_prompt(record, raw_output, last_error)
            )
            request = LLMRequest(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                record=payload_context,
                is_repair=attempt > 0,
            )
            try:
                raw_output = await self.provider.complete(request)
            except LLMQuotaError as exc:
                calls += 1
                self.stats.record_failure("llm_quota")
                if exc.circuit_open:
                    logger.error(
                        "Record NOT processed by %s - quota circuit open, API call "
                        "skipped record_id=%s",
                        self.provider.name, record.id,
                    )
                else:
                    logger.error(
                        "Record NOT processed by %s - API returned free-tier quota "
                        "exhausted record_id=%s retry_after=%ss error=%s",
                        self.provider.name, record.id, exc.retry_after, exc,
                    )
                return AgentOutcome(
                    record=record,
                    error=f"Gemini quota exhausted: {exc}",
                    llm_calls=calls,
                    quota_blocked=True,
                )
            except LLMError as exc:
                calls += 1
                self.stats.record_failure("llm_call")
                logger.error(
                    "LLM call failed record_id=%s attempt=%s/%s error=%s",
                    record.id, attempt + 1, 1 + self.max_repairs, exc,
                )
                return AgentOutcome(record=record, error=f"LLM call failed: {exc}", llm_calls=calls)

            calls += 1
            if attempt > 0:
                logger.info("Retrying LLM record_id=%s attempt=%s (repair)", record.id, attempt + 1)

            try:
                payload = extract_json(raw_output)
            except ValueError as exc:
                last_error = str(exc)
                self.stats.record_failure("llm_repair")
                logger.warning(
                    "LLM returned invalid structured output record_id=%s attempt=%s error=%s",
                    record.id, attempt + 1, exc,
                )
                continue

            try:
                result = validate_agent_result(
                    record.id, record.source, record.record_type, payload
                )
            except ResultValidationError as exc:
                last_error = str(exc)
                self.stats.record_failure("llm_repair")
                logger.warning(
                    "AgentResult validation failed record_id=%s attempt=%s error=%s",
                    record.id, attempt + 1, exc,
                )
                continue

            result = result.model_copy(update={"model_provider": self.provider.name})
            self._log_decision(record, result, self.provider.name)
            return AgentOutcome(record=record, result=result, llm_calls=calls)

        self.stats.record_failure("llm_invalid_output")
        logger.error(
            "Agent failed to produce valid output record_id=%s attempts=%s last_error=%s",
            record.id, 1 + self.max_repairs, last_error,
        )
        return AgentOutcome(
            record=record,
            error=f"invalid LLM output after {1 + self.max_repairs} attempts: {last_error}",
            llm_calls=calls,
        )

    async def process_many(self, records: list[UnifiedRecord]) -> list[AgentOutcome]:
        """Process records sequentially (ordered, rate-limit friendly)."""
        outcomes: list[AgentOutcome] = []
        for record in records:
            try:
                outcomes.append(await self.process(record))
            except Exception as exc:  # noqa: BLE001 - one record must never kill the batch
                logger.error("Unexpected agent error record_id=%s error=%s", record.id, exc)
                outcomes.append(AgentOutcome(record=record, error=f"unexpected error: {exc}"))
        return outcomes

    @staticmethod
    def _log_decision(record: UnifiedRecord, result: AgentResult, provider_name: str) -> None:
        logger.info(
            "Agent classification=%s action=%s priority=%s channel=%s tone=%s "
            "record_id=%s processed_by=%s",
            result.classification,
            result.action,
            result.priority,
            result.channel,
            result.tone,
            record.id,
            provider_name,
        )
        logger.info("Rationale record_id=%s: %s", record.id, result.rationale)


def parse_agent_result(raw: str, record: UnifiedRecord) -> AgentResult:
    """Utility used by tests: decode + validate in one step."""
    payload = extract_json(raw)
    if not isinstance(payload, dict):
        raise ResultValidationError("model output was not a JSON object")
    result = validate_agent_result(record.id, record.source, record.record_type, payload)
    if not isinstance(result, AgentResult):  # pragma: no cover - defensive
        raise ValidationError  # type: ignore[misc]
    return result
