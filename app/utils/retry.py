"""Bounded retry helpers with exponential backoff.

Design rules:

* Only *retryable* failures are retried (timeouts, connection errors, 429/5xx).
* Validation / authentication / 4xx failures fail fast - retrying them is
  pointless and hides bugs.
* Every attempt is logged so operators can see failures and recoveries.
* The number of attempts is always bounded.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from app.utils.logger import get_logger

T = TypeVar("T")
logger = get_logger("retry")


class RetryableError(Exception):
    """A transient failure that is safe to retry (timeout, 5xx, 429...)."""


class NonRetryableError(Exception):
    """A permanent failure (validation, auth, 404...). Do not retry."""


@dataclass
class RetryPolicy:
    """Configuration for bounded exponential backoff."""

    attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 8.0
    jitter: float = 0.3
    #: Upper bound for a server-supplied retry hint (e.g. a 429 RetryInfo).
    max_server_delay: float = 60.0

    def delay_for(self, attempt: int) -> float:
        """Delay (seconds) before the given 1-based retry attempt."""
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        if self.jitter:
            delay += random.uniform(0, self.jitter)
        return delay


@dataclass
class RetryStats:
    """Mutable counters shared across the pipeline run."""

    attempts: int = 0
    recoveries: int = 0
    exhausted: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    def record_failure(self, label: str) -> None:
        self.failures[label] = self.failures.get(label, 0) + 1


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    description: str,
    stats: RetryStats | None = None,
    retryable: tuple[type[BaseException], ...] = (RetryableError,),
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``operation`` with bounded retries and exponential backoff.

    Raises the last encountered exception once attempts are exhausted.
    """
    last_error: BaseException | None = None

    for attempt in range(1, policy.attempts + 1):
        try:
            result = await operation()
            if attempt > 1 and stats is not None:
                stats.recoveries += 1
                logger.info("Request recovered on retry attempt=%s op=%s", attempt, description)
            return result
        except retryable as exc:  # noqa: PERF203 - bounded loop
            last_error = exc
            if stats is not None:
                stats.attempts += 1
                stats.record_failure(description)
            logger.error(
                "%s failed attempt=%s/%s error=%s",
                description,
                attempt,
                policy.attempts,
                exc,
            )
            if attempt >= policy.attempts:
                if stats is not None:
                    stats.exhausted += 1
                logger.error("Giving up on %s after %s attempts", description, policy.attempts)
                break
            delay = policy.delay_for(attempt)
            # Honour a server-supplied hint (e.g. a 429 RetryInfo) when it asks
            # for a longer wait than our own backoff, capped to stay bounded.
            hint = getattr(exc, "retry_after", None)
            if isinstance(hint, (int, float)) and not isinstance(hint, bool) and hint > 0:
                delay = min(max(delay, float(hint)), policy.max_server_delay)
            logger.warning(
                "Retrying request description=%s sleep=%.2fs", description, delay
            )
            await sleep(delay)
        except NonRetryableError:
            raise

    assert last_error is not None
    raise last_error
