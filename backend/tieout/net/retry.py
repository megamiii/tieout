"""Retry with exponential backoff and full jitter.

Shared by the SEC/GitHub downloader and (later) the LLM client, which both
need to survive 429 rate-limit responses from free-tier APIs.
"""

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RetryableError(Exception):
    """Raise inside a retried function to request another attempt.

    `retry_after` (seconds) comes from a server's Retry-After header, if any,
    and overrides the computed backoff delay.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Full-jitter backoff: uniform in [0, min(cap, base * 2**attempt)].

    Jitter spreads out retries so many clients that were throttled at the
    same moment don't all retry at the same moment.
    """
    return random.uniform(0, min(cap, base * 2**attempt))


def retry_call(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying on RetryableError up to `max_attempts` times."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except RetryableError as err:
            if attempt == max_attempts - 1:
                raise
            if err.retry_after is not None:
                delay = min(err.retry_after, max_delay)
            else:
                delay = backoff_delay(attempt, base_delay, max_delay)
            sleep(delay)
    raise AssertionError("unreachable")
