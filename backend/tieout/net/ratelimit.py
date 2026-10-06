"""A minimal thread-safe rate limiter: at most `rate` calls per second."""

import threading
import time
from collections.abc import Callable


class RateLimiter:
    """Enforces a minimum interval of 1/rate seconds between calls.

    This is stricter than a token bucket (no bursts), which is what we want
    for the SEC's "no more than 10 requests per second" rule.
    """

    def __init__(
        self,
        rate: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if now < self._next_allowed:
                self._sleep(self._next_allowed - now)
                now = self._next_allowed
            self._next_allowed = now + self.interval
