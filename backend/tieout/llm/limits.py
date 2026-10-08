"""Free-tier rate limits: a client-side limiter and a daily usage ledger."""

import json
import os
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from tieout.llm.types import Usage


@dataclass(frozen=True)
class FreeTierLimits:
    """None means unknown / not enforced."""

    rpm: int | None  # requests per minute
    rpd: int | None  # requests per day
    tpm: int | None  # tokens per minute
    tpd: int | None  # tokens per day


# Groq free tier, from https://console.groq.com/docs/rate-limits (checked 2026-10-07).
GROQ_FREE_TIER = {
    "openai/gpt-oss-20b": FreeTierLimits(rpm=30, rpd=1_000, tpm=8_000, tpd=200_000),
    "openai/gpt-oss-120b": FreeTierLimits(rpm=30, rpd=1_000, tpm=8_000, tpd=200_000),
}


# Daily quotas reset at midnight in the provider's timezone, not the user's.
# Gemini: midnight Pacific (https://ai.google.dev/gemini-api/docs/rate-limits).
QUOTA_TIMEZONES = {"gemini": "America/Los_Angeles", "groq": "UTC"}


def quota_day(timezone: str | None) -> Callable[[], date]:
    """'Today' for quota accounting: in `timezone`, or the local date if None."""
    if not timezone:
        return date.today
    tz = ZoneInfo(timezone)
    return lambda: datetime.now(tz).date()


def limits_for(
    provider: str, model: str, overrides: FreeTierLimits | None = None
) -> FreeTierLimits | None:
    """Limits for a provider/model: configured overrides first, then built-in presets.

    Gemini's free-tier limits are shown per project in AI Studio rather than
    published, so they come from LLM_RPM / LLM_TPM / LLM_RPD / LLM_TPD.
    Returns None when nothing is known (e.g. local Ollama has no limits).
    """
    preset = GROQ_FREE_TIER.get(model) if provider == "groq" else None
    if overrides is None or not any((overrides.rpm, overrides.tpm, overrides.rpd, overrides.tpd)):
        return preset
    return FreeTierLimits(
        rpm=overrides.rpm or (preset.rpm if preset else None),
        rpd=overrides.rpd or (preset.rpd if preset else None),
        tpm=overrides.tpm or (preset.tpm if preset else None),
        tpd=overrides.tpd or (preset.tpd if preset else None),
    )


class WindowLimiter:
    """Waits *before* a request that would exceed requests/min or tokens/min.

    Keeps the (time, tokens) of requests in the last 60 s. Proactive waiting
    avoids most 429s; the client still retries if one happens anyway.
    """

    WINDOW = 60.0

    def __init__(
        self,
        rpm: int | None,
        tpm: int | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rpm = rpm if rpm else 10**9  # unknown: don't throttle on this axis
        self.tpm = tpm if tpm else 10**12
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= self.WINDOW:
            self._events.popleft()

    def wait_for(self, estimated_tokens: int) -> float:
        """Block until a request of ~estimated_tokens fits. Returns seconds waited."""
        waited = 0.0
        with self._lock:
            while True:
                now = self._clock()
                self._prune(now)
                used = sum(t for _, t in self._events)
                fits_requests = len(self._events) < self.rpm
                # A request bigger than the whole budget can only run in an empty window.
                fits_tokens = used + estimated_tokens <= self.tpm or not self._events
                if fits_requests and fits_tokens:
                    return waited
                delay = self.WINDOW - (now - self._events[0][0]) + 0.05
                self._sleep(delay)
                waited += delay

    def record(self, tokens: int) -> None:
        with self._lock:
            self._events.append((self._clock(), tokens))


class UsageLedger:
    """Per-day token and request counts in data/usage/YYYY-MM-DD.json.

    The day is the provider's quota day (see quota_day), so "used today"
    matches what the provider counts against the daily limit.
    """

    def __init__(self, directory: Path, today: Callable[[], date] = date.today) -> None:
        self.directory = directory
        self._today = today
        self._lock = threading.Lock()

    def _path(self, day: date) -> Path:
        return self.directory / f"{day.isoformat()}.json"

    def read(self, day: date | None = None) -> dict[str, Any]:
        path = self._path(day or self._today())
        if not path.exists():
            return {}
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def add(
        self,
        model: str,
        usage: Usage,
        *,
        cached: bool,
        rate_headers: dict[str, str] | None = None,
    ) -> None:
        with self._lock:
            day = self._today()
            data = self.read(day)
            m = data.setdefault(
                model,
                {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_hits": 0,
                    "tokens_saved_by_cache": 0,
                },
            )
            if cached:
                m["cache_hits"] += 1
                m["tokens_saved_by_cache"] += usage.total_tokens
            else:
                m["requests"] += 1
                m["prompt_tokens"] += usage.prompt_tokens
                m["completion_tokens"] += usage.completion_tokens
            if rate_headers:
                m["last_rate_limit_headers"] = rate_headers
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".part-")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path(day))
