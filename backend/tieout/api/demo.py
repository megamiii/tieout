"""Public demo mode: curated questions, replay-only, bounded writes.

In demo mode the LLM client and the database tools run offline against one
recording directory, so a request replays exactly if it was recorded and
fails with a cache miss if it wasn't. There is no per-question special
casing: the curated questions are simply the ones that were recorded.
"""

import json
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Request
from pydantic import BaseModel

NOT_RECORDED = (
    "This public demo only replays recorded runs, so it needs no API key and costs "
    "nothing. This request was not recorded: pick one of the demo questions, or run "
    "tieout locally with a free Gemini or Groq key to ask anything."
)


class DemoQuestion(BaseModel):
    id: str
    question: str
    shows: str  # what the run demonstrates, shown in the UI


def questions_path(recording_dir: Path) -> Path:
    return recording_dir / "questions.json"


def load_demo_questions(recording_dir: Path) -> list[DemoQuestion]:
    path = questions_path(recording_dir)
    if not path.exists():
        return []
    return [DemoQuestion.model_validate(q) for q in json.loads(path.read_text(encoding="utf-8"))]


def client_key(request: Request) -> str:
    """The client's address; behind a proxy (Hugging Face Spaces) the first forwarded one."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class WriteLimiter:
    """At most `limit` writes per client per sliding hour (in memory, per process)."""

    def __init__(self, limit: int, window_s: float = 3600.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True
