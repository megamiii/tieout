import json
from pathlib import Path

from tieout.api.demo import WriteLimiter, load_demo_questions


def test_write_limiter_is_per_client_and_slides() -> None:
    limiter = WriteLimiter(2, window_s=60)
    assert limiter.allow("a", now=0) and limiter.allow("a", now=1)
    assert not limiter.allow("a", now=2)
    assert limiter.allow("b", now=2)  # another client has its own budget
    assert limiter.allow("a", now=60.5)  # the first hit left the window


def test_demo_questions_load_or_default_to_none(tmp_path: Path) -> None:
    assert load_demo_questions(tmp_path) == []
    (tmp_path / "questions.json").write_text(
        json.dumps([{"id": "q1", "question": "Q?", "shows": "s"}]), encoding="utf-8"
    )
    assert [q.id for q in load_demo_questions(tmp_path)] == ["q1"]
