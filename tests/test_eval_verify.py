from typing import Any

import pytest

from tieout.config import Settings
from tieout.evals import runner
from tieout.evals.sets import EvalItem
from tieout.llm.types import LLMCacheMiss


def item(item_id: str) -> EvalItem:
    return EvalItem(
        id=item_id,
        eval_set="xbrl_numeric",
        question=f"{item_id}?",
        answer="1",
        answer_kind="numeric",
        ticker="MMM",
        fiscal_year=2018,
        doc_name="3M_2018_10K",
    )


COMMITTED = {
    "same": {"status": "answered", "answer": "A", "stop_reason": "submitted", "citations": []},
    "changed": {"status": "answered", "answer": "A", "stop_reason": "submitted", "citations": []},
    "unrecorded": {"status": "answered", "answer": "A", "stop_reason": "submitted"},
}


def fake_answer(it: EvalItem) -> dict[str, Any]:
    if it.id == "unrecorded":
        raise LLMCacheMiss("request abc not in recorded cache")
    answer = "B" if it.id == "changed" else "A"
    return {"status": "answered", "answer": answer, "stop_reason": "submitted", "citations": []}


def test_verify_reports_changed_and_unrecorded_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "make_answerer", lambda *a, **k: (fake_answer, None))
    monkeypatch.setattr(runner, "load_result", lambda exp, item_id: COMMITTED.get(item_id))
    items = [item(i) for i in ("same", "changed", "unrecorded", "missing")]
    replays = runner.verify_replay("agent_full", items, Settings(), [])
    assert [(r.item_id, r.outcome) for r in replays] == [
        ("same", "same"),
        ("changed", "different"),
        ("unrecorded", "not_recorded"),
        ("missing", "no_result"),
    ]
    assert replays[1].detail == "answer"
