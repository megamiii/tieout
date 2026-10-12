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


def test_replay_uses_the_recorded_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Settings] = []

    def capture(experiment: str, settings: Settings, *a: Any, **k: Any) -> Any:
        seen.append(settings)
        return fake_answer, None

    monkeypatch.setattr(runner, "make_answerer", capture)
    monkeypatch.setattr(runner, "load_result", lambda exp, item_id: None)
    local = Settings(llm_provider="groq", llm_model="openai/gpt-oss-20b", agent_max_steps=5)
    runner.verify_replay("agent_full", [item("x")], local, [])
    assert seen[0].llm_provider == "gemini" and seen[0].agent_max_steps == 8


def test_recorded_configuration_matches_the_eval_report() -> None:
    import json

    from tieout.config import REPO_ROOT

    report = json.loads((REPO_ROOT / "evals" / "reports" / "eval.json").read_text("utf-8"))
    assert report["meta"]["agent_model"] == runner.RECORDED_WITH["llm_model"]
