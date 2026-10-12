import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from tieout.agent.finance_tools import CalculateTool, SubmitAnswerTool
from tieout.agent.loop import run_agent
from tieout.llm.client import LLMClient
from tieout.llm.limits import UsageLedger
from tieout.tracing.store import json_safe
from tieout.tracing.tracer import annotate, current_tracer, span, tracing


def test_spans_nest_and_record_errors() -> None:
    with tracing("agent", "q") as tr, span("agent", "run") as root:
        with span("step", "step 1") as step, span("tool", "search", input={"q": "capex"}) as tool:
            annotate(cached=True)
            tool.output = {"hits": 3}
        with pytest.raises(ValueError), span("tool", "boom"):
            raise ValueError("bad input")
    assert current_tracer() is None  # deactivated after the block
    names = [(s.type, s.name, s.parent_id) for s in tr.spans]
    assert names == [
        ("agent", "run", None),
        ("step", "step 1", root.span_id),
        ("tool", "search", step.span_id),
        ("tool", "boom", root.span_id),
    ]
    assert tr.spans[2].meta == {"cached": True} and tr.spans[2].output == {"hits": 3}
    assert tr.spans[3].error == "ValueError: bad input"
    assert all(s.duration_ms >= 0 for s in tr.spans)


def test_spans_are_noops_without_a_tracer() -> None:
    with span("tool", "search") as s:
        s.output = {"ignored": True}
        annotate(x=1)  # must not fail
    assert current_tracer() is None


def test_json_safe_handles_decimals_dates_and_size() -> None:
    from decimal import Decimal

    assert json_safe({"v": Decimal("1.5"), "d": date(2022, 1, 1)}) == {
        "v": "1.5",
        "d": "2022-01-01",
    }
    big = json_safe({"text": "x" * 50_000})
    assert big["truncated"] is True and len(big["preview"]) == 20_000


# --- a recorded agent run through the real LLM client -------------------------


def mock_server() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_tool = any(m["role"] == "tool" for m in body["messages"])
        if seen_tool:
            name, args = (
                "submit_answer",
                {
                    "answer": "Capex rose by $204 million.",
                    "confidence": 0.9,
                    "rationale": "r",
                    "citations": [],
                    "needs_review": True,
                    "review_reason": "no sources in this test",
                },
            )
        else:
            name, args = "calculate", {"expression": "1577 - 1373"}
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": msg, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 20},
            },
        )

    return httpx.MockTransport(handler)


def client(tmp_path: Path) -> LLMClient:
    return LLMClient(
        base_url="https://api.example.test/v1",
        api_key="",
        model="m",
        provider="gemini",
        cache_dir=tmp_path / "llm",
        ledger=UsageLedger(tmp_path / "usage", today=lambda: date(2026, 10, 13)),
        transport=mock_server(),
    )


def test_agent_run_produces_a_span_tree(tmp_path: Path) -> None:
    tools = [CalculateTool(), SubmitAnswerTool()]
    with tracing("agent", "capex?") as tr:
        result = run_agent(client(tmp_path), tools, "sys", "How much did capex rise?")
    assert result.final.status == "needs_review"
    tree = [(s.type, s.name) for s in tr.spans]
    assert tree == [
        ("agent", "agent run"),
        ("step", "step 1"),
        ("llm", "m"),
        ("tool", "calculate"),
        ("step", "step 2"),
        ("llm", "m"),
        ("tool", "submit_answer"),
        ("validation", "validate submit_answer"),
    ]
    first_llm, second_llm = (s for s in tr.spans if s.type == "llm")
    # The first call records the whole conversation; later calls only what's new.
    assert [m["role"] for m in first_llm.input["new_messages"]] == ["system", "user"]
    assert [m["role"] for m in second_llm.input["new_messages"]] == ["tool"]
    assert second_llm.input["conversation_offset"] == 3
    assert second_llm.prompt_tokens == 300 and not second_llm.cached
    validation = tr.spans[-1]
    assert validation.output["accepted"] is True
    totals = tr.totals()
    assert totals["llm_requests"] == 2 and totals["tool_calls"] == 2 and totals["errors"] == 0


def test_replayed_run_marks_llm_spans_cached(tmp_path: Path) -> None:
    tools: list[Any] = [CalculateTool(), SubmitAnswerTool()]
    run_agent(client(tmp_path), tools, "sys", "How much did capex rise?")  # records the cache
    with tracing("agent", "replay") as tr:
        run_agent(client(tmp_path), tools, "sys", "How much did capex rise?")
    llm = [s for s in tr.spans if s.type == "llm"]
    assert llm and all(s.cached for s in llm)
    assert tr.totals()["llm_requests"] == 0 and tr.totals()["cached"] == 2
