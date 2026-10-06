"""Agent loop tests. No API key, database or embedding model needed.

- Loop logic with a scripted fake model and stub tools.
- Record/replay: the real LLMClient against a mock OpenAI-compatible server
  plus RecordedTool, then the same run offline, where any network or
  database access fails the test.
"""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

from tieout.agent.finance_tools import CalculateTool, SubmitAnswerTool
from tieout.agent.loop import FORCE_SUBMIT, NUDGE, AgentLimits, run_agent
from tieout.agent.recording import RecordedTool
from tieout.agent.tools import Tool, ToolInputError, chunk_evidence
from tieout.agent.validation import Evidence
from tieout.llm.client import LLMClient
from tieout.llm.limits import UsageLedger
from tieout.llm.types import ChatResponse, Message, ToolCall, Usage

CHUNK = "ch_be9a964b06a211ba"
FACT = "xf_77403cb0139a90aa"
CHUNK_TEXT = "Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420)"


class SearchArgs(BaseModel):
    query: str = Field(min_length=2)
    fiscal_year: int | None = None


class StubSearch(Tool):
    """Stands in for search_filings: one canned passage."""

    name = "search_filings"
    description = "Search filings."
    args_model = SearchArgs

    def __init__(self) -> None:
        self.calls = 0

    def run(self, args: SearchArgs) -> dict[str, Any]:
        self.calls += 1
        if args.query == "boom":
            raise ToolInputError("unknown ticker 'XYZ'")
        return {
            "results": [
                {
                    "chunk_id": CHUNK,
                    "ticker": "MMM",
                    "fiscal_year": 2018,
                    "page": 60,
                    "text": CHUNK_TEXT,
                }
            ]
        }

    def evidence(self, output: Any) -> list[Evidence]:
        return chunk_evidence(output["results"])


class FactsArgs(BaseModel):
    query: str


class StubFacts(Tool):
    name = "get_financial_facts"
    description = "XBRL facts."
    args_model = FactsArgs

    def run(self, args: FactsArgs) -> dict[str, Any]:
        return {"fact_id": FACT, "value": "1577000000"}

    def evidence(self, output: Any) -> list[Evidence]:
        restated = [{"value": "1600000000", "filed": "2020-02-06"}]
        return [
            Evidence(
                FACT,
                "fact",
                "Capex FY2018: 1577000000",
                [Decimal(1577000000)],
                {"fiscal_year": 2018, "restated": restated},
            )
        ]


def tools(*extra: Tool) -> list[Tool]:
    return [StubSearch(), *extra, CalculateTool(), SubmitAnswerTool()]


def resp(*calls: tuple[str, Any], content: str | None = None, tokens: int = 100) -> ChatResponse:
    tool_calls = [
        ToolCall(id=f"call_{i}", name=n, arguments=a if isinstance(a, str) else json.dumps(a))
        for i, (n, a) in enumerate(calls)
    ]
    return ChatResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=Usage(tokens, 0),
        model="fake",
        request_hash="x",
        cached=False,
        latency_ms=1.0,
    )


class ScriptedLLM:
    model = "fake"

    def __init__(self, script: list[ChatResponse]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        self.requests.append({"messages": [dict(m) for m in messages], **kwargs})
        return self.script.pop(0)


SEARCH = (
    "search_filings",
    {"query": "purchases of property, plant and equipment", "fiscal_year": 2018},
)


def submit(answer: str, quote: str = "(1,577)", source: str = CHUNK, **kw: Any) -> tuple[str, Any]:
    args = {
        "answer": answer,
        "confidence": 0.9,
        "rationale": "cash flow statement",
        "citations": [{"source_id": source, "quote": quote}],
        **kw,
    }
    return ("submit_answer", args)


def test_search_then_valid_submit_is_answered() -> None:
    llm = ScriptedLLM([resp(SEARCH), resp(submit("3M's FY2018 capex was $1,577 million."))])
    result = run_agent(llm, tools(), "sys", "3M FY2018 capex?")
    assert result.stop_reason == "submitted"
    assert result.final.status == "answered"
    assert result.final.citations[0]["source_id"] == CHUNK
    assert result.final.citations[0]["page"] == 60
    # The second request carries the assistant tool call and the tool result.
    second = llm.requests[1]["messages"]
    assert second[2]["tool_calls"][0]["function"]["name"] == "search_filings"
    assert second[3]["role"] == "tool" and CHUNK in second[3]["content"]


def test_llm_arithmetic_is_rejected_then_fixed_with_calculator() -> None:
    llm = ScriptedLLM(
        [
            resp(SEARCH),
            resp(submit("Capex rose by $204 million.", quote="(1,577) (1,373)")),
            resp(("calculate", {"expression": "1577 - 1373"})),
            resp(submit("Capex rose by $204 million.", quote="(1,577) (1,373)")),
        ]
    )
    result = run_agent(llm, tools(), "sys", "q")
    assert result.final.status == "answered" and result.rejections == 1
    rejection = json.loads(llm.requests[2]["messages"][-1]["content"])["error"]
    assert rejection.startswith("answer rejected:") and "calculate" in rejection


def test_repeated_invalid_submissions_go_to_review() -> None:
    bad = submit("It was $1,577 million.", source="ch_0000000000000000")
    llm = ScriptedLLM([resp(SEARCH), resp(bad), resp(bad)])
    result = run_agent(llm, tools(), "sys", "q")
    assert result.stop_reason == "validation_failed"
    assert result.final.status == "needs_review"
    assert result.final.review_reasons[0] == "answer failed validation 2 times"


@pytest.mark.parametrize(
    ("kw", "reason_part"),
    [
        ({"confidence": 0.4}, "below 0.70"),
        ({"needs_review": True, "review_reason": "10-K and XBRL disagree"}, "disagree"),
    ],
)
def test_low_confidence_or_flag_needs_review(kw: dict[str, Any], reason_part: str) -> None:
    llm = ScriptedLLM([resp(SEARCH), resp(submit("It was $1,577 million.", **kw))])
    result = run_agent(llm, tools(), "sys", "q")
    assert result.final.status == "needs_review"
    assert reason_part in result.final.review_reasons[0]


def test_plain_text_reply_is_nudged_to_submit() -> None:
    llm = ScriptedLLM(
        [resp(SEARCH), resp(content="It was $1,577M."), resp(submit("It was $1,577 million."))]
    )
    result = run_agent(llm, tools(), "sys", "q")
    assert result.steps[1].nudged
    assert llm.requests[2]["messages"][-1] == {"role": "user", "content": NUDGE}
    assert result.final.status == "answered"


def test_last_step_forces_submit_answer() -> None:
    llm = ScriptedLLM([resp(SEARCH), resp(SEARCH), resp(submit("It was $1,577 million."))])
    result = run_agent(llm, tools(), "sys", "q", AgentLimits(max_steps=3))
    assert [r["tool_choice"] for r in llm.requests] == ["auto", "auto", FORCE_SUBMIT]
    assert result.final.status == "answered" and result.steps[-1].forced_submit


def test_out_of_steps_without_answer_needs_review() -> None:
    llm = ScriptedLLM([resp(SEARCH)] * 2)  # ignores the forced submit
    result = run_agent(llm, tools(), "sys", "q", AgentLimits(max_steps=2))
    assert result.stop_reason == "max_steps"
    assert result.final.status == "needs_review" and result.final.answer is None


def test_token_budget_stops_the_run() -> None:
    llm = ScriptedLLM([resp(SEARCH, tokens=600), resp(SEARCH, tokens=600)])
    result = run_agent(llm, tools(), "sys", "q", AgentLimits(max_steps=5, max_tokens_per_run=1000))
    assert result.stop_reason == "token_budget" and len(result.steps) == 2
    assert result.final.status == "needs_review"


def test_restated_fact_becomes_a_warning_not_an_abstention() -> None:
    llm = ScriptedLLM(
        [
            resp(("get_financial_facts", {"query": "capex"})),
            resp(submit("It was $1,577 million.", quote="1577000000", source=FACT)),
        ]
    )
    result = run_agent(llm, tools(StubFacts()), "sys", "q")
    assert result.final.status == "answered"
    assert "later reported as 1600000000" in result.final.warnings[0]


def test_without_calculator_figures_are_not_checked() -> None:
    # Ablation config (no calculate tool): the rule could not be satisfied.
    llm = ScriptedLLM([resp(SEARCH), resp(submit("Capex rose by $204 million.", quote="(1,577)"))])
    result = run_agent(llm, [StubSearch(), SubmitAnswerTool()], "sys", "q")
    assert result.final.status == "answered"


@pytest.mark.parametrize(
    ("call", "error_part"),
    [
        (("search_filings", "{not json"), "not valid JSON"),
        (("search_filings", '{"query": "x"}'), "invalid arguments: query"),
        (("search_filings", '{"query": "boom"}'), "unknown ticker"),
        (("calculate", '{"expression": "__import__(1)"}'), "not allowed"),
        (("read_minds", "{}"), "unknown tool"),
    ],
)
def test_bad_tool_calls_are_returned_to_the_model(call: tuple[str, str], error_part: str) -> None:
    llm = ScriptedLLM([resp(call), resp(SEARCH), resp(submit("It was $1,577 million."))])
    result = run_agent(llm, tools(), "sys", "q")
    outcome = result.steps[0].tool_outcomes[0]
    assert outcome.error is not None and error_part in outcome.error
    assert json.loads(llm.requests[1]["messages"][3]["content"]) == {"error": outcome.error}
    assert result.final.status == "answered"


def test_tool_schema_has_no_pydantic_titles() -> None:
    schema = SubmitAnswerTool().schema()
    assert '"title"' not in json.dumps(schema)
    assert "citations" in schema["function"]["parameters"]["properties"]


# --- record / replay through LLMClient + RecordedTool ----------------------------


def mock_server() -> httpx.MockTransport:
    """OpenAI-compatible server: search first, then submit once it sees a tool result."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_tool = any(m["role"] == "tool" for m in body["messages"])
        name, args = submit("3M's FY2018 capex was $1,577 million.") if seen_tool else SEARCH
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }
        return httpx.Response(
            200,
            json={
                "model": "openai/gpt-oss-20b",
                "choices": [{"message": message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 40},
            },
        )

    return httpx.MockTransport(handler)


def llm_client(tmp_path: Path, transport: httpx.BaseTransport, offline: bool) -> LLMClient:
    return LLMClient(
        base_url="https://api.groq.com/openai/v1",
        api_key="",
        model="openai/gpt-oss-20b",
        provider="groq",
        cache_dir=tmp_path / "rec" / "llm",
        ledger=UsageLedger(tmp_path / "usage", today=lambda: date(2026, 10, 7)),
        offline=offline,
        transport=transport,
    )


class ExplodingSearch(StubSearch):
    def run(self, args: SearchArgs) -> dict[str, Any]:
        raise AssertionError("replay touched the database")


def test_record_then_replay_offline(tmp_path: Path) -> None:
    rec_tools = tmp_path / "rec" / "tools"
    live_tools = [RecordedTool(StubSearch(), rec_tools), CalculateTool(), SubmitAnswerTool()]
    live = run_agent(llm_client(tmp_path, mock_server(), False), live_tools, "sys", "3M capex?")
    assert live.final.status == "answered" and not live.cached

    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError("replay used the network")

    replay_tools = [
        RecordedTool(ExplodingSearch(), rec_tools, offline=True),
        CalculateTool(),
        SubmitAnswerTool(),
    ]
    replay_llm = llm_client(tmp_path, httpx.MockTransport(no_network), True)
    replay = run_agent(replay_llm, replay_tools, "sys", "3M capex?")

    assert replay.cached
    assert replay.final == live.final
    assert replay.usage == live.usage
    assert [s.response.request_hash for s in replay.steps] == [
        s.response.request_hash for s in live.steps
    ]
