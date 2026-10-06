"""Agent loop tests. No API key, database or model needed.

Two kinds:
- loop logic, with a scripted fake model;
- record/replay: the real LLMClient against a mock OpenAI-compatible server
  (which writes the response cache), then the same run again offline, where
  any network call fails the test.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

from tieout.agent.loop import AgentLimits, run_agent
from tieout.agent.tools import Tool, ToolInputError
from tieout.llm.client import LLMClient
from tieout.llm.limits import UsageLedger
from tieout.llm.types import ChatResponse, Message, ToolCall, Usage


class LookupArgs(BaseModel):
    query: str = Field(min_length=2)
    fiscal_year: int | None = None


class StubSearch(Tool):
    """Stands in for search_filings: returns one canned passage."""

    name = "search_filings"
    description = "Search filings."
    args_model = LookupArgs

    def __init__(self) -> None:
        self.calls: list[LookupArgs] = []

    def run(self, args: LookupArgs) -> dict[str, Any]:
        self.calls.append(args)
        if args.query == "boom":
            raise ToolInputError("unknown ticker 'XYZ'")
        return {
            "results": [
                {
                    "chunk_id": "ch_be9a964b06a211ba",
                    "ticker": "MMM",
                    "fiscal_year": 2018,
                    "page": 60,
                    "text": "Purchases of property, plant and equipment (PP&E) (1,577) (1,373)",
                }
            ]
        }


def resp(
    content: str | None = None, calls: list[tuple[str, str]] | None = None, tokens: int = 100
) -> ChatResponse:
    tool_calls = [
        ToolCall(id=f"call_{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls or [])
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
    '{"query": "purchases of property, plant and equipment", "fiscal_year": 2018}',
)


def test_tool_call_then_answer() -> None:
    llm = ScriptedLLM([resp(calls=[SEARCH]), resp("Capex was $1,577M [ch_be9a964b06a211ba].")])
    tool = StubSearch()
    result = run_agent(llm, [tool], "sys", "3M FY2018 capex?")

    assert result.stop_reason == "answered"
    assert result.answer == "Capex was $1,577M [ch_be9a964b06a211ba]."
    assert tool.calls[0].fiscal_year == 2018
    # The second request carries the assistant tool call and the tool result.
    second = llm.requests[1]["messages"]
    assert second[2]["tool_calls"][0]["function"]["name"] == "search_filings"
    assert second[3]["role"] == "tool" and second[3]["tool_call_id"] == "call_0"
    assert "1,577" in second[3]["content"]
    assert result.usage.total_tokens == 200 and result.tool_calls == 1


def test_last_step_forces_an_answer() -> None:
    llm = ScriptedLLM([resp(calls=[SEARCH]), resp(calls=[SEARCH]), resp("best effort")])
    result = run_agent(llm, [StubSearch()], "sys", "q", AgentLimits(max_steps=3))
    assert [r["tool_choice"] for r in llm.requests] == ["auto", "auto", "none"]
    assert result.answer == "best effort" and result.steps[-1].forced_answer


def test_max_steps_without_answer() -> None:
    # A model that ignores tool_choice="none" still cannot loop forever.
    llm = ScriptedLLM([resp(calls=[SEARCH])] * 2)
    result = run_agent(llm, [StubSearch()], "sys", "q", AgentLimits(max_steps=2))
    assert result.stop_reason == "max_steps" and result.answer is None


def test_token_budget_stops_the_run() -> None:
    llm = ScriptedLLM([resp(calls=[SEARCH], tokens=600), resp(calls=[SEARCH], tokens=600)])
    result = run_agent(
        llm, [StubSearch()], "sys", "q", AgentLimits(max_steps=5, max_tokens_per_run=1000)
    )
    assert result.stop_reason == "token_budget" and len(result.steps) == 2


@pytest.mark.parametrize(
    ("call", "error_part"),
    [
        (("search_filings", "{not json"), "not valid JSON"),
        (("search_filings", '{"query": "x"}'), "invalid arguments: query"),
        (("search_filings", '{"query": "boom"}'), "unknown ticker"),
        (("read_minds", "{}"), "unknown tool"),
    ],
)
def test_bad_tool_calls_are_returned_to_the_model(call: tuple[str, str], error_part: str) -> None:
    llm = ScriptedLLM([resp(calls=[call]), resp("sorry")])
    result = run_agent(llm, [StubSearch()], "sys", "q")
    outcome = result.steps[0].tool_outcomes[0]
    assert outcome.error is not None and error_part in outcome.error
    tool_msg = llm.requests[1]["messages"][3]
    assert json.loads(tool_msg["content"]) == {"error": outcome.error}
    assert result.stop_reason == "answered"


def test_tool_schema_has_no_pydantic_titles() -> None:
    schema = StubSearch().schema()
    assert schema["function"]["name"] == "search_filings"
    assert "title" not in json.dumps(schema)


# --- record / replay through the real LLMClient --------------------------------


def openai_completion(message: dict[str, Any], prompt_tokens: int) -> dict[str, Any]:
    return {
        "model": "openai/gpt-oss-20b",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 30},
    }


def mock_server() -> httpx.MockTransport:
    """An OpenAI-compatible server: tool call first, answer once it sees a tool result."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not any(m["role"] == "tool" for m in body["messages"]):
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": SEARCH[0], "arguments": SEARCH[1]},
                    }
                ],
            }
            return httpx.Response(200, json=openai_completion(msg, 400))
        answer = "3M's FY2018 capital expenditure was $1,577 million [ch_be9a964b06a211ba]."
        return httpx.Response(
            200, json=openai_completion({"role": "assistant", "content": answer}, 700)
        )

    return httpx.MockTransport(handler)


def client(tmp_path: Path, transport: httpx.BaseTransport, offline: bool) -> LLMClient:
    return LLMClient(
        base_url="https://api.groq.com/openai/v1",
        api_key="",
        model="openai/gpt-oss-20b",
        provider="groq",
        cache_dir=tmp_path / "recording",
        ledger=UsageLedger(tmp_path / "usage", today=lambda: date(2026, 10, 7)),
        offline=offline,
        transport=transport,
    )


def test_record_then_replay_offline(tmp_path: Path) -> None:
    live = run_agent(client(tmp_path, mock_server(), False), [StubSearch()], "sys", "3M capex?")
    assert live.stop_reason == "answered" and not live.cached

    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError("replay used the network")

    replay_llm = client(tmp_path, httpx.MockTransport(no_network), True)
    replay = run_agent(replay_llm, [StubSearch()], "sys", "3M capex?")

    assert replay.cached
    assert replay.answer == live.answer
    assert replay.usage == live.usage  # budgets behave identically on replay
    assert [s.response.request_hash for s in replay.steps] == [
        s.response.request_hash for s in live.steps
    ]
    usage = replay_llm.ledger.read()["openai/gpt-oss-20b"]  # type: ignore[union-attr]
    assert usage["requests"] == 2 and usage["cache_hits"] == 2
