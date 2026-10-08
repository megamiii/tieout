import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from tieout.config import Settings
from tieout.llm.client import LLMClient, make_llm_client, request_hash, retry_delay
from tieout.llm.limits import FreeTierLimits, UsageLedger, WindowLimiter, limits_for
from tieout.llm.types import LLMCacheMiss, LLMError, Usage


def completion(content: str = "hi", tool_calls: list[dict[str, Any]] | None = None) -> dict:
    return {
        "model": "openai/gpt-oss-20b",
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def make_client(
    tmp_path: Path, handler: Any, *, offline: bool = False, sleeps: list[float] | None = None
) -> LLMClient:
    return LLMClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="openai/gpt-oss-20b",
        provider="groq",
        cache_dir=tmp_path / "llm",
        ledger=UsageLedger(tmp_path / "usage", today=lambda: date(2026, 10, 7)),
        offline=offline,
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


MSGS = [{"role": "user", "content": "What was 3M's FY2018 capex?"}]


def test_second_identical_request_is_served_from_cache(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=completion("answer"))

    client = make_client(tmp_path, handler)
    first = client.chat(MSGS)
    second = client.chat(MSGS)
    assert len(calls) == 1
    assert calls[0].headers["Authorization"] == "Bearer test-key"
    assert (first.cached, second.cached) == (False, True)
    assert second.content == "answer" and second.usage == Usage(100, 20)

    ledger = client.ledger.read()  # type: ignore[union-attr]
    m = ledger["openai/gpt-oss-20b"]
    assert m["requests"] == 1 and m["prompt_tokens"] == 100 and m["completion_tokens"] == 20
    assert m["cache_hits"] == 1 and m["tokens_saved_by_cache"] == 120


def test_cache_file_never_contains_the_api_key(tmp_path: Path) -> None:
    client = make_client(tmp_path, lambda r: httpx.Response(200, json=completion()))
    client.chat(MSGS)
    files = list((tmp_path / "llm").rglob("*.json"))
    assert len(files) == 1
    assert "test-key" not in files[0].read_text(encoding="utf-8")


def test_request_hash_ignores_key_order_but_not_content() -> None:
    a = {"model": "m", "messages": MSGS, "temperature": 0}
    b = {"temperature": 0, "messages": MSGS, "model": "m"}
    assert request_hash("u", a) == request_hash("u", b)
    assert request_hash("u", a) != request_hash("u", {**a, "temperature": 1})
    assert request_hash("u", a) != request_hash("other-provider", a)


def test_429_is_retried_after_retry_after_seconds(tmp_path: Path) -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"retry-after": "7"}, text="rate limited"),
            httpx.Response(
                200,
                json=completion("ok"),
                headers={"x-ratelimit-remaining-tokens": "7880"},
            ),
        ]
    )
    sleeps: list[float] = []
    client = make_client(tmp_path, lambda r: next(responses), sleeps=sleeps)
    assert client.chat(MSGS).content == "ok"
    assert sleeps == [7.0]
    m = client.ledger.read()["openai/gpt-oss-20b"]  # type: ignore[union-attr]
    assert m["last_rate_limit_headers"] == {"x-ratelimit-remaining-tokens": "7880"}


def test_auth_error_is_not_retried(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="invalid api key")

    with pytest.raises(LLMError, match="401"):
        make_client(tmp_path, handler).chat(MSGS)
    assert calls == [1]


def test_offline_mode_never_touches_the_network(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network used in offline mode")

    with pytest.raises(LLMCacheMiss):
        make_client(tmp_path, handler, offline=True).chat(MSGS)


def test_request_body_shape(tmp_path: Path) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    tool = {"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}
    make_client(tmp_path, handler).chat(
        MSGS, tools=[tool], tool_choice="none", extra={"reasoning_effort": "low"}
    )
    body = seen[0]
    assert body["tools"] == [tool] and body["tool_choice"] == "none"
    assert body["temperature"] == 0.0 and body["reasoning_effort"] == "low"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def test_window_limiter_waits_for_token_budget() -> None:
    clock = FakeClock()
    lim = WindowLimiter(rpm=30, tpm=8000, clock=clock, sleep=clock.sleep)
    assert lim.wait_for(5000) == 0
    lim.record(5000)
    clock.now = 10
    # 5000 used in the window; 4000 more would exceed 8000 -> wait until t=60
    waited = lim.wait_for(4000)
    assert waited == pytest.approx(50.05)
    lim.record(4000)


def test_window_limiter_waits_for_request_budget() -> None:
    clock = FakeClock()
    lim = WindowLimiter(rpm=2, tpm=10**9, clock=clock, sleep=clock.sleep)
    for _ in range(2):
        lim.wait_for(1)
        lim.record(1)
    assert lim.wait_for(1) == pytest.approx(60.05)


def test_oversized_request_runs_alone() -> None:
    clock = FakeClock()
    lim = WindowLimiter(rpm=30, tpm=8000, clock=clock, sleep=clock.sleep)
    assert lim.wait_for(20_000) == 0  # empty window: allowed (the API decides)


def test_gemini_retry_delay_in_body_is_honoured(tmp_path: Path) -> None:
    gemini_429 = (
        '[{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": '
        '[{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}]}}]'
    )
    responses = iter([httpx.Response(429, text=gemini_429), httpx.Response(200, json=completion())])
    sleeps: list[float] = []
    make_client(tmp_path, lambda r: next(responses), sleeps=sleeps).chat(MSGS)
    assert sleeps == [37.0]


def test_retry_delay_parsing() -> None:
    assert retry_delay("7", "") == 7.0
    assert retry_delay(None, '{"retryDelay": "1.5s"}') == 1.5
    assert retry_delay("not-a-number", "no hint") is None


def test_configured_limits_override_presets() -> None:
    preset = limits_for("groq", "openai/gpt-oss-20b")
    assert preset is not None and preset.tpd == 200_000
    # Gemini has no built-in preset: limits come from .env (copied from AI Studio).
    assert limits_for("gemini", "gemini-3.8-flash") is None
    custom = limits_for("gemini", "gemini-3.8-flash", FreeTierLimits(rpm=10, rpd=0, tpm=0, tpd=0))
    assert custom == FreeTierLimits(rpm=10, rpd=None, tpm=None, tpd=None)
    # A partial override keeps the rest of a preset.
    mixed = limits_for("groq", "openai/gpt-oss-20b", FreeTierLimits(rpm=5, rpd=0, tpm=0, tpd=0))
    assert mixed == FreeTierLimits(rpm=5, rpd=1_000, tpm=8_000, tpd=200_000)


def test_unknown_limits_do_not_throttle() -> None:
    clock = FakeClock()
    lim = WindowLimiter(rpm=None, tpm=None, clock=clock, sleep=clock.sleep)
    for _ in range(100):
        assert lim.wait_for(50_000) == 0
        lim.record(50_000)


def test_gemini_preset_requires_its_key() -> None:
    settings = Settings(llm_provider="gemini", llm_model="gemini-3.8-flash", gemini_api_key="")
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        make_llm_client(settings)
    client = make_llm_client(settings, offline=True)  # replay needs no key
    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_provider_extras_on_tool_calls_are_echoed_back(tmp_path: Path) -> None:
    signature = {"google": {"thought_signature": "c2lnbmF0dXJl"}}
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search_filings", "arguments": "{}"},
        "extra_content": signature,
    }
    client = make_client(
        tmp_path, lambda r: httpx.Response(200, json=completion("", tool_calls=[tool_call]))
    )
    msg = client.chat(MSGS).assistant_message()
    assert msg["tool_calls"][0]["extra_content"] == signature
    assert msg["tool_calls"][0]["function"]["name"] == "search_filings"
