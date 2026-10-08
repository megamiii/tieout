"""OpenAI-compatible chat client with a response cache, rate limiting and retries.

Switch providers with one setting (LLM_PROVIDER): every supported provider
speaks the same /chat/completions API.

Caching: the request body (plus the provider URL) is serialized as canonical
JSON and hashed; the response is stored under that hash. Re-running the same
request costs no tokens. The cache doubles as a recording: with
`offline=True`, the client serves only recorded responses and never touches
the network, which is how tests (and the public demo) replay agent runs.
"""

import hashlib
import json
import os
import re
import ssl
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import truststore

from tieout.config import Settings
from tieout.llm.limits import FreeTierLimits, UsageLedger, WindowLimiter, limits_for
from tieout.llm.types import (
    ChatResponse,
    LLMCacheMiss,
    LLMError,
    Message,
    ToolCall,
    Usage,
)
from tieout.net.retry import RetryableError, retry_call

PROVIDER_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    # https://ai.google.dev/gemini-api/docs/openai (checked 2026-10-09)
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434/v1",
}
KEY_HELP = {
    "groq": "GROQ_API_KEY is not set (free key: https://console.groq.com)",
    "gemini": "GEMINI_API_KEY is not set (free key: https://aistudio.google.com/apikey)",
}
# Gemini sends its retry hint in the error body ("retryDelay": "37s"), not a header.
RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
RETRY_STATUSES = {429, 500, 502, 503, 504}
RATE_HEADER_PREFIX = "x-ratelimit-"


def retry_delay(header: str | None, body: str) -> float | None:
    """Seconds to wait from a Retry-After header or a Gemini-style retryDelay."""
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    m = RETRY_DELAY_RE.search(body)
    return float(m.group(1)) if m else None


def request_hash(base_url: str, body: dict[str, Any]) -> str:
    """Stable hash of a request: same model + messages + tools -> same hash."""
    canonical = json.dumps({"base_url": base_url, "body": body}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def estimate_tokens(body: dict[str, Any]) -> int:
    """Rough prompt size (~4 characters per token) plus the completion cap."""
    prompt = len(json.dumps(body["messages"])) + len(json.dumps(body.get("tools", [])))
    return prompt // 4 + int(body.get("max_tokens") or 1024)


def parse_response(
    raw: dict[str, Any], *, request_hash_: str, cached: bool, latency_ms: float
) -> ChatResponse:
    choice = raw["choices"][0]
    msg = choice["message"]
    calls = [
        ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=tc["function"]["arguments"])
        for tc in (msg.get("tool_calls") or [])
    ]
    usage = raw.get("usage") or {}
    return ChatResponse(
        content=msg.get("content"),
        tool_calls=calls,
        finish_reason=choice.get("finish_reason"),
        usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
        model=raw.get("model", ""),
        request_hash=request_hash_,
        cached=cached,
        latency_ms=latency_ms,
        reasoning=msg.get("reasoning"),
        raw=raw,
    )


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        provider: str,
        cache_dir: Path,
        ledger: UsageLedger | None = None,
        limiter: WindowLimiter | None = None,
        offline: bool = False,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider
        self.cache_dir = cache_dir
        self.ledger = ledger
        self.limiter = limiter
        self.offline = offline
        self._sleep = sleep
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        verify: ssl.SSLContext = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._http = httpx.Client(
            timeout=timeout, headers=headers, transport=transport, verify=verify
        )

    def close(self) -> None:
        self._http.close()

    def _cache_path(self, h: str) -> Path:
        return self.cache_dir / h[:2] / f"{h}.json"

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,
    ) -> ChatResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice
        if extra:
            body.update(extra)
        h = request_hash(self.base_url, body)
        path = self._cache_path(h)

        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))["response"]
            resp = parse_response(raw, request_hash_=h, cached=True, latency_ms=0.0)
            if self.ledger:
                self.ledger.add(self.model, resp.usage, cached=True)
            return resp
        if self.offline:
            raise LLMCacheMiss(f"request {h[:12]} not in recorded cache {self.cache_dir}")

        if self.limiter:
            self.limiter.wait_for(estimate_tokens(body))
        start = time.perf_counter()
        raw, rate_headers = retry_call(lambda: self._post(body), sleep=self._sleep)
        latency_ms = (time.perf_counter() - start) * 1000
        resp = parse_response(raw, request_hash_=h, cached=False, latency_ms=latency_ms)
        if self.limiter:
            self.limiter.record(resp.usage.total_tokens)
        if self.ledger:
            self.ledger.add(self.model, resp.usage, cached=False, rate_headers=rate_headers)
        self._write_cache(path, body, raw)
        return resp

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        try:
            r = self._http.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.TransportError as err:
            raise RetryableError(f"transport error: {err}") from err
        rate_headers = {
            k: v for k, v in r.headers.items() if k.lower().startswith(RATE_HEADER_PREFIX)
        }
        if r.status_code in RETRY_STATUSES:
            delay = retry_delay(r.headers.get("retry-after"), r.text)
            raise RetryableError(f"HTTP {r.status_code}: {r.text[:200]}", retry_after=delay)
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
        data: dict[str, Any] = r.json()
        return data, rate_headers

    def _write_cache(self, path: Path, body: dict[str, Any], raw: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"provider": self.provider, "request": body, "response": raw}
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".part-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1, ensure_ascii=False)
        os.replace(tmp, path)


def configured_limits(settings: Settings) -> FreeTierLimits | None:
    overrides = FreeTierLimits(
        rpm=settings.llm_rpm, rpd=settings.llm_rpd, tpm=settings.llm_tpm, tpd=settings.llm_tpd
    )
    return limits_for(settings.llm_provider, settings.llm_model, overrides)


def make_llm_client(
    settings: Settings, *, offline: bool = False, cache_dir: Path | None = None
) -> LLMClient:
    provider = settings.llm_provider
    base_url = settings.llm_base_url or PROVIDER_URLS.get(provider, "")
    if not base_url:
        raise LLMError(f"set LLM_BASE_URL for provider {provider!r}")
    api_key = {"groq": settings.groq_api_key, "gemini": settings.gemini_api_key}.get(
        provider, settings.llm_api_key
    )
    if provider in KEY_HELP and not api_key and not offline:
        raise LLMError(KEY_HELP[provider])
    limits = configured_limits(settings)
    limiter = WindowLimiter(limits.rpm, limits.tpm) if limits else None
    return LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=settings.llm_model,
        provider=provider,
        cache_dir=cache_dir or settings.llm_cache_dir,
        ledger=UsageLedger(settings.usage_dir),
        limiter=limiter,
        offline=offline,
    )
