import json
from pathlib import Path

import httpx
import pytest

from tieout.net.cached_client import CachedClient, HostPolicy, HttpError, cache_path_for
from tieout.net.ratelimit import RateLimiter
from tieout.net.retry import RetryableError, retry_call


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_calls() -> None:
    clock = FakeClock()
    limiter = RateLimiter(5, clock=clock, sleep=clock.sleep)
    for _ in range(4):
        limiter.wait()
    # First call is free; each later call waits 1/5 s.
    assert clock.sleeps == pytest.approx([0.2, 0.2, 0.2])


def test_rate_limiter_no_wait_when_idle() -> None:
    clock = FakeClock()
    limiter = RateLimiter(5, clock=clock, sleep=clock.sleep)
    limiter.wait()
    clock.now += 10
    limiter.wait()
    assert clock.sleeps == []


def test_retry_gives_up_after_max_attempts() -> None:
    calls = []

    def boom() -> int:
        calls.append(1)
        raise RetryableError("429")

    with pytest.raises(RetryableError):
        retry_call(boom, max_attempts=3, sleep=lambda s: None)
    assert len(calls) == 3


def test_retry_honours_retry_after() -> None:
    sleeps: list[float] = []
    attempts = iter([RetryableError("429", retry_after=7), None])

    def flaky() -> str:
        err = next(attempts)
        if err:
            raise err
        return "ok"

    assert retry_call(flaky, sleep=sleeps.append) == "ok"
    assert sleeps == [7]


def test_cache_path_is_readable_and_safe(tmp_path: Path) -> None:
    p = cache_path_for(tmp_path, "https://data.sec.gov/api/xbrl/companyfacts/CIK0000066740.json")
    assert p == tmp_path / "data.sec.gov" / "api" / "xbrl" / "companyfacts" / "CIK0000066740.json"
    # path traversal segments are dropped
    evil = cache_path_for(tmp_path, "https://x.com/a/../../etc/passwd")
    assert tmp_path in evil.parents


def make_client(tmp_path: Path, handler: httpx.MockTransport, **policy: object) -> CachedClient:
    pol = HostPolicy(rate=1000, **policy)  # type: ignore[arg-type]
    return CachedClient(tmp_path, {"data.sec.gov": pol}, transport=handler, sleep=lambda s: None)


def test_client_caches_and_sends_user_agent(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    url = "https://data.sec.gov/submissions/CIK0000000001.json"
    with make_client(tmp_path, httpx.MockTransport(handler), user_agent="t a@b.co") as client:
        assert client.get_json(url) == {"ok": True}
        assert client.get_json(url) == {"ok": True}
        assert client.stats.network == 1
        assert client.stats.cache_hits == 1
    assert len(seen) == 1
    assert seen[0].headers["User-Agent"] == "t a@b.co"
    assert json.loads(cache_path_for(tmp_path, url).read_text()) == {"ok": True}


def test_client_retries_429_then_succeeds(tmp_path: Path) -> None:
    responses = iter([httpx.Response(429), httpx.Response(503), httpx.Response(200, text="x")])

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    with make_client(tmp_path, httpx.MockTransport(handler), user_agent="t a@b.co") as client:
        assert client.get_bytes("https://data.sec.gov/x") == b"x"
        assert client.stats.network == 3


def test_client_does_not_cache_or_retry_404(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    url = "https://data.sec.gov/missing"
    with make_client(tmp_path, httpx.MockTransport(handler), user_agent="t a@b.co") as client:
        with pytest.raises(HttpError):
            client.get_bytes(url)
        assert client.stats.network == 1
    assert not cache_path_for(tmp_path, url).exists()


def test_client_refuses_sec_without_user_agent(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not hit the network")

    transport = httpx.MockTransport(handler)
    with (
        make_client(tmp_path, transport, require_user_agent=True) as client,
        pytest.raises(RuntimeError, match="SEC_USER_AGENT"),
    ):
        client.get_bytes("https://data.sec.gov/x")
