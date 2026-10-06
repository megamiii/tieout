"""HTTP GET with a permanent disk cache, per-host rate limits and retries.

Every file we download (SEC JSON, FinanceBench data and PDFs) goes through
this client, so each URL is fetched from the network at most once.
"""

import json
import os
import ssl
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import truststore

from tieout.net.ratelimit import RateLimiter
from tieout.net.retry import RetryableError, retry_call

SEC_HOSTS = ("sec.gov", "www.sec.gov", "data.sec.gov")
RETRY_STATUSES = {429, 500, 502, 503, 504}


class HttpError(Exception):
    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.url = url
        self.status = status


@dataclass
class HostPolicy:
    rate: float  # max requests per second
    user_agent: str | None = None
    require_user_agent: bool = False


@dataclass
class Stats:
    network: int = 0
    cache_hits: int = 0


def cache_path_for(cache_dir: Path, url: str) -> Path:
    """Map a URL to a readable, deterministic path under `cache_dir`.

    https://data.sec.gov/api/xbrl/companyfacts/CIK0000066740.json
      -> <cache_dir>/data.sec.gov/api/xbrl/companyfacts/CIK0000066740.json
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"not an http(s) URL: {url}")
    path = parts.path.lstrip("/") or "index"
    if path.endswith("/"):
        path += "index"
    if parts.query:
        safe_query = "".join(c if c.isalnum() or c in "-_=." else "_" for c in parts.query)
        path += f"__{safe_query}"
    segments = [s for s in path.split("/") if s not in ("", ".", "..")]
    return cache_dir.joinpath(parts.netloc, *segments)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via a temp file + rename, so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".part-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class CachedClient:
    def __init__(
        self,
        cache_dir: Path,
        policies: dict[str, HostPolicy],
        *,
        default_policy: HostPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.policies = policies
        self.default_policy = default_policy or HostPolicy(rate=2.0)
        self.stats = Stats()
        self._sleep = sleep
        self._limiters: dict[str, RateLimiter] = {}
        verify: ssl.SSLContext | bool = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._http = httpx.Client(
            timeout=timeout, follow_redirects=True, transport=transport, verify=verify
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "CachedClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _policy(self, host: str) -> HostPolicy:
        return self.policies.get(host, self.default_policy)

    def _limiter(self, host: str) -> RateLimiter:
        # One limiter per *policy*, not per host: www.sec.gov and data.sec.gov
        # share the SEC policy, and the SEC's 10 req/s limit is per client.
        policy = self._policy(host)
        key = str(id(policy))
        if key not in self._limiters:
            if self._sleep is not None:
                self._limiters[key] = RateLimiter(policy.rate, sleep=self._sleep)
            else:
                self._limiters[key] = RateLimiter(policy.rate)
        return self._limiters[key]

    def _fetch(self, url: str) -> bytes:
        host = urlsplit(url).netloc
        policy = self._policy(host)
        headers = {"Accept-Encoding": "gzip, deflate"}
        if policy.user_agent:
            headers["User-Agent"] = policy.user_agent
        elif policy.require_user_agent:
            raise RuntimeError(
                f"{host} requires a User-Agent with a contact email; set SEC_USER_AGENT in .env"
            )

        def attempt() -> bytes:
            self._limiter(host).wait()
            self.stats.network += 1
            try:
                resp = self._http.get(url, headers=headers)
            except httpx.TransportError as err:
                raise RetryableError(f"transport error for {url}: {err}") from err
            if resp.status_code in RETRY_STATUSES:
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else None
                raise RetryableError(f"HTTP {resp.status_code} for {url}", retry_after=delay)
            if resp.status_code != 200:
                raise HttpError(url, resp.status_code)
            return resp.content

        if self._sleep is not None:
            return retry_call(attempt, sleep=self._sleep)
        return retry_call(attempt)

    def path(self, url: str, *, refresh: bool = False) -> Path:
        """Return the local path of `url`, downloading it only if not cached."""
        path = cache_path_for(self.cache_dir, url)
        if path.exists() and not refresh:
            self.stats.cache_hits += 1
            return path
        _atomic_write(path, self._fetch(url))
        return path

    def get_bytes(self, url: str, *, refresh: bool = False) -> bytes:
        return self.path(url, refresh=refresh).read_bytes()

    def get_json(self, url: str, *, refresh: bool = False) -> Any:
        return json.loads(self.get_bytes(url, refresh=refresh))


def make_client(cache_dir: Path, sec_user_agent: str, sec_max_rps: float) -> CachedClient:
    """The standard client: SEC hosts rate-limited and identified, others polite."""
    sec = HostPolicy(
        rate=min(sec_max_rps, 10.0),
        user_agent=sec_user_agent or None,
        require_user_agent=True,
    )
    policies = {host: sec for host in SEC_HOSTS}
    return CachedClient(cache_dir, policies, default_policy=HostPolicy(rate=4.0))
