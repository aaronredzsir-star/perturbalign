"""Concurrent execution and adaptive token bucket rate limiting engine.

Addresses industrial challenges in high-throughput online perturbation annotation:
1. Rate-limit protection: Token bucket rate limiters tailored for NCBI (3 req/s), PubChem (5 req/s),
   STRING (5 req/s), EBI (5 req/s).
2. Latency mitigation: Thread-pool based concurrent I/O batch querying.
3. Zero heavyweight dependencies: Implemented purely using Python standard library
   (threading, time, collections, asyncio, concurrent.futures).
"""
from __future__ import annotations

import collections
import concurrent.futures
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Sequence

_TIMEOUT = 10
_RETRIES = 3
_RETRY_DELAY = 0.5
_RETRY_STATUS = {429, 500, 502, 503, 504}
_USER_AGENT = "perturbalign/0.9 (single-cell perturbation alignment; mailto:contact@perturbalign.org)"


class TokenBucketLimiter:
    """Thread-safe Token Bucket Rate Limiter.

    Parameters:
      rate: Allowed requests per second.
      capacity: Maximum burst capacity of the token bucket.
    """

    def __init__(self, rate: float = 5.0, capacity: float | None = None) -> None:
        self.rate = max(float(rate), 0.1)
        self.capacity = max(float(capacity or rate * 2.0), 1.0)
        self._tokens = self.capacity
        self._last_update = time.time()
        self._lock = threading.RLock()

    def acquire(self, tokens: float = 1.0, timeout: float = 30.0) -> bool:
        """Acquire tokens; sleeps if insufficient until timeout or tokens are acquired."""
        deadline = time.time() + timeout
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self._last_update
                self._last_update = now
                # Replenish tokens
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                missing = tokens - self._tokens
                wait_time = missing / self.rate

            if time.time() + wait_time > deadline:
                return False

            time.sleep(min(wait_time, 0.2))


class DomainRateLimiterRegistry:
    """Global rate limiter registry partitioned dynamically by target domain."""

    def __init__(self) -> None:
        self._limiters: dict[str, TokenBucketLimiter] = {
            "ncbi.nlm.nih.gov": TokenBucketLimiter(rate=3.0, capacity=5.0),
            "pubchem.ncbi.nlm.nih.gov": TokenBucketLimiter(rate=5.0, capacity=10.0),
            "string-db.org": TokenBucketLimiter(rate=5.0, capacity=10.0),
            "stitch.embl.de": TokenBucketLimiter(rate=5.0, capacity=10.0),
            "ebi.ac.uk": TokenBucketLimiter(rate=5.0, capacity=10.0),
            "uniprot.org": TokenBucketLimiter(rate=5.0, capacity=10.0),
            "reactome.org": TokenBucketLimiter(rate=5.0, capacity=10.0),
            "opentargets.org": TokenBucketLimiter(rate=5.0, capacity=10.0),
            "jaspar": TokenBucketLimiter(rate=5.0, capacity=10.0),
            "proteinatlas.org": TokenBucketLimiter(rate=5.0, capacity=10.0),
            "api.fda.gov": TokenBucketLimiter(rate=4.0, capacity=8.0),
            "cancerrxgene.org": TokenBucketLimiter(rate=5.0, capacity=10.0),
            "bindingdb.org": TokenBucketLimiter(rate=3.0, capacity=6.0),
            "default": TokenBucketLimiter(rate=5.0, capacity=10.0),
        }
        self._lock = threading.RLock()

    def get_limiter_for_url(self, url: str) -> TokenBucketLimiter:
        """Match domain rate limiter for target URL."""
        with self._lock:
            for domain, limiter in self._limiters.items():
                if domain in url:
                    return limiter
            return self._limiters["default"]

    def set_rate(self, domain: str, rate: float, capacity: float | None = None) -> None:
        """Configure custom rate for a specific domain."""
        with self._lock:
            self._limiters[domain] = TokenBucketLimiter(rate=rate, capacity=capacity)


# Global default rate limiter instance
global_rate_limiters = DomainRateLimiterRegistry()


def run_concurrent_tasks(
    func: Callable[[Any], Any],
    items: Sequence[Any],
    max_workers: int = 5,
    rate_limiter: TokenBucketLimiter | None = None,
) -> list[Any]:
    """Execute batch tasks concurrently with optional rate limiting and error resilience.

    Parameters:
      func: Processing function for a single item (e.g. query_gene or align_single).
      items: Sequence of input items.
      max_workers: Number of concurrent worker threads (default 5).
      rate_limiter: Optional global rate limiter instance.

    Returns:
      List of results strictly preserving input order.
    """
    if not items:
        return []

    results: list[Any] = [None] * len(items)

    def _worker(idx: int, item: Any) -> tuple[int, Any]:
        if rate_limiter is not None:
            rate_limiter.acquire(1.0)
        res = func(item)
        return idx, res

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_worker, i, it): i for i, it in enumerate(items)}
        for future in concurrent.futures.as_completed(future_map):
            try:
                idx, out = future.result()
                results[idx] = out
            except Exception as e:
                orig_idx = future_map[future]
                results[orig_idx] = None

    return results



def http_get_json(
    url: str,
    retries: int = _RETRIES,
    timeout: float = _TIMEOUT,
    headers: dict[str, str] | None = None,
    rate_limit: bool = True,
) -> dict:
    """Execute GET JSON with domain token bucket rate limiting, timeout, and exponential backoff retry."""
    if rate_limit:
        limiter = global_rate_limiters.get_limiter_for_url(url)
        limiter.acquire(1.0)

    req_headers = {"User-Agent": _USER_AGENT}
    if headers:
        req_headers.update(headers)

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in _RETRY_STATUS and attempt < retries - 1:
                time.sleep(_RETRY_DELAY * (2 ** attempt))
                continue
            raise
        except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(_RETRY_DELAY * (2 ** attempt))
                continue
            raise
    if last_exc:
        raise last_exc
    return {}


def http_post_json(
    url: str,
    json_body: dict,
    retries: int = _RETRIES,
    timeout: float = _TIMEOUT,
    headers: dict[str, str] | None = None,
    rate_limit: bool = True,
) -> dict:
    """Execute POST JSON with domain token bucket rate limiting, timeout, and exponential backoff retry."""
    if rate_limit:
        limiter = global_rate_limiters.get_limiter_for_url(url)
        limiter.acquire(1.0)

    req_headers = {"User-Agent": _USER_AGENT, "Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    data_bytes = json.dumps(json_body).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data_bytes, headers=req_headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in _RETRY_STATUS and attempt < retries - 1:
                time.sleep(_RETRY_DELAY * (2 ** attempt))
                continue
            raise
        except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(_RETRY_DELAY * (2 ** attempt))
                continue
            raise
    if last_exc:
        raise last_exc
    return {}


def http_get_text(
    url: str,
    retries: int = 2,
    timeout: float = _TIMEOUT,
    rate_limit: bool = True,
) -> str | None:
    """Execute safe GET plain text with retry and rate limiting (returns None on 404 or network error)."""
    if rate_limit:
        limiter = global_rate_limiters.get_limiter_for_url(url)
        limiter.acquire(1.0)

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8").strip()
                if "<html" in text.lower() or "not found" in text.lower() or not text:
                    return None
                return text
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in _RETRY_STATUS and attempt < retries - 1:
                time.sleep(_RETRY_DELAY * (2 ** attempt))
                continue
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(_RETRY_DELAY * (2 ** attempt))
                continue
            return None
    return None


__all__ = [
    "TokenBucketLimiter",
    "DomainRateLimiterRegistry",
    "global_rate_limiters",
    "run_concurrent_tasks",
    "http_get_json",
    "http_post_json",
    "http_get_text",
]

