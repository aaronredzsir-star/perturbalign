"""v0.6.4 network  Test ：     /   backoff retry /   failure fallback / cache    。

all   mock，zero network 。   P0     。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from perturbalign import annotate as A
from helpers import DictCache


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_json(url, retries=3):
    """   mock：  URL return    JSON。"""
    raise NotImplementedError


# ---------------- （ cids + props） ----------------

def test_batch_success(monkeypatch):
    """  ：2     cids + 1     props，    return all    。"""
    calls = []

    def fake(url, retries=3):
        calls.append(url)
        if "name/DrugA" in url:
            return {"IdentifierList": {"CID": [1]}}
        if "name/DrugB" in url:
            return {"IdentifierList": {"CID": [2, 3]}}
        if "/cid/" in url:  # props（CID ）
            return {"PropertyTable": {"Properties": [
                {"CID": 1, "CanonicalSMILES": "C1", "InChIKey": "K1"},
                {"CID": 2, "CanonicalSMILES": "C2", "InChIKey": "K2"},
                {"CID": 3, "CanonicalSMILES": "C3", "InChIKey": "K3"},
            ]}}
        raise AssertionError(url)

    monkeypatch.setattr(A, "_http_get_json", fake)
    cache = DictCache()
    out = A.pubchem_lookup_batch(["DrugA", "DrugB"], cache=cache, batch_size=10)
    assert out["DrugA"][0]["pubchem_cid"] == 1
    assert [c["pubchem_cid"] for c in out["DrugB"]] == [2, 3]
    assert len(calls) == 3  # 2 cids + 1 props（ not is 4 props）
    # cache ： multi ， align
    assert cache.get("pubchem:name:druga")["candidates"][0]["pubchem_cid"] == 1


def test_batch_cache_hit_zero_network(monkeypatch):
    """cache   ：  zero network   。"""
    cache = DictCache()
    cache.set("pubchem:name:druga", {"candidates": [{"pubchem_cid": 1}]})
    cache.set("pubchem:name:drugb", {"unreachable": True})

    def boom(url, retries=3):
        raise AssertionError("should not hit network")

    monkeypatch.setattr(A, "_http_get_json", boom)
    out = A.pubchem_lookup_batch(["DrugA", "DrugB"], cache=cache)
    assert out["DrugA"][0]["pubchem_cid"] == 1
    assert out["DrugB"] is None


def test_batch_empty_name_cached(monkeypatch):
    """    PubChem no   ：  empty cache ，return None。"""
    calls = []

    def fake(url, retries=3):
        calls.append(url)
        if "name/Ghost" in url:
            return {"IdentifierList": {"CID": []}}
        if "name/Real" in url:
            return {"IdentifierList": {"CID": [7]}}
        if "/cid/" in url:
            return {"PropertyTable": {"Properties": [{"CID": 7, "CanonicalSMILES": "C7", "InChIKey": "K7"}]}}
        raise AssertionError(url)

    monkeypatch.setattr(A, "_http_get_json", fake)
    cache = DictCache()
    out = A.pubchem_lookup_batch(["Ghost", "Real"], cache=cache, batch_size=10)
    assert out["Ghost"] is None
    assert cache.get("pubchem:name:ghost") == {"empty": True}
    assert out["Real"][0]["pubchem_cid"] == 7


# ---------------- failure fallback ----------------

def test_batch_fallback_to_single(monkeypatch):
    """   cids failure ->   fallback pubchem_lookup_multi（   cache ），not     。"""
    def fake(url, retries=3):
        if "name/BadDrug" in url:
            raise RuntimeError("single name lookup down")
        if "name/GoodDrug" in url:
            return {"IdentifierList": {"CID": [42]}}
        if "/cid/" in url:
            return {"PropertyTable": {"Properties": [{"CID": 42, "CanonicalSMILES": "C42", "InChIKey": "K42"}]}}
        raise AssertionError(url)

    monkeypatch.setattr(A, "_http_get_json", fake)
    cache = DictCache()
    out = A.pubchem_lookup_batch(["BadDrug", "GoodDrug"], cache=cache, batch_size=10)
    assert out["BadDrug"] is None  # failure -> None
    assert out["GoodDrug"][0]["pubchem_cid"] == 42  # not
    assert cache.get("pubchem:name:gooddrug")["candidates"][0]["pubchem_cid"] == 42


# ---------------- backoff retry ----------------

def test_retry_on_503_then_success(monkeypatch):
    """503  retry success 。"""
    import urllib.error
    import perturbalign.annotate as Ann
    state = {"n": 0}

    def fake_urlopen(req, timeout=10):
        state["n"] += 1
        if state["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
        return _FakeResp('{"ok": true}')

    monkeypatch.setattr(Ann.urllib.request, "urlopen", fake_urlopen)
    result = Ann._http_get_json("https://x", retries=2)
    assert result == {"ok": True}
    assert state["n"] == 2  # 503， success


def test_retry_exhausted_raises(monkeypatch):
    """retry      （   degradation    cache ）。"""
    import urllib.error
    import perturbalign.annotate as Ann
    state = {"n": 0}

    def fake_urlopen(req, timeout=10):
        state["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)

    monkeypatch.setattr(Ann.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        Ann._http_get_json("https://x", retries=3)
    assert state["n"] == 3  # 3


def test_non_retryable_4xx_no_retry(monkeypatch):
    """404（  429/5xx）not retry ，   。"""
    import urllib.error
    import perturbalign.annotate as Ann
    state = {"n": 0}

    def fake_urlopen(req, timeout=10):
        state["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)

    monkeypatch.setattr(Ann.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        Ann._http_get_json("https://x", retries=3)
    assert state["n"] == 1  # 404 not retry


# ---------------- 404 （v0.6.4 ） ----------------

def test_404_is_empty_not_unreachable(monkeypatch):
    """PubChem 404（deterministic    ）    empty cache ，   1h unreachable。"""
    import urllib.error
    import perturbalign.annotate as Ann
    cache = DictCache()

    def fake_urlopen(req, timeout=10):
        raise urllib.error.HTTPError(req.full_url, 404, "PUGREST.NotFound", {}, None)

    monkeypatch.setattr(Ann.urllib.request, "urlopen", fake_urlopen)
    res = Ann.pubchem_lookup_multi("GhostDrug", cache=cache)
    assert res is None
    cached = cache.get("pubchem:name:ghostdrug")
    assert cached == {"empty": True}          # empty
    assert cached.get("unreachable") is None  # not is network cache


def test_timeout_is_unreachable(monkeypatch):
    """network timeout   1h unreachable（retry    ）。"""
    import socket
    import perturbalign.annotate as Ann
    cache = DictCache()

    def fake_urlopen(req, timeout=10):
        raise socket.timeout("timed out")

    monkeypatch.setattr(Ann.urllib.request, "urlopen", fake_urlopen)
    res = Ann.pubchem_lookup_multi("AnyDrug", cache=cache)
    assert res is None
    assert cache.get("pubchem:name:anydrug") == {"unreachable": True}


# ---------------- fallback（v0.6.4 ） ----------------

def test_greek_transliterate_fallback(monkeypatch):
    """        404 ->                 cache 。"""
    import urllib.error
    import perturbalign.annotate as Ann
    cache = DictCache()
    calls = []

    def fake_urlopen(req, timeout=10):
        calls.append(req.full_url)
        if "18%CE%B2" in req.full_url:  # （β ）-> 404
            raise urllib.error.HTTPError(req.full_url, 404, "PUGREST.NotFound", {}, None)
        if "18beta" in req.full_url:    # ->
            return _FakeResp('{"IdentifierList": {"CID": [10114]}}')
        if "/cid/10114" in req.full_url:
            return _FakeResp('{"PropertyTable": {"Properties": [{"CID": 10114, '
                             '"CanonicalSMILES": "C1", "InChIKey": "K10114"}]}}')
        raise AssertionError(req.full_url)

    monkeypatch.setattr(Ann.urllib.request, "urlopen", fake_urlopen)
    res = Ann.pubchem_lookup_multi("18β-Glycyrrhetinic acid", cache=cache)
    assert res and res[0]["pubchem_cid"] == 10114
    # cache （ offline ，zero network ）
    assert cache.get("pubchem:name:18β-glycyrrhetinic acid")["candidates"][0]["pubchem_cid"] == 10114
    assert any("18beta" in u for u in calls)  # 
