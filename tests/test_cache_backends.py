"""Unit tests for cache abstraction layer: BaseCache, MemoryCache, NullCache."""
import time
import pytest
from perturbalign import (
    BaseCache,
    MemoryCache,
    NullCache,
    SQLiteCache,
    align,
)


def test_base_cache_subclassing():
    """Verify BaseCache     。"""
    assert issubclass(SQLiteCache, BaseCache)
    assert issubclass(MemoryCache, BaseCache)
    assert issubclass(NullCache, BaseCache)


def test_memory_cache_crud():
    """Verify MemoryCache    、  、TTL      。"""
    cache = MemoryCache()
    assert cache.count() == 0

    cache.set("pubchem:name:doxorubicin", {"canonical_name": "Doxorubicin", "cid": 31703}, ttl_seconds=3600)
    assert cache.count() == 1

    hit = cache.get("pubchem:name:doxorubicin")
    assert hit is not None
    assert hit["canonical_name"] == "Doxorubicin"

    # 
    assert cache.get("pubchem:name:nonexistent") is None

    # TTL Test
    cache.set("temp_key", {"data": 123}, ttl_seconds=0.01)
    time.sleep(0.02)
    assert cache.get("temp_key") is None

    # empty
    cleared = cache.clear()
    assert cleared == 2  # （ ）
    assert cache.count() == 0


def test_memory_cache_iter_names():
    """Verify MemoryCache   iter_names Fuzzy matching    。"""
    cache = MemoryCache()
    cache.set("pubchem:name:trametinib", {"canonical_name": "Trametinib", "cid": 11707110})
    cache.set("pubchem:name:cobimetinib", {"canonical_name": "Cobimetinib", "cid": 25123612})
    cache.set("pubchem:name:unreachable_drug", {"unreachable": True})
    cache.set("other_prefix:key", {"data": 1})

    items = cache.iter_names("pubchem:name:")
    names = [k for k, _ in items]
    assert "pubchem:name:trametinib" in names
    assert "pubchem:name:cobimetinib" in names
    # unreachable cache
    assert "pubchem:name:unreachable_drug" not in names
    # prefix
    assert "other_prefix:key" not in names


def test_memory_cache_version_invalidation():
    """Verify vocabulary           。"""
    cache = MemoryCache()
    cache.set("k1", {"val": 1}, vocab_version="v0.7.0")
    cache.set("k2", {"val": 2}, vocab_version="v0.8.0")

    # 
    assert cache.get("k1", vocab_version="v0.8.0") is None
    assert cache.get("k2", vocab_version="v0.8.0") is not None

    # 
    invalidated = cache.invalidate_version("v0.8.0")
    assert invalidated == 1
    assert cache.count() == 1
    assert cache.get("k2") is not None


def test_null_cache():
    """Verify NullCache   and no    。"""
    cache = NullCache()
    assert cache.count() == 0
    cache.set("any_key", {"data": 1})
    assert cache.count() == 0
    assert cache.get("any_key") is None
    assert cache.iter_names() == []
    assert cache.clear() == 0


def test_align_with_memory_cache():
    """Verify align()   MemoryCache  to    。"""
    mem_cache = MemoryCache()
    # cache
    mem_cache.set(
        "pubchem:name:imatinib",
        {
            "canonical_name": "Imatinib",
            "canonical_smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
            "inchikey": "KTUFNOKKBVMGRW-UHFFFAOYSA-N",
            "pubchem_cid": 5291,
        },
    )

    results = align(["Imatinib 100nM"], online=False, cache=mem_cache)
    assert len(results) == 1
    r = results[0]
    assert r.status == "exact"
    assert r.entity is not None
    assert r.entity.canonical_name == "Imatinib"
    assert r.entity.pubchem_cid == 5291
    assert r.parsed_dose == 100.0
    assert r.parsed_unit == "nM"
