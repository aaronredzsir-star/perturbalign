# -*- coding: utf-8 -*-
"""Test     （  test_ prefix ，pytest not   ）。

v0.6   ： no    、no data  ；alignment    Verify    ——
  1) fixture cache （  "API        "）Verify cache   /  match ；
  2) mock API Verify online   （PubChem     -> online/ambiguous）。
   zero network ，Test deterministic 。
"""
from __future__ import annotations


class DictCache:
    """dict  cache （   get/set/iter_names，  SQLiteCache  protocol ）。"""

    def __init__(self):
        self._d: dict = {}

    def get(self, key: str, vocab_version: str | None = None):
        return self._d.get(key)

    def set(self, key: str, result: dict, ttl_seconds=None, vocab_version=None):
        self._d[key] = result

    def iter_names(self, prefix: str = "pubchem:name:", vocab_version=None):
        # SQLiteCache.iter_names alignment ： /empty cache
        return [(k, v) for k, v in self._d.items()
                if k.startswith(prefix)
                and not v.get("unreachable") and not v.get("empty")]


def build_fixture_cache(cases: list[dict]) -> DictCache:
    """from       expected_drug_name/SMILES     cache 。

      "        API     cache "——cache    。
    """
    c = DictCache()
    for case in cases:
        dn = case.get("expected_drug_name")
        sm = case.get("expected_smiles")
        if case.get("expected_status") == "exact" and dn:
            c.set(
                f"pubchem:name:{dn.lower()}",
                {"canonical_name": dn, "canonical_smiles": sm,
                 "inchikey": None, "pubchem_cid": None},
            )
    return c


def make_fake_multi(canonical_name=None):
    """   mock   pubchem_lookup_multi：    return    （canonical_name=   ）。"""
    def fake_multi(query, cache=None):
        return [{
            "canonical_name": canonical_name or query,
            "canonical_smiles": "CCCC",
            "inchikey": "FAKE-INCHIKEY",
            "pubchem_cid": 424242,
        }]
    return fake_multi
