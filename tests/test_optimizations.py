"""Test v0.6.6+    all         。

    ：
1. db       DeprecationWarning；
2. Candidate properties   （canonical_name, inchikey etc. ）；
3. user_mapping             ；
4. PreprocessConfig.with_extra_controls zero    IO     ；
5.    single-cell     （50,000 row   input      ）；
6. vocabulary    Hash     ；
7. SQLite cache export_corpus   import_corpus offline     ；
8. zero      difflib Fuzzy matchingdegradation 。
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import perturbalign
from perturbalign.align import align
from perturbalign.schema import Candidate, DrugEntity
from helpers import DictCache




def test_candidate_property_delegates():
    """Verify Candidate   properties       。"""
    ent = DrugEntity(
        input_string="Doxorubicin",
        canonical_name="Doxorubicin",
        canonical_smiles="CC1=C",
        inchikey="INCHIKEY123",
        pubchem_cid=12345,
        parent_inchikey="PARENTKEY123",
    )
    cand = Candidate(entity=ent, score=0.9, matched_by="test")
    assert cand.canonical_name == "Doxorubicin"
    assert cand.canonical_smiles == "CC1=C"
    assert cand.inchikey == "INCHIKEY123"
    assert cand.pubchem_cid == 12345
    assert cand.parent_inchikey == "PARENTKEY123"

    # empty entity
    empty_cand = Candidate(entity=None, score=0.0)  # type: ignore
    assert empty_cand.canonical_name is None
    assert empty_cand.inchikey is None


def test_user_mapping_override():
    """Verify user_mapping        ， parse       code 。"""
    cache = DictCache()
    custom_map = {
        "Compound-X": "Doxorubicin",
        "Compound-Y 10uM": DrugEntity(input_string="Compound-Y", canonical_name="Paclitaxel", inchikey="PAC123"),
    }
    # offline 、empty cache ， user_mapping
    res = align(["Compound-X 5nM", "Compound-Y 10uM"], online=False, cache=cache, user_mapping=custom_map)
    assert len(res) == 2

    # 1.
    assert res[0].status == "exact"
    assert res[0].parsed_dose == 5.0
    assert res[0].parsed_unit == "nM"
    assert res[0].entity.canonical_name == "Doxorubicin"
    assert res[0].candidates[0].canonical_name == "Doxorubicin"

    # 2. DrugEntity
    assert res[1].status == "exact"
    assert res[1].entity.canonical_name == "Paclitaxel"
    assert res[1].entity.inchikey == "PAC123"


def test_preprocess_config_fast_clone():
    """Verify with_extra_controls     not        TSV。"""
    cfg = perturbalign.PreprocessConfig()
    cfg_new = cfg.with_extra_controls(["custom_blank", "ctrl_mock"])
    assert "custom_blank" in cfg_new.all_control_tokens
    assert "ctrl_mock" in cfg_new.all_control_tokens
    # not
    assert "custom_blank" not in cfg.all_control_tokens


def test_unique_mapping_scale_performance():
    """  single-cell 50,000    （   has 10   ），Verify         row 。"""
    ten_drugs = [
        "Trametinib 10nM", "DMSO", "vehicle", "Doxorubicin HCl 5uM",
        "GDC-0032 1uM", "EMPTY_VECTOR", "A + B", None, "nan", "untreated",
    ]
    large_input = ten_drugs * 5000  # 50,000 row

    start = time.perf_counter()
    res = align(large_input, online=False, cache=DictCache())
    duration = time.perf_counter() - start

    assert len(res) == 50000
    # 5 row single-cell input in / （ < 0.5s）
    assert duration < 2.0, f"Expected < 2.0s, got {duration:.3f}s"
    assert res[0].parsed_name == "Trametinib"
    assert res[0].parsed_dose == 10.0
    assert res[1].status == "control"
    assert res[7].status == "empty"


def test_vocabulary_version_dynamic_hash():
    """Verify VOCABULARY_VERSION     and     specification 。"""
    ver = perturbalign.resources.VOCABULARY_VERSION
    assert ver.startswith("0.9.0-")
    assert len(ver.split("-")[-1]) == 8


def test_sqlite_cache_export_import():
    """Verify SQLiteCache    and     （offline     ）。"""
    from perturbalign.cache import SQLiteCache

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f1:
        path1 = f1.name
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f2:
        path2 = f2.name

    try:
        c1 = SQLiteCache(path1)
        c1.set("pubchem:name:drug_a", {"canonical_name": "DrugA", "inchikey": "KEYA"})
        c1.set("pubchem:name:drug_b", {"canonical_name": "DrugB", "inchikey": "KEYB"})
        c1.set("pubchem:name:net_fail", {"unreachable": True})  # cache not

        n_exported = c1.export_corpus(path2)
        assert n_exported == 2
        c1.close()

        # in
        c2 = SQLiteCache(path2)
        assert c2.get("pubchem:name:drug_a")["canonical_name"] == "DrugA"
        assert c2.get("pubchem:name:drug_b")["canonical_name"] == "DrugB"
        assert c2.get("pubchem:name:net_fail") is None
        c2.close()
    finally:
        for p in (path1, path2):
            if os.path.exists(p):
                os.remove(p)


def test_difflib_fuzzy_fallback_precision():
    """Verify difflib Fuzzy matchingdegradation        1-2      。"""
    c = DictCache()
    c.set("pubchem:name:doxorubicin", {"canonical_name": "Doxorubicin", "inchikey": "DOX123"})
    # : Doxorubcin ( i)
    res = align(["Doxorubcin"], fuzzy=True, online=False, cache=c)[0]
    assert res.status == "fuzzy"
    assert res.entity.canonical_name == "Doxorubicin"
    assert res.entity.confidence < 0.75
