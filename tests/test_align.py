"""perturbalign Test ：   align()     Test column   +      Test 。

  :
    pytest tests/                  # 
    python tests/test_align.py     # no pytest row （ ）
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import perturbalign  # noqa: E402
from perturbalign.align import align, to_dataframe  # noqa: E402
from helpers import DictCache, build_fixture_cache, make_fake_multi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK = os.path.join(ROOT, "data", "benchmark", "benchmark_cases.json")

CASES = json.load(open(BENCHMARK, encoding="utf-8"))["cases"]
FIXTURE_CACHE = build_fixture_cache(CASES)  # cache （ API ）

# ： + fixture cache （online row is case ）
def _align(values, **kwargs):
    kwargs.setdefault("cache", FIXTURE_CACHE)
    kwargs.setdefault("online", False)
    return align(values, **kwargs)


def _check_case(case: dict) -> list[str]:
    """          ，return failure   column  （empty column   =   ）。"""
    fails = []
    kwargs = {"cache": FIXTURE_CACHE}
    # online combo mock API（data API Verify ）
    kwargs["online"] = case["expected_status"] in ("online", "combo")
    result = _align(case["input"], **kwargs)[0]
    entity = result.entity

    def expect(label, got, expected):
        if got != expected:
            fails.append(f"[{case['input']}] {label}:    {expected!r},    {got!r}")

    expect("status", result.status, case["expected_status"])
    expect("is_control", result.is_control, case["expected_is_control"])
    expect("parsed_name", result.parsed_name, case["expected_parsed_name"])
    if case.get("expected_dose") is not None:
        expect("parsed_dose", result.parsed_dose, case["expected_dose"])
    if case.get("expected_unit") is not None:
        expect("parsed_unit", result.parsed_unit, case["expected_unit"])
    # drug_name smiles（exact fixture cache ，online mock API）
    if case["expected_status"] not in ("control", "combo", "empty", "unmatched"):
        expect("drug_name", entity.canonical_name if entity else None,
               case["expected_drug_name"])
        if case.get("expected_smiles") is not None:
            expect("canonical_smiles", entity.canonical_smiles if entity else None,
                   case["expected_smiles"])
    if case["expected_status"] == "combo":
        exp_parts = {p.lower() for p in (case.get("expected_part_drugs") or [])}
        got_parts = {p.entity.canonical_name.lower() for p in result.parts
                     if p.entity is not None}
        if not exp_parts.issubset(got_parts):
            fails.append(
                f"[{case['input']}] combo   :     {sorted(exp_parts)}, "
                f"   {sorted(got_parts)}"
            )
    return fails


def test_benchmark_all(monkeypatch):
    """35         ：    + fixture cache + mock online   all   。"""
    monkeypatch.setattr("perturbalign.annotate.pubchem_lookup_multi",
                        make_fake_multi())
    fails = []
    for case in CASES:
        fails.extend(_check_case(case))
    assert not fails, "\n".join(fails)


# ---- Test ----

def test_no_builtin_database():
    """ no       data ： offline     （dose /control /combination ），structure is empty 。"""
    assert align(["Trametinib"], online=False, cache=DictCache())[0].status == "unmatched"
    assert align(["Trametinib 10nM"], online=False, cache=DictCache())[0].parsed_dose == 10.0
    assert align(["DMSO"], online=False, cache=DictCache())[0].status == "control"
    assert align(["vemurafenib_cobimetinib"], online=False,
                 cache=DictCache())[0].status == "combo"


def test_online_default_api(monkeypatch):
    """data      API：no cache 、no      align()    online=True   PubChem。"""
    monkeypatch.setattr("perturbalign.annotate.pubchem_lookup_multi",
                        make_fake_multi())
    r = align(["FakeDrugX"])[0]
    assert r.status == "online"
    assert r.entity.canonical_smiles == "CCCC"
    assert r.entity.source == "pubchem"
    # ： offline -> unmatched
    r2 = align(["FakeDrugX"], online=False, cache=DictCache())[0]
    assert r2.status == "unmatched"


def test_online_ambiguous_multi_candidates(monkeypatch):
    """API    and   not   -> ambiguous（not      top1）。"""
    def fake_multi(query, cache=None):
        return [{"canonical_name": query, "canonical_smiles": "S1",
                 "inchikey": "K1", "pubchem_cid": 1},
                {"canonical_name": query, "canonical_smiles": "S2",
                 "inchikey": "K2", "pubchem_cid": 2}]
    monkeypatch.setattr("perturbalign.annotate.pubchem_lookup_multi", fake_multi)
    r = align(["AmbiguousDrug"])[0]
    assert r.status == "ambiguous"
    assert r.entity is None
    assert len(r.candidates) == 2


def test_extra_control_tokens():
    """data     ：not           control token（   ，no    ）。"""
    result = align(["blank well"], extra_control_tokens=["blank well"],
                   cache=DictCache())[0]
    assert result.status == "control"
    assert result.is_control is True


def test_cache_corpus_fuzzy():
    """cache    ：   API     offline Fuzzy matching（rapidfuzz   ）。"""
    cache = DictCache()
    cache.set("pubchem:name:vemurafenib",
              {"canonical_name": "Vemurafenib", "canonical_smiles": "CCCC",
               "inchikey": None, "pubchem_cid": None})
    r = align(["Vemurafneib"], fuzzy=True, online=False, cache=cache)[0]
    assert r.status == "fuzzy"
    assert r.entity.canonical_name == "Vemurafenib"
    # fuzzy not tolerance
    r2 = align(["Vemurafneib"], fuzzy=False, online=False, cache=cache)[0]
    assert r2.status == "unmatched"


def test_cache_layer():
    """SQLite cache  ：  /  /vocabulary     /key_text     。"""
    import tempfile
    from perturbalign.cache import SQLiteCache
    from perturbalign import resources
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = f.name
    try:
        cache = SQLiteCache(path)
        ver = resources.VOCABULARY_VERSION
        cache.set("pubchem:name:testdrug", {"candidates": [{"pubchem_cid": 1}]},
                  ttl_seconds=3600, vocab_version=ver)
        assert cache.get("pubchem:name:testdrug", vocab_version=ver)["candidates"]
        # vocabulary ->
        assert cache.get("pubchem:name:testdrug", vocab_version="9.9.9") is None
        # 
        names = cache.iter_names("pubchem:name:", vocab_version=ver)
        assert any(k == "pubchem:name:testdrug" for k, _ in names)
        cache.close()
    finally:
        os.unlink(path)


def test_pubchem_offline_safe(monkeypatch):
    """online API：  network exception   degradation is None（Offline safety +   cache ）。"""
    from perturbalign import annotate

    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(annotate.urllib.request, "urlopen", boom)
    assert annotate.pubchem_lookup_multi("SomeDrug") is None


def test_align_series_dataframe():
    """Series /  column DataFrame   ； column DataFrame     。"""
    import pandas as pd
    ser = pd.Series(["DMSO", "Trametinib 10nM"])
    results = _align(ser)
    assert results[0].status == "control"
    assert results[1].status == "exact"
    results2 = _align(pd.DataFrame({"perturbation": ["Sorafenib ", "vehicle"]}))
    assert results2[0].status == "exact"
    assert results2[1].status == "control"
    try:
        _align(pd.DataFrame({"a": [1], "b": [2]}))
        raise AssertionError(" column DataFrame    ")
    except ValueError:
        pass


def test_align_single_string_and_list():
    """str / list / tuple input   。"""
    assert _align("DMSO")[0].status == "control"
    assert _align(["DMSO", "control"])[1].status == "control"
    assert _align(("DMSO",))[0].status == "control"


def test_align_none_and_nan():
    """None / NaN input  is empty （v0.6   ：   str(None)='None'     ）。"""
    import pandas as pd
    res = _align([None, float("nan"), pd.NA, ""])
    assert all(r.status == "empty" for r in res)


def test_to_dataframe_output():
    """to_dataframe output      。"""
    results = _align(["DMSO", "Trametinib 10nM", "vemurafenib_cobimetinib"])
    df = to_dataframe(results)
    assert "status" in df.columns and "drug_name" in df.columns
    assert len(df) == 3
    row = df.iloc[1]
    assert row["status"] == "exact"
    assert row["drug_name"] == "Trametinib"
    assert row["parsed_dose"] == 10.0
    assert row["parsed_unit"] == "nM"
    assert row["is_control"] == False  # noqa: E712 —— pandas column return np.bool_


if __name__ == "__main__":  # row （no pytest）
    import perturbalign.annotate as _annotate
    _annotate.pubchem_lookup_multi = make_fake_multi()  # mock online
    fails = []
    for case in CASES:
        fails.extend(_check_case(case))
    extra = [
        ("empty input ", _align("")[0].status, "empty"),
        ("None", _align(None)[0].status, "empty"),
    ]
    for label, got, want in extra:
        if got != want:
            fails.append(f"[{label}]    {want!r},    {got!r}")
    if fails:
        print("\n".join(fails))
        sys.exit(1)
    print(f"{len(CASES) + len(extra)}/{len(CASES) + len(extra)}     , 0 failure ")
