"""Edge case tests for perturbation alignment and normalization.

All tests run offline with empty cache and zero network access.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from perturbalign.align import align
from helpers import DictCache

_C = DictCache()


def _align(v, **kw):
    kw.setdefault("cache", DictCache())
    kw.setdefault("online", False)
    return align(v, **kw)[0]


# ---------------- 1. salt form （ x prefix ） ----------------

@pytest.mark.parametrize("label,expect", [
    ("Doxorubicin HCl", "Doxorubicin"),
    ("Doxorubicin 2HCl", "Doxorubicin"),
    ("Doxorubicin xHCl", "Doxorubicin"),     # v0.6 ：xHCl not "Doxorubicin x"
    ("Doxorubicin xHBr", "Doxorubicin"),
    ("Doxorubicin hydrochloride", "Doxorubicin"),
    ("Doxorubicin dihydrochloride", "Doxorubicin"),
    ("Taselisib HCl", "Taselisib"),
    ("Toremifene Citrate", "Toremifene"),
    ("Ixazomib citrate", "Ixazomib"),
    ("Pevonedistat hydrochloride", "Pevonedistat"),
    ("Alendronate sodium trihydrate", "Alendronate"),  # v0.6.2 ：trihydrate not
    ("Alendronate sodium dihydrate", "Alendronate"),
    ("Sodium Phenylbutyrate", "Phenylbutyrate"),       # v0.6.2 ：prefix metal
    ("Divalproex Sodium", "Divalproex"),
])
def test_salt_stripping(label, expect):
    r = _align(label)
    assert r.parsed_name == expect


# ---------------- 2. empty （ row / / empty ） ----------------

@pytest.mark.parametrize("label,dose,unit", [
    ("Trametinib\n10nM", 10.0, "nM"),      # v0.6 ： row dose extract
    ("Trametinib\t10nM", 10.0, "nM"),
    ("Trametinib  10   nM", 10.0, "nM"),
    ("  Trametinib  ", None, None),
])
def test_whitespace_folding(label, dose, unit):
    r = _align(label)
    assert r.parsed_dose == dose
    assert r.parsed_unit == unit
    assert r.parsed_name == "Trametinib"


# ---------------- 3. / etc. control ----------------

@pytest.mark.parametrize("label", ["DMEM", "FBS", "dmem", "PBS", "media", "water", "saline"])
def test_media_and_buffer_controls(label):
    assert _align(label).status == "control"


# ---------------- 4. （not combo） ----------------

@pytest.mark.parametrize("label,expect", [
    ("Trametinib (DMSO_TF solvate)", "Trametinib"),   # not
    ("Obatoclax (GX15-070)", "Obatoclax"),
    ("Lenalidomide (CC-5013)", "Lenalidomide"),
    ("Patupilone (EPO906, Epothilone B)", "Patupilone"),
    ("Drug (free base)", "Drug"),
    ("(S)-Rivastigmine", "Rivastigmine"),   # prefix
    ("(+)-JQ1", "JQ1"),
    ("(R)-JQ1", "JQ1"),
])
def test_paren_handling(label, expect):
    r = _align(label)
    assert r.status != "combo"              # not is combination
    assert r.parsed_name == expect


# ---------------- 5. dose unit （Unicode/ unit ） ----------------

@pytest.mark.parametrize("label,dose,unit", [
    ("Geldanamycin_1uM", 1.0, "µM"),
    ("Geldanamycin 1uM", 1.0, "µM"),
    ("Geldanamycin 1μM", 1.0, "µM"),
    ("Geldanamycin 1um", 1.0, "µM"),
    ("Geldanamycin 1nM", 1.0, "nM"),
    ("Geldanamycin 1mM", 1.0, "mM"),
    # M（Molar） from unit ：perturbation not M concentration ，
    # and gene mutation （EGFR_T790M） M dose （v0.6.5）
    # ("Geldanamycin 1M", 1.0, "M"),
    ("Trametinib 10mg/mL", 10.0, "mg/ml"),
    ("Trametinib 10 ug/mL", 10.0, "µg/ml"),
    ("Trametinib 10 µg/mL", 10.0, "µg/ml"),
    ("DMSO 0.1%", 0.1, "%"),
    ("DMSO 0.1% v/v", 0.1, "% v/v"),
    ("10nM Trametinib", 10.0, "nM"),
    ("Trametinib (10nM)", 10.0, "nM"),
])
def test_dose_unit_matrix(label, dose, unit):
    r = _align(label)
    assert r.parsed_dose == dose, f"{label}: {r.parsed_dose}"
    assert r.parsed_unit == unit, f"{label}: {r.parsed_unit!r}"


# ---------------- 6. combination ----------------

@pytest.mark.parametrize("label,n_parts", [
    ("A 10nM + B 5uM", 2),
    ("A + B + C", 3),
    ("A/B mixture", 2),
    ("A and B", 2),
    ("A_B", 2),
    ("1:5 A:B", 2),
    ("EGF and bFGF", 2),
    ("vemurafenib_cobimetinib", 2),
])
def test_combo_forms(label, n_parts):
    r = _align(label)
    assert r.status == "combo", f"{label}: {r.status}"
    assert len(r.parts) == n_parts


# ---------------- 7. control （ / / ） ----------------

@pytest.mark.parametrize("label", [
    "dmso", "DMSO", "DMSO DMSO", "vehicle control", "NEGATIVE CONTROL",
    "negative control", "ctrl", "untreated", "untrt", "mock", "scramble",
    "wt", "parental", "ns", "DMEM", "FBS",
])
def test_control_variants(label):
    assert _align(label).status == "control"


# ---------------- 8. empty input ----------------

@pytest.mark.parametrize("label", ["", "   ", "\n", None, float("nan")])
def test_empty_forms(label):
    assert _align([label]).status == "empty"


# ---------------- 9. code / （clean not ） ----------------

@pytest.mark.parametrize("label,expect", [
    ("5-FU", "5-FU"),
    ("6-OHDA", "6-OHDA"),
    ("A-366", "A-366"),
    ("GDC-0941", "GDC-0941"),
    ("BRD-K07682953", "BRD-K07682953"),
    ("PCI-34051", "PCI-34051"),
    ("SRT1720", "SRT1720"),
])
def test_code_names_preserved(label, expect):
    r = _align(label)
    assert r.status != "combo"             # code not is combination
    assert r.parsed_name == expect


# ---------------- 10. / ----------------

@pytest.mark.parametrize("label", ["α-amanitin", "β-lapachone"])
def test_greek_letter_names(label):
    r = _align(label)
    assert r.parsed_name == label          # （API ）


# ---------------- 11. cache / ----------------

def test_corpus_exact_ci():
    """cache        not   （API        not   ）。"""
    cache = DictCache()
    cache.set("pubchem:name:vemurafenib",
              {"canonical_name": "Vemurafenib", "canonical_smiles": "CCCC"})
    assert align(["Vemurafenib"], cache=cache, online=False)[0].status == "exact"
    assert align(["vemurafenib"], cache=cache, online=False)[0].status == "exact"
    assert align(["VEMURAFENIB"], cache=cache, online=False)[0].status == "exact"


def test_corpus_negative_cache_excluded_from_fuzzy():
    """  cache （unreachable）not       。"""
    cache = DictCache()
    cache.set("pubchem:name:vemurafenib", {"unreachable": True})
    r = align(["Vemurafneib"], fuzzy=True, online=False, cache=cache)[0]
    assert r.status == "unmatched"


# ---------------- 13. / / salt （v0.6.3 ， PubChem ） ----------------

@pytest.mark.parametrize("label,expect", [
    ("EDTA disodium salt", "EDTA"),
    ("disodium edta", "edta"),
    ("Edetate sodium tetrahydrate", "Edetate"),
    ("Cobaltous chloride hexahydrate", "Cobaltous chloride"),
    ("Cetylpyridinium chloride monohydrate", "Cetylpyridinium chloride"),
    ("Edetate tetrasodium", "Edetate"),
    ("Alendronate sodium trihydrate", "Alendronate"),
    ("Ceftriaxone disodium", "Ceftriaxone"),
    ("Lactate (calcium)", "Lactate"),   # v0.6.3 ：salt form empty fallback
    ("calcium lactate", "calcium"),
    ("Doxorubicin lactate", "Doxorubicin"),
])
def test_hydrate_and_poly_na_salts(label, expect):
    r = _align(label)
    assert r.parsed_name == expect, f"{label}: {r.parsed_name!r}"


# ---------------- 14. LINCS/ control （ / /sgRNA/DMSO all ，v0.6.3 ） ----------------

@pytest.mark.parametrize("label", [
    "dimethyl sulfoxide", "dimethylsulfoxide", "GFP", "eGFP", "luciferase",
    "RFP", "HcRed", "lacZ", "PGW", "empty vector", "sgNT", "non-targeting",
    "dmso vehicle control",
])
def test_lincs_controls(label):
    assert _align(label).status == "control"


# ---------------- 15. cache ： ----------------

def test_corpus_multi_candidate_ambiguous():
    """cache     （     CID）-> ambiguous，not    top1。"""
    cache = DictCache()
    cache.set("pubchem:name:jq1", {"candidates": [
        {"canonical_name": "JQ1", "canonical_smiles": "S1", "inchikey": "K1", "pubchem_cid": 1},
        {"canonical_name": "JQ1", "canonical_smiles": "S2", "inchikey": "K2", "pubchem_cid": 2},
    ]})
    r = align(["JQ1"], online=False, cache=cache)[0]
    assert r.status == "ambiguous"
    assert r.entity is None
    assert len(r.candidates) == 2
