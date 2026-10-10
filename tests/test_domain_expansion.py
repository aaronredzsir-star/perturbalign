"""perturbalign              Test 。

Verify   ：
  1.  empty   parse （Duration/Timepoint extract ，    not   24h   ）
  2.   perturbation multimodal parse （CRISPR KO / CRISPRi / CRISPRa / shRNA / OE / mutation ）
  3.      /cytokine /    /    
  4.        perturbation   
  5.              （SMILES / InChIKey / CAS   10    / ChEMBL ID / CID）
  6.        fallback   
  7.    Schema   as_dict()    
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from perturbalign.align import align
from perturbalign.identifiers import (
    detect_identifier,
    is_cas_number,
    is_chembl_id,
    is_inchikey,
    is_smiles,
    parse_pubchem_cid,
)
from perturbalign.preprocess import (
    detect_biologic,
    detect_environmental,
    detect_genetic_perturbation,
    parse_duration,
    parse_spatiotemporal,
)
from perturbalign.schema import (
    MODALITY_BIOLOGIC,
    MODALITY_CONTROL,
    MODALITY_ENVIRONMENTAL,
    MODALITY_GENETIC,
    MODALITY_SMALL_MOLECULE,
)
from helpers import DictCache


# ---------------- 1. empty parse Test ----------------

@pytest.mark.parametrize("label,expected_name,expected_dose,expected_unit,expected_dur,expected_dur_unit", [
    ("Trametinib 10nM 24h", "Trametinib", 10.0, "nM", 24.0, "h"),
    ("Trametinib 24h 10nM", "Trametinib", 10.0, "nM", 24.0, "h"),
    ("Trametinib_10nM_24hr", "Trametinib", 10.0, "nM", 24.0, "h"),
    ("Trametinib_10nM_48hrs", "Trametinib", 10.0, "nM", 48.0, "h"),
    ("Trametinib (10nM, 24h)", "Trametinib", 10.0, "nM", 24.0, "h"),
    ("Trametinib (24h, 10nM)", "Trametinib", 10.0, "nM", 24.0, "h"),
    ("10nM Trametinib 24h", "Trametinib", 10.0, "nM", 24.0, "h"),
    ("DMSO 6h", "DMSO", None, None, 6.0, "h"),
    ("EGF 100ng/ml 30min", "EGF", 100.0, "ng/ml", 30.0, "min"),
    ("Tamoxifen 1uM 7d", "Tamoxifen", 1.0, "µM", 7.0, "d"),
    ("Geldanamycin 500nM 2weeks", "Geldanamycin", 500.0, "nM", 2.0, "wk"),
    ("Hypoxia 48h", "Hypoxia", None, None, 48.0, "h"),
])
def test_spatiotemporal_parsing(label, expected_name, expected_dose, expected_unit, expected_dur, expected_dur_unit):
    name, dose, unit, dur, dur_unit = parse_spatiotemporal(label)
    assert name == expected_name
    assert dose == expected_dose
    assert unit == expected_unit
    assert dur == expected_dur
    assert dur_unit == expected_dur_unit


def test_align_with_duration_cleans_name():
    """Verify  empty      align    ，     extract ，duration       ，not    network /cache   。"""
    cache = DictCache()
    # cache
    cache.set("pubchem:name:trametinib", {
        "canonical_name": "Trametinib",
        "canonical_smiles": "CC1=C(C=C(C=C1)I)NC(=O)C2=C(C(=O)N(C(=O)N2C3=CC=CC=C3F)C)NC4=CC=C(C=C4)C(=O)NC5CC5",
        "inchikey": "LMDZBCPBFSXMTL-UHFFFAOYSA-N",
    })
    r = align(["Trametinib 10nM 24h"], online=False, cache=cache)[0]
    assert r.status == "exact"
    assert r.parsed_name == "Trametinib"
    assert r.parsed_dose == 10.0
    assert r.parsed_unit == "nM"
    assert r.parsed_duration == 24.0
    assert r.parsed_duration_unit == "h"
    assert r.entity.canonical_name == "Trametinib"


# ---------------- 2. perturbation multimodal parse ----------------

@pytest.mark.parametrize("label,expected_target,expected_action", [
    ("sgTP53", "TP53", "KO"),
    ("sgRNA-TP53", "TP53", "KO"),
    ("sg_TP53_1", "TP53", "KO"),
    ("sgBRCA1-2", "BRCA1", "KO"),
    ("CRISPRi_MYC", "MYC", "CRISPRi"),
    ("CRISPRa_IL6", "IL6", "CRISPRa"),
    ("shEGFR", "EGFR", "KD"),
    ("shRNA-KRAS", "KRAS", "KD"),
    ("TP53_KO", "TP53", "KO"),
    ("MYC_OE", "MYC", "OE"),
    ("PTEN-KO", "PTEN", "KO"),
    ("EGFR_T790M", "EGFR", "mutant:T790M"),
    ("KRAS_G12D", "KRAS", "mutant:G12D"),
    ("BCR-ABL_A269V", "BCR-ABL", "mutant:A269V"),
])
def test_genetic_perturbation_detection(label, expected_target, expected_action):
    info = detect_genetic_perturbation(label)
    assert info is not None
    target, action = info
    assert target == expected_target
    assert action == expected_action


def test_align_genetic_perturbation():
    """align()     perturbation ，extract target   action，not   small molecule desalt or   PubChem。"""
    r = align(["sgTP53", "CRISPRi_MYC", "MYC_OE"], online=False)[0]
    assert r.modality == MODALITY_GENETIC
    assert r.genetic_target == "TP53"
    assert r.genetic_action == "KO"


@pytest.mark.parametrize("ctrl_token", [
    "sgNT", "sgNonTargeting", "non-targeting", "scramble", "scrambled",
    "aavs1_sg", "luciferase_sg", "safe_targeting", "cas9_ctrl", "shNT",
])
def test_genetic_controls_recognized_as_control(ctrl_token):
    """Verify Perturb-seq   control not      is    'NT' or 'NonTargeting'  gene perturbation 。"""
    r = align([ctrl_token], online=False)[0]
    assert r.is_control is True
    assert r.status == "control"
    assert r.modality == MODALITY_CONTROL


# ---------------- 3. /cytokine / parse ----------------

@pytest.mark.parametrize("label,expected_canonical,expected_cat", [
    ("IL-6", "IL-6", "cytokine"),
    ("il-6", "IL-6", "cytokine"),
    ("TNF-alpha", "TNF-alpha", "cytokine"),
    ("tnf-a", "TNF-alpha", "cytokine"),
    ("IFN-gamma", "IFN-gamma", "cytokine"),
    ("TGF-beta1", "TGF-beta1", "growth_factor"),
    ("EGF", "EGF", "growth_factor"),
    ("Trastuzumab", "Trastuzumab", "antibody"),
    ("trastuzumab", "Trastuzumab", "antibody"),
    ("Pembrolizumab", "Pembrolizumab", "antibody"),
    ("Rituximab", "Rituximab", "antibody"),
])
def test_biologics_detection(label, expected_canonical, expected_cat):
    bio = detect_biologic(label)
    assert bio is not None
    assert bio["canonical_name"] == expected_canonical
    assert bio["category"] == expected_cat


def test_align_biologic_entity():
    """align()  cytokine and     is biologic   ，   specification   。"""
    r = align(["IL-6 10ng/ml 2h"], online=False)[0]
    assert r.modality == MODALITY_BIOLOGIC
    assert r.status == "exact"
    assert r.parsed_name == "IL-6"
    assert r.parsed_dose == 10.0
    assert r.parsed_unit == "ng/ml"
    assert r.parsed_duration == 2.0
    assert r.parsed_duration_unit == "h"
    assert r.entity is not None
    assert r.entity.canonical_name == "IL-6"


# ---------------- 4. perturbation parse ----------------

@pytest.mark.parametrize("label,expected_name,expected_cat", [
    ("Hypoxia", "Hypoxia", "oxygen"),
    ("hypoxia", "Hypoxia", "oxygen"),
    ("Heat shock", "Heat shock", "temperature"),
    ("heat shock 42C", "Heat shock", "temperature"),
    ("Radiation", "Radiation", "radiation"),
    ("radiation 2Gy", "Radiation", "radiation"),
    ("Starvation", "Starvation", "nutrient"),
    ("serum starvation", "Serum starvation", "nutrient"),
])
def test_environmental_detection(label, expected_name, expected_cat):
    env = detect_environmental(label)
    assert env is not None
    assert env["canonical_name"] == expected_name
    assert env["category"] == expected_cat


def test_align_environmental_entity():
    """align()        perturbation ，   to environmental   。"""
    r = align(["Hypoxia 24h"], online=False)[0]
    assert r.modality == MODALITY_ENVIRONMENTAL
    assert r.status == "unmatched"
    assert r.parsed_name == "Hypoxia"
    assert r.parsed_duration == 24.0
    assert r.parsed_duration_unit == "h"


# ---------------- 5. Test ----------------

def test_inchikey_detection():
    valid_ik = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    invalid_ik = "BSYNRYMUTXBXSQ-UHFFFAOYSA"
    assert is_inchikey(valid_ik) is True
    assert is_inchikey(invalid_ik) is False
    assert detect_identifier(valid_ik) == ("inchikey", valid_ik)


def test_cas_number_checksum():
    """Test CAS    10   and   。"""
    aspirin_cas = "50-78-2"
    caffeine_cas = "58-08-2"
    doxorubicin_cas = "23214-92-8"
    assert is_cas_number(aspirin_cas) is True
    assert is_cas_number(caffeine_cas) is True
    assert is_cas_number(doxorubicin_cas) is True

    # （ ）
    assert is_cas_number("50-78-3") is False
    assert is_cas_number("58-08-9") is False
    assert is_cas_number("1-2-3") is False


def test_chembl_id_and_cid_detection():
    assert is_chembl_id("CHEMBL25") is True
    assert is_chembl_id("chembl1743082") is True
    assert is_chembl_id("CHEMBL") is False
    assert detect_identifier("CHEMBL25") == ("chembl_id", "CHEMBL25")

    assert parse_pubchem_cid("CID: 2244") == 2244
    assert parse_pubchem_cid("CID2244") == 2244
    assert detect_identifier("CID: 2244") == ("pubchem_cid", 2244)


def test_smiles_detection():
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    assert is_smiles(aspirin_smiles) is True
    assert is_smiles("C1=CC=CC=C1") is True

    # not is SMILES
    assert is_smiles("Aspirin") is False
    assert is_smiles("Trametinib") is False
    assert is_smiles("Vehicle") is False
    assert is_smiles("DMSO") is False


# ---------------- 6. data online fallback ----------------

def test_resolve_cascade_identifier_routing(monkeypatch):
    """Verify  input is SMILES  ，  parse     pubchem_lookup_by_smiles    name   。"""
    from perturbalign import annotate

    smiles_called = []
    def fake_smiles_lookup(s, cache=None):
        smiles_called.append(s)
        return [{"canonical_name": "Aspirin", "canonical_smiles": s, "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "pubchem_cid": 2244}]

    monkeypatch.setattr(annotate, "pubchem_lookup_by_smiles", fake_smiles_lookup)

    hits, sources = annotate.resolve_online_cascade("CC(=O)Oc1ccccc1C(=O)O")
    assert len(smiles_called) == 1
    assert smiles_called[0] == "CC(=O)Oc1ccccc1C(=O)O"
    assert hits is not None
    assert hits[0]["canonical_name"] == "Aspirin"
    assert "pubchem:smiles" in sources


def test_resolve_cascade_chembl_fallback(monkeypatch):
    """Verify   PubChem     ，  parse      ChEMBL     fallback 。"""
    from perturbalign import annotate

    # PubChem return empty （ ）
    monkeypatch.setattr(annotate, "pubchem_lookup_multi", lambda q, cache=None: None)

    # ChEMBL success
    def fake_chembl_search(q, cache=None):
        return [{
            "canonical_name": "Investigational-Probe-X",
            "canonical_smiles": "CC(=O)N1CCNCC1",
            "inchikey": "ABCDEF-UHFFFAOYSA-N",
            "chembl_id": "CHEMBL999999",
            "source": "chembl",
        }]

    monkeypatch.setattr(annotate, "chembl_search_fallback", fake_chembl_search)

    hits, sources = annotate.resolve_online_cascade("Investigational-Probe-X")
    assert hits is not None
    assert hits[0]["chembl_id"] == "CHEMBL999999"
    assert "chembl" in sources


# ---------------- 7. output Schema ----------------

def test_align_result_schema_fields():
    r = align(["Trametinib 10nM 24h"])[0]
    d = r.as_dict()
    assert "modality" in d
    assert "parsed_duration" in d
    assert "parsed_duration_unit" in d
    assert "genetic_target" in d
    assert "genetic_action" in d
    assert "sources_resolved" in d
    assert d["parsed_duration"] == 24.0
    assert d["parsed_duration_unit"] == "h"
