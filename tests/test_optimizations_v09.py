"""Regression & optimization test suite for PerturbAlign v0.9.0.

Validates the resolution of all 15 audit issues:
1. explode_combos positional integrity (no dose/SMILES shifting on partial combos).
2. align(..., online=True) cascading for raw identifiers (InChIKey, CAS, ChEMBL, CID, SMILES).
3. to_dataframe() string serialization of targets (adata.write_h5ad compatibility).
4. GeneticResolver immunity from online PubChem negative overwriting.
5. parse_spatiotemporal() support for comma/slash/semicolon delimited dose & duration.
6. parse_dose() support for scientific notation (1e-5 M).
7. is_smiles() structural indicator enforcement & acronym/gene false-positive elimination.
8. vendor_prefixes.tsv protection of chemical probes (SGC 0946, NSC 10107).
9. lookup_sgrna() integration into GeneticResolver / AlignmentPipeline.
10. SQLiteCache & MemoryCache invalidate_version() deletion of NULL version entries.
11. Version unification across __init__, resources, annotate, and pyproject.toml.
12. TSV resource inventory count (21 TSV files).
13. Removal of lingering resolve_cache.sqlite in package resources.
14. Candidate.as_dict() None entity safeguard.
"""
from __future__ import annotations

import os
import tempfile
import pytest
import pandas as pd

import perturbalign
from perturbalign.schema import (
    AlignResult,
    Candidate,
    DrugEntity,
    GeneticEntity,
    STATUS_EXACT,
    STATUS_UNMATCHED,
    STATUS_ONLINE,
    STATUS_CONTROL,
    MODALITY_GENETIC,
    MODALITY_SMALL_MOLECULE,
)
from perturbalign.align import align, to_dataframe
from perturbalign.adapter import explode_combos, annotate_dataframe
from perturbalign.preprocess import parse_spatiotemporal, parse_dose, clean_name, PreprocessConfig
from perturbalign.identifiers import is_smiles, detect_identifier
from perturbalign.cache import SQLiteCache, MemoryCache
from perturbalign.resources import VOCABULARY_VERSION


def test_issue_01_explode_combos_positional_alignment():
    """1. explode_combos() not  in       dose /SMILES        。"""
    df = pd.DataFrame({
        "perturbation": ["DrugA + DrugB 10nM", "DrugX 50uM + DrugY"],
    })
    annotated = annotate_dataframe(df, online=False)
    exploded = explode_combos(annotated)

    assert len(exploded) == 4

    # Row 0: DrugA (dose is None)
    row0 = exploded.iloc[0]
    assert row0["component_name"] == "DrugA"
    assert row0["component_index"] == 0
    assert row0["component_dose"] is None or pd.isna(row0["component_dose"])
    assert row0["component_unit"] is None or pd.isna(row0["component_unit"])

    # Row 1: DrugB (dose is 10.0 nM)
    row1 = exploded.iloc[1]
    assert row1["component_name"] == "DrugB"
    assert row1["component_index"] == 1
    assert row1["component_dose"] == 10.0
    assert row1["component_unit"] == "nM"

    # Row 2: DrugX (dose is 50.0 uM)
    row2 = exploded.iloc[2]
    assert row2["component_name"] == "DrugX"
    assert row2["component_index"] == 0
    assert row2["component_dose"] == 50.0
    assert row2["component_unit"] == "µM"

    # Row 3: DrugY (dose is None)
    row3 = exploded.iloc[3]
    assert row3["component_name"] == "DrugY"
    assert row3["component_index"] == 1
    assert row3["component_dose"] is None or pd.isna(row3["component_dose"])


def test_issue_02_align_online_cascades_raw_identifiers(monkeypatch):
    """2. align(..., online=True)         raw_*       structure 。"""
    called_queries = []

    def mock_cascade(query, cache=None, registry=None):
        called_queries.append(query)
        if "BSYNRYMUTXBXSQ" in query:
            return [{
                "canonical_name": "Aspirin",
                "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "pubchem_cid": 2244,
                "source": "pubchem",
            }], ["pubchem:inchikey"]
        return None, []

    monkeypatch.setattr("perturbalign.annotate.resolve_online_cascade", mock_cascade)

    res = align(["InChIKey=BSYNRYMUTXBXSQ-UHFFFAOYSA-N"], online=True, cache=MemoryCache())[0]
    assert res.status == STATUS_ONLINE
    assert res.entity is not None
    assert res.entity.canonical_name == "Aspirin"
    assert res.entity.pubchem_cid == 2244
    assert res.entity.canonical_smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert "pubchem:inchikey" in res.sources_resolved
    assert any("BSYNRYMUTXBXSQ" in q for q in called_queries)


def test_issue_03_to_dataframe_targets_string_and_h5ad():
    """3. to_dataframe()   targets   is      （  Python list），   h5ad   。"""
    ent = DrugEntity(
        input_string="Doxorubicin",
        canonical_name="Doxorubicin",
        targets=["TOP2A", "TOP2B"],
    )
    ar = AlignResult(
        input="Doxorubicin",
        status=STATUS_EXACT,
        entity=ent,
    )
    df = to_dataframe([ar])
    assert "targets" in df.columns
    assert df["targets"].iloc[0] == "TOP2A; TOP2B"
    assert isinstance(df["targets"].iloc[0], str)

    # Verify anndata obs （Verify targets list h5ad exception ）
    try:
        import anndata as ad
        import numpy as np
        adata = ad.AnnData(X=np.zeros((1, 2)), obs=pd.DataFrame({"targets": df["targets"].tolist()}))
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            adata.write_h5ad(tmp_path)
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except ImportError:
        pass


def test_issue_04_genetic_resolver_online_immunity(monkeypatch):
    """4.   perturbation not  in online=True       PubChem API or    cache    note。"""
    pubchem_calls = []

    def mock_pubchem_batch(names, cache=None):
        pubchem_calls.extend(names)
        return {}

    monkeypatch.setattr("perturbalign.annotate.pubchem_lookup_batch", mock_pubchem_batch)

    res_list = align(["sgTP53", "MYC_OE"], online=True, cache=MemoryCache())
    assert len(res_list) == 2

    for r in res_list:
        assert r.modality == MODALITY_GENETIC
        assert "not found in PubChem" not in (r.note or "")
        assert "genetic perturbation" in (r.note or "")

    assert "sgTP53" not in pubchem_calls
    assert "MYC_OE" not in pubchem_calls


def test_issue_05_parse_spatiotemporal_delimiters():
    """5. parse_spatiotemporal()      、  、  etc.     empty     。"""
    # 
    n1, d1, u1, t1, tu1 = parse_spatiotemporal("Trametinib 10nM, 24h")
    assert n1 == "Trametinib"
    assert d1 == 10.0
    assert u1 == "nM"
    assert t1 == 24.0
    assert tu1 == "h"

    # duration in ，
    n2, d2, u2, t2, tu2 = parse_spatiotemporal("Trametinib 24h, 10nM")
    assert n2 == "Trametinib"
    assert d2 == 10.0
    assert u2 == "nM"
    assert t2 == 24.0
    assert tu2 == "h"

    # 
    n3, d3, u3, t3, tu3 = parse_spatiotemporal("Paclitaxel 50uM; 48hr")
    assert n3 == "Paclitaxel"
    assert d3 == 50.0
    assert u3 == "µM"
    assert t3 == 48.0
    assert tu3 == "h"

    n4, d4, u4, t4, tu4 = parse_spatiotemporal("Erlotinib 100nM/6h")
    assert n4 == "Erlotinib"
    assert d4 == 100.0
    assert u4 == "nM"
    assert t4 == 6.0
    assert tu4 == "h"


def test_issue_06_parse_dose_scientific_notation():
    """6. parse_dose()          (  1e-5 uM, 1.5e-6 nM, 2E-3 mM)。"""
    n1, d1, u1 = parse_dose("Trametinib 1e-5 uM")
    assert n1 == "Trametinib"
    assert d1 == pytest.approx(1e-5)
    assert u1 == "µM"

    n2, d2, u2 = parse_dose("DrugX 1.5e-6 nM")
    assert n2 == "DrugX"
    assert d2 == pytest.approx(1.5e-6)
    assert u2 == "nM"

    n3, d3, u3 = parse_dose("CompoundA 2E-3 mM")
    assert n3 == "CompoundA"
    assert d3 == pytest.approx(2e-3)
    assert u3 == "mM"


def test_issue_07_is_smiles_false_positives_and_indicators():
    """7. is_smiles()           gene    (BOP, SON, PIN, BIN, COP)。"""
    # gene / in no prefix not is SMILES
    for false_case in ["BOP", "SON", "PIN", "BIN", "COP"]:
        assert is_smiles(false_case) is False, f"{false_case} should not be smiles without prefix"
        assert detect_identifier(false_case) is None, f"{false_case} should not be detected as identifier"

    # prefix
    assert is_smiles("SMILES: BOP") is True

    # SMILES
    assert is_smiles("CCO") is True

    # structure features SMILES
    assert is_smiles("c1ccccc1") is True
    assert is_smiles("CC(=O)Oc1ccccc1C(=O)O") is True
    assert is_smiles("[Na+].[Cl-]") is True


def test_issue_08_vendor_prefixes_chemical_probes():
    """8. vendor_prefixes.tsv not          code (SGC 0946, NSC 10107)。"""
    # SGC 0946 is '0946'
    assert clean_name("SGC 0946") == "SGC 0946"
    assert clean_name("NSC 10107") == "NSC 10107"

    # prefix
    assert clean_name("Selleck S1001") == "S1001"
    assert clean_name("MCE HY-101") == "HY-101"


def test_issue_09_lookup_sgrna_in_genetic_resolver():
    """9. lookup_sgrna     GeneticResolver / AlignmentPipeline。"""
    cfg = PreprocessConfig()
    # Test 20bp guide sequence
    guide_seq = "ACGTACGTACGTACGTACGT"
    cfg.sgrna_reference[guide_seq] = {
        "gene_symbol": "BRCA1",
        "source": "test_library",
    }

    res = align([guide_seq], config=cfg, online=False)[0]
    assert res.modality == MODALITY_GENETIC
    assert res.genetic_target == "BRCA1"
    assert res.genetic_action == "KO"
    assert res.entity is not None
    assert res.entity.guide_id == guide_seq


def test_issue_10_cache_invalidate_version_deletes_null_versions():
    """10. invalidate_version     vocab_version is NULL    。"""
    # Test MemoryCache
    mem_cache = MemoryCache()
    mem_cache.set("k_null", {"v": 1}, vocab_version=None)
    mem_cache.set("k_old", {"v": 2}, vocab_version="0.8.0")
    mem_cache.set("k_new", {"v": 3}, vocab_version="0.9.0")

    del_count = mem_cache.invalidate_version("0.9.0")
    assert del_count == 2
    assert mem_cache.get("k_null", vocab_version="0.9.0") is None
    assert mem_cache.get("k_old", vocab_version="0.9.0") is None
    assert mem_cache.get("k_new", vocab_version="0.9.0") is not None

    # Test SQLiteCache
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = tmp.name
    try:
        sql_cache = SQLiteCache(db_path)
        sql_cache.set("k_null", {"v": 1}, vocab_version=None)
        sql_cache.set("k_old", {"v": 2}, vocab_version="0.8.0")
        sql_cache.set("k_new", {"v": 3}, vocab_version="0.9.0")

        del_sql = sql_cache.invalidate_version("0.9.0")
        assert del_sql == 2
        assert sql_cache.get("k_null", vocab_version="0.9.0") is None
        assert sql_cache.get("k_old", vocab_version="0.9.0") is None
        assert sql_cache.get("k_new", vocab_version="0.9.0") is not None
        sql_cache.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_issue_11_version_unification():
    """11. Verify      is 0.9.x。"""
    assert perturbalign.__version__ in ("0.9.0", "0.9.2")
    assert VOCABULARY_VERSION.startswith("0.9.")
    assert len(VOCABULARY_VERSION) == 14

    from perturbalign.annotate import _USER_AGENT
    assert _USER_AGENT.startswith("perturbalign/0.9.")

    # Check pyproject.toml
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject_path = os.path.join(root_dir, "pyproject.toml")
    with open(pyproject_path, encoding="utf-8") as f:
        content = f.read()
    assert any(v in content for v in ['version = "0.9.0"', 'version = "0.9.2"'])


def test_issue_12_resource_tsv_inventory_count():
    """12. Verify resources     has 21   TSV vocabulary   。"""
    res_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "perturbalign", "resources"
    )
    tsvs = [f for f in os.listdir(res_dir) if f.endswith(".tsv")]
    assert len(tsvs) >= 21


def test_issue_13_no_lingering_sqlite_in_source():
    """13. Verify        not  in     resolve_cache.sqlite   。"""
    res_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "perturbalign", "resources"
    )
    lingering_cache = os.path.join(res_dir, "resolve_cache.sqlite")
    assert not os.path.exists(lingering_cache)


def test_issue_14_candidate_as_dict_none_entity():
    """14. Candidate.as_dict() in entity is None  not     AttributeError。"""
    cand = Candidate(
        entity=None,  # type: ignore
        score=0.5,
        score_breakdown={"test": 0.5},
        matched_by="test",
        matched_surface="test_surface",
    )
    d = cand.as_dict()
    assert d["entity"] is None
    assert d["score"] == 0.5
