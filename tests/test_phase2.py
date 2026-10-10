"""perturbalign Phase 2   Test ：Fuzzy matching / annotate_anndata / ChEMBL / RDKit。

      ：rapidfuzz      ；anndata not   （  stub）；
ChEMBL network   mock；rdkit not   （     Verify     ）。
"""
from __future__ import annotations

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import pytest

from perturbalign.align import align
from perturbalign import adapter, annotate, structure
from helpers import DictCache, build_fixture_cache  # noqa: E402

_BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "benchmark", "benchmark_cases.json")
import json as _json  # noqa: E402
FIXTURE_CACHE = build_fixture_cache(_json.load(open(_BENCH, encoding="utf-8"))["cases"])


def _align(values, **kw):
    # / Test ： + fixture cache （online row is test_align.py mock ）
    return align(values, cache=FIXTURE_CACHE, online=False, **kw)


# ---------------- 1. Level-2 Fuzzy matching（cache ） ----------------

def test_fuzzy_off_by_default():
    """fuzzy     ：       unmatched（    ）。"""
    r = _align(["Vemurafneib"])[0]
    assert r.status == "unmatched"


def test_fuzzy_typo_tolerance():
    """fuzzy   ：1         cache      Vemurafenib，       。"""
    r = _align(["Vemurafneib"], fuzzy=True)[0]
    assert r.status == "fuzzy"
    assert r.entity.canonical_name == "Vemurafenib"
    assert r.entity.confidence < 0.75          # 
    assert r.candidates[0].matched_by == "fuzzy matcher"


def test_fuzzy_threshold():
    """    not   ：cache    has Vemurafenib  ，Sorafneib    not  。"""
    cache = DictCache()
    cache.set("pubchem:name:vemurafenib",
              {"canonical_name": "Vemurafenib", "canonical_smiles": "CCCC"})
    r = align(["Sorafneib"], fuzzy=True, online=False, cache=cache)[0]
    assert r.status == "unmatched"


# ---------------- 2. annotate_anndata（data cache /API，zero ） ----------------

class _FakeAnnData:
    def __init__(self, obs, uns=None):
        self.obs = obs
        self.uns = uns or {}


def test_annotate_anndata_writes_obs_and_audit():
    """obs column    + uns audit （data    fixture cache ，obs structure column not   ）。"""
    labels = ["DMSO", "Trametinib 10nM", "vemurafenib_cobimetinib", "GDC-0032"]
    obs = pd.DataFrame({"perturbation": labels})
    stub = _FakeAnnData(obs)
    out = adapter.annotate_anndata(stub, drug_col="perturbation",
                                   cache=FIXTURE_CACHE, online=False)
    assert out is stub  # inplace
    assert "perturbalign_status" in stub.obs.columns
    assert list(stub.obs["perturbalign_status"]) == [
        "control", "exact", "combo", "unmatched",
    ]
    dose = stub.obs["perturbalign_parsed_dose"]
    assert list(pd.isna(dose)) == [True, False, True, True]  # None -> NaN（pandas ）
    assert dose[1] == 10.0
    assert stub.obs["perturbalign_drug_name"][1] == "Trametinib"
    assert list(stub.obs["perturbalign_is_control"]) == [True, False, False, False]
    audit = stub.uns["perturbalign"]
    assert audit["n_total"] == 4
    assert audit["stats"]["control"] == 1
    assert audit["stats"]["combo"] == 1
    assert audit["unmatched"] == ["GDC-0032"]
    assert audit["version"]  # empty


def test_annotate_anndata_cache_hit():
    """no structure column  data   cache /API：Trametinib    fixture cache （exact）。"""
    labels = ["DMSO", "Trametinib 10nM"]
    stub = _FakeAnnData(pd.DataFrame({"perturbation": labels}))
    out = adapter.annotate_anndata(stub, drug_col="perturbation",
                                   cache=FIXTURE_CACHE, online=False)
    assert list(out.obs["perturbalign_status"]) == ["control", "exact"]
    assert out.uns["perturbalign"]["stats"]["control"] == 1


def test_annotate_anndata_preview_mode():
    labels = ["DMSO", "Sorafenib "]
    stub = _FakeAnnData(pd.DataFrame({"perturbation": labels}))
    new_obs, audit = adapter.annotate_anndata(stub, drug_col="perturbation",
                                              inplace=False,
                                              cache=FIXTURE_CACHE, online=False)
    assert "perturbalign_status" not in stub.obs.columns  # 
    assert pd.isna(new_obs["perturbalign_drug_name"][0])  # control no
    assert new_obs["perturbalign_drug_name"][1] == "Sorafenib"
    assert audit["stats"]["control"] == 1
    assert audit["unmatched"] == []


def test_annotate_anndata_missing_column():
    stub = _FakeAnnData(pd.DataFrame({"foo": ["DMSO"]}))
    with pytest.raises(KeyError):
        adapter.annotate_anndata(stub, drug_col="perturbation")


# ---------------- 3. ChEMBL target /MoA （mock network ） ----------------

def _fake_chembl_http(url: str) -> dict:
    if "/molecule/search" in url:
        return {"molecules": [{"molecule_chembl_id": "CHEMBL1234567"}]}
    if "/mechanism" in url:
        return {"mechanisms": [
            {"mechanism_of_action": "Topoisomerase II inhibitor",
             "target_chembl_id": "CHEMBL_T1"},
            {"mechanism_of_action": "Topoisomerase II inhibitor",
             "target_chembl_id": "CHEMBL_T2"},
        ]}
    if "/target/CHEMBL_T1.json" in url:
        return {"target_components": [{"target_component_synonyms": [
            {"syn_type": "GENE_SYMBOL", "component_synonym": "TOP2A"},
            {"syn_type": "UNIPROT", "component_synonym": "P11388"},
        ]}]}
    if "/target/CHEMBL_T2.json" in url:
        return {"target_components": [{"target_component_synonyms": [
            {"syn_type": "GENE_SYMBOL", "component_synonym": "TOP2B"},
        ]}]}
    return {"molecules": []}


def test_chembl_targets_lookup(monkeypatch):
    from perturbalign.cache import SQLiteCache
    monkeypatch.setattr(annotate, "_http_get_json", _fake_chembl_http)
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = f.name
    try:
        cache = SQLiteCache(path)
        info = annotate.chembl_targets_lookup("Doxorubicin", cache=cache)
        assert info["chembl_id"] == "CHEMBL1234567"
        assert info["targets"] == ["TOP2A", "TOP2B"]   # +
        assert info["moa"] == ["Topoisomerase II inhibitor"]
        # cache ： network return
        monkeypatch.setattr(annotate, "_http_get_json",
                            lambda url: (_ for _ in ()).throw(AssertionError("no network")))
        cached = annotate.chembl_targets_lookup("Doxorubicin", cache=cache)
        assert cached == info
        cache.close()
    finally:
        os.unlink(path)


def test_chembl_lookup_offline_safe(monkeypatch):
    """offline /network exception ->    None（not      ）。"""
    monkeypatch.setattr(annotate, "_http_get_json",
                        lambda url: (_ for _ in ()).throw(OSError("offline")))
    assert annotate.chembl_targets_lookup("Anything") is None


# ---------------- 4. RDKit InChIKey（ rdkit Verify ） ----------------

def _install_fake_rdkit(monkeypatch) -> dict:
    """    rdkit：GetParentMol desalt + MolToInchiKey return    。"""
    fake = types.ModuleType("rdkit")
    chem = types.ModuleType("rdkit.Chem")
    inchi_mod = types.ModuleType("rdkit.Chem.inchi")
    std = types.ModuleType("rdkit.Chem.MolStandardize")
    stdmod = types.ModuleType("rdkit.Chem.MolStandardize.rdMolStandardize")

    class _Mol:
        def __init__(self, smiles):
            self.smiles = smiles

    def _from_smiles(smiles):
        return _Mol(smiles)

    def _parent_mol(mol):
        # ： ".Cl" is
        parent_smiles = mol.smiles.split(".")[0]
        return _Mol(parent_smiles)

    def _inchi_key(mol):
        base = {"Doxorubicin": "QZUUJTQFUZLSLG", "CCCC": "XXXXABCDEFGH"}.get(
            mol.smiles, "FAKEIK00000000")
        return base + "-JHYXSHQKSA-N"

    chem.MolFromSmiles = staticmethod(_from_smiles)
    inchi_mod.MolToInchiKey = _inchi_key
    stdmod.GetParentMol = staticmethod(_parent_mol)
    std.rdMolStandardize = stdmod
    chem.MolStandardize = std
    chem.inchi = inchi_mod
    fake.Chem = chem
    monkeypatch.setitem(sys.modules, "rdkit", fake)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", chem)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.inchi", inchi_mod)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.MolStandardize", std)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.MolStandardize.rdMolStandardize", stdmod)
    return fake


def test_parent_inchikey_with_rdkit(monkeypatch):
    _install_fake_rdkit(monkeypatch)
    assert structure.parent_inchikey("Doxorubicin.Cl") == "QZUUJTQFUZLSLG-JHYXSHQKSA-N"
    assert structure.parent_inchikey(None) is None


def test_cluster_by_parent_groups_salt_and_parent(monkeypatch):
    _install_fake_rdkit(monkeypatch)
    from perturbalign.schema import DrugEntity, AlignResult

    def rec(inp, smiles):
        return AlignResult(input=inp, status="exact",
                           entity=DrugEntity(input_string=inp, canonical_name=inp,
                                             canonical_smiles=smiles))
    mapping = [
        rec("Doxorubicin", "Doxorubicin"),
        rec("Doxorubicin HCl", "Doxorubicin.Cl"),   # ->
        rec("Trametinib", None),                    # no SMILES ->
    ]
    groups = structure.cluster_by_parent(mapping)
    assert len(groups["QZUUJTQFUZLSLG"]) == 2          # + salt form to
    assert "Trametinib" in groups[""]                  # no SMILES


def test_structure_degrades_without_rdkit():
    """rdkit    ：parent_inchikey return None，cluster   to empty  。"""
    saved = dict(sys.modules)
    for name in list(sys.modules):
        if name == "rdkit" or name.startswith("rdkit."):
            sys.modules.pop(name, None)
    try:
        assert structure.parent_inchikey("CCCC") is None
        from perturbalign.schema import DrugEntity, AlignResult
        rec = AlignResult(input="X", status="exact",
                          entity=DrugEntity(input_string="X", canonical_name="X",
                                            canonical_smiles="CCCC"))
        groups = structure.cluster_by_parent([rec])
        assert groups == {"": ["X"]}
    finally:
        sys.modules.clear()
        sys.modules.update(saved)
