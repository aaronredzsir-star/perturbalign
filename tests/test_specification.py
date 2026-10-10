"""  perturbation data specification (UPDS)   Test   。

  ：
1. PerturbationAtom:     、pconc     、 column  、fingerprint   
2. PerturbationUnit:   /  perturbation   、fingerprint   、JSON-LD   、   Verify 3. PerturbationDataset:     、DataFrame   、  extract (NumPy/PyTorch)、AnnData   
4. parse_input:    、  、column  、DataFrame         
"""
import pytest
import numpy as np
import perturbalign
from perturbalign import (
    PerturbationAtom,
    PerturbationUnit,
    PerturbationDataset,
    parse_input,
)


def test_perturbation_atom_basic():
    atom = PerturbationAtom(
        name="Doxorubicin",
        modality="small_molecule",
        dose=10.0,
        dose_unit="nM",
        duration=24.0,
        duration_unit="h",
        identifiers={"pubchem_cid": 31703, "inchikey": "AOJJSUZBOXZQNB-VTCOVDECSA-N"},
    )
    assert atom.name == "Doxorubicin"
    assert atom.modality == "small_molecule"
    assert atom.dose == 10.0
    assert atom.dose_unit == "nM"
    # Verify pconc : 10 nM = 10^-8 M -> pconc = 8.0
    assert pytest.approx(atom.pconc, rel=1e-3) == 8.0
    assert atom.is_control is False
    assert atom.canonical_string() == "Doxorubicin[10nM@24h]"

    # column column
    d = atom.to_dict()
    assert d["name"] == "Doxorubicin"
    atom_restored = PerturbationAtom.from_dict(d)
    assert atom_restored.name == atom.name
    assert atom_restored.dose == atom.dose
    assert atom_restored.pconc == atom.pconc


def test_perturbation_atom_control():
    dmso = PerturbationAtom(name="DMSO", modality="control")
    assert dmso.is_control is True

    vehicle = PerturbationAtom(name="Vehicle", modality="control")
    assert vehicle.is_control is True


def test_perturbation_unit_single_and_combo():
    a1 = PerturbationAtom(name="Trametinib", dose=10.0, dose_unit="nM")
    a2 = PerturbationAtom(name="Pictilisib", dose=1.0, dose_unit="µM")

    # single agent
    u_single = PerturbationUnit(raw_input="Trametinib 10nM", atoms=[a1])
    assert u_single.is_combination is False
    assert u_single.is_control is False
    assert u_single.primary_atom.name == "Trametinib"
    assert "Trametinib" in u_single.canonical_signature

    # perturbation
    u_combo = PerturbationUnit(raw_input="Trametinib 10nM + Pictilisib 1uM", atoms=[a1, a2])
    assert u_combo.is_combination is True
    assert len(u_combo.atoms) == 2
    assert u_combo.names == ["Trametinib", "Pictilisib"]
    assert "Pictilisib" in u_combo.canonical_signature
    assert "Trametinib" in u_combo.canonical_signature

    # JSON-LD
    jsonld = u_combo.to_jsonld()
    assert jsonld["@context"] == "https://schema.org"
    assert len(jsonld["interventions"]) == 2


def test_perturbation_unit_validation():
    # 
    u_valid = PerturbationUnit(raw_input="sgTP53", atoms=[PerturbationAtom(name="TP53", action="KO")])
    warns = u_valid.validate()
    assert len(warns) == 0

    # exception : dose
    u_invalid = PerturbationUnit(
        raw_input="Drug X -5mg",
        atoms=[PerturbationAtom(name="DrugX", dose=-5.0, dose_unit="mg/kg")]
    )
    warns = u_invalid.validate()
    assert any("NEGATIVE_DOSE" in w for w in warns)


def test_perturbation_dataset_operations():
    u1 = PerturbationUnit("Doxorubicin 10nM", [PerturbationAtom("Doxorubicin", dose=10.0, dose_unit="nM")], unified_embedding=[0.1]*512)
    u2 = PerturbationUnit("sgTP53", [PerturbationAtom("TP53", modality="genetic", action="KO")], unified_embedding=[0.2]*512)
    u3 = PerturbationUnit("DMSO", [PerturbationAtom("DMSO", modality="control")], unified_embedding=[0.0]*512)

    ds = PerturbationDataset([u1, u2, u3], dataset_id="test_exp_01")
    assert len(ds) == 3
    assert ds.is_control_mask == [False, False, True]

    # 
    sub = ds[0:2]
    assert len(sub) == 2
    assert isinstance(sub, PerturbationDataset)

    # DataFrame
    df = ds.to_dataframe()
    assert len(df) == 3
    assert "perturbation" in df.columns
    assert "primary_name" in df.columns
    assert "primary_pconc" in df.columns

    # extract
    tensors = ds.to_tensors(format="numpy")
    assert tensors["embeddings"].shape == (3, 512)
    assert tensors["doses"].shape == (3, 1)
    assert tensors["is_control"].tolist() == [False, False, True]

    # AnnData
    class MockAnnData:
        def __init__(self, obs_df):
            self.obs = obs_df
            self.obsm = {}
            self.uns = {}
    adata = MockAnnData(df.copy())
    adata_out = ds.to_anndata(adata)
    assert "X_pert_embedding" in adata_out.obsm
    assert adata_out.obsm["X_pert_embedding"].shape == (3, 512)
    assert adata_out.uns["pert_dataset_id"] == "test_exp_01"


def test_parse_input_facade():
    # 1. parse
    u1 = parse_input("Doxorubicin HCl 10nM 24h")
    assert isinstance(u1, PerturbationUnit)
    assert u1.primary_atom.name == "Doxorubicin"
    assert u1.primary_atom.dose == 10.0
    assert u1.primary_atom.dose_unit == "nM"
    assert u1.primary_atom.duration == 24.0

    # 2. parse combination
    u2 = parse_input("Trametinib 10nM + Pictilisib 1uM")
    assert u2.is_combination is True
    assert len(u2.atoms) == 2

    # 3. parse
    u3 = parse_input({"name": "Staurosporine", "dose": 50, "dose_unit": "nM", "modality": "small_molecule"})
    assert isinstance(u3, PerturbationUnit)
    assert u3.primary_atom.name == "Staurosporine"
    assert u3.primary_atom.dose == 50.0

    # 4. parse column
    ds = parse_input(["sgTP53", "DMSO 0.1%", "Trametinib 10nM"])
    assert isinstance(ds, PerturbationDataset)
    assert len(ds) == 3
    assert ds[1].is_control is True
