"""Unit tests for the schema refactor: typed enums and polymorphic entity hierarchy."""
import pytest
from perturbalign.schema import (
    AlignResult,
    AlignStatus,
    BiologicEntity,
    Candidate,
    DrugEntity,
    EnvironmentalEntity,
    GeneticEntity,
    Modality,
    PerturbationEntity,
    SmallMoleculeEntity,
    STATUS_EXACT,
    STATUS_CONTROL,
    STATUS_UNMATCHED,
    MODALITY_SMALL_MOLECULE,
    MODALITY_GENETIC,
    MODALITY_BIOLOGIC,
    MODALITY_ENVIRONMENTAL,
)


def test_modality_enum_str_compatibility():
    """Modality(str, Enum)        100%   。"""
    assert Modality.SMALL_MOLECULE == "small_molecule"
    assert Modality.SMALL_MOLECULE == MODALITY_SMALL_MOLECULE
    assert isinstance(Modality.SMALL_MOLECULE, str)
    assert Modality.GENETIC == "genetic"
    assert Modality.BIOLOGIC == "biologic"
    assert Modality.ENVIRONMENTAL == "environmental"
    assert Modality.CONTROL == "control"
    assert Modality.COMBO == "combo"
    assert Modality.EMPTY == "empty"


def test_align_status_enum_str_compatibility():
    """AlignStatus(str, Enum)        100%   。"""
    assert AlignStatus.EXACT == "exact"
    assert AlignStatus.EXACT == STATUS_EXACT
    assert isinstance(AlignStatus.EXACT, str)
    assert AlignStatus.CONTROL == "control"
    assert AlignStatus.CONTROL == STATUS_CONTROL
    assert AlignStatus.UNMATCHED == "unmatched"
    assert AlignStatus.UNMATCHED == STATUS_UNMATCHED
    assert AlignStatus.ONLINE == "online"
    assert AlignStatus.AMBIGUOUS == "ambiguous"
    assert AlignStatus.FUZZY == "fuzzy"


def test_small_molecule_entity():
    """Verify DrugEntity / SmallMoleculeEntity and  properties    。"""
    ent = DrugEntity(
        input_string="Doxorubicin 10nM",
        canonical_name="Doxorubicin",
        canonical_smiles="CC1...",
        inchikey="AOZJ...",
        pubchem_cid=31703,
        chembl_id="CHEMBL53463",
        targets=["TOP2A"],
        moa="Topoisomerase II inhibitor",
    )
    assert isinstance(ent, PerturbationEntity)
    assert ent.modality == Modality.SMALL_MOLECULE
    assert ent.canonical_name == "Doxorubicin"
    assert ent.pubchem_cid == 31703
    assert SmallMoleculeEntity is DrugEntity

    d = ent.as_dict()
    assert d["canonical_name"] == "Doxorubicin"
    assert d["modality"] == "small_molecule"
    assert d["chembl_id"] == "CHEMBL53463"


def test_genetic_entity():
    """Verify GeneticEntity properties 、         。"""
    gent = GeneticEntity(
        input_string="sgTP53",
        gene_symbol="TP53",
        action="KO",
        ensembl_id="ENSG00000141510",
        entrez_id="7157",
        guide_id="sgTP53_1",
        mutation="R175H",
    )
    assert isinstance(gent, PerturbationEntity)
    assert gent.modality == Modality.GENETIC
    assert gent.canonical_name == "TP53"  # gene_symbol
    assert gent.action == "KO"
    assert gent.ensembl_id == "ENSG00000141510"

    d = gent.as_dict()
    assert d["modality"] == "genetic"
    assert d["gene_symbol"] == "TP53"
    assert d["action"] == "KO"


def test_biologic_entity():
    """Verify BiologicEntity and  properties 。"""
    bent = BiologicEntity(
        input_string="IL-6",
        canonical_name="IL-6",
        category="cytokine",
        hgnc_or_uniprot="P05231",
        target_antigen="IL6R",
    )
    assert isinstance(bent, PerturbationEntity)
    assert bent.modality == Modality.BIOLOGIC
    assert bent.category == "cytokine"
    assert bent.target_antigen == "IL6R"

    d = bent.as_dict()
    assert d["category"] == "cytokine"
    assert d["target_antigen"] == "IL6R"


def test_environmental_entity():
    """Verify EnvironmentalEntity and  properties 。"""
    eent = EnvironmentalEntity(
        input_string="Hypoxia 1% O2",
        canonical_name="Hypoxia",
        category="oxygen",
        parameter="1% O2",
    )
    assert isinstance(eent, PerturbationEntity)
    assert eent.modality == Modality.ENVIRONMENTAL
    assert eent.parameter == "1% O2"

    d = eent.as_dict()
    assert d["category"] == "oxygen"
    assert d["parameter"] == "1% O2"


def test_align_result_with_polymorphic_entities():
    """Verify AlignResult            。"""
    gent = GeneticEntity(input_string="sgMYC", gene_symbol="MYC", action="KO")
    cand = Candidate(entity=gent, score=1.0, matched_by="test")
    res = AlignResult(
        input="sgMYC",
        status=AlignStatus.UNMATCHED,
        modality=Modality.GENETIC,
        genetic_target="MYC",
        genetic_action="KO",
        entity=gent,
        candidates=[cand],
    )
    assert res.entity is gent
    assert res.entity.modality == "genetic"
    assert res.modality == "genetic"
    assert len(res.candidates) == 1
    assert res.candidates[0].entity.canonical_name == "MYC"

    d = res.as_dict()
    assert d["entity"]["modality"] == "genetic"
    assert d["entity"]["gene_symbol"] == "MYC"
