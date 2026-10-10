"""Tests for perturbation knowledge enrichment and cross-species orthology mapping."""
import pytest
import perturbalign as pa
from perturbalign.schema import EnrichedPerturbation


def test_small_molecule_knowledge_enrichment_offline():
    """Verify small molecule offline standardization and investigational code resolution."""
    res = pa.enrich("AZD-6244 100nM 24h", online=False)
    assert isinstance(res, EnrichedPerturbation)
    assert res.is_control is False
    assert res.modality == "small_molecule"

    # Dose and duration extraction
    assert res.dose == 100.0
    assert res.dose_unit == "nM"
    assert res.duration == 24.0
    assert res.duration_unit == "h"

    # Standard canonical name and SMILES from offline TSV
    assert res.canonical_name == "Selumetinib"
    assert res.identifiers.get("smiles") is not None
    assert "C(=O)NOCCO" in res.identifiers.get("smiles", "")


@pytest.mark.network
def test_small_molecule_knowledge_enrichment_online():
    """Verify small molecule live online enrichment with PubChem and ChEMBL."""
    res = pa.enrich("Gleevec 100nM 24h", online=True)
    assert isinstance(res, EnrichedPerturbation)
    assert res.is_control is False
    assert res.modality == "small_molecule"

    assert res.dose == 100.0
    assert res.dose_unit == "nM"
    assert res.duration == 24.0
    assert res.duration_unit == "h"

    assert res.identifiers.get("pubchem_cid") == 5291
    assert "CC1=C(C=C(C=C1)NC(=O)" in res.identifiers.get("smiles", "")
    assert res.identifiers.get("chembl_id") == "CHEMBL941"


def test_genetic_knockout_and_cross_species_orthologs():
    """Verify gene knockout input -> species orthologs + annotations + pathways."""
    # 1. Knockout input with mouse gene
    res_mouse = pa.enrich("sgTrp53", online=False)
    assert res_mouse.modality == "genetic"
    assert res_mouse.identifiers.get("organism") == "mouse"

    # Species ortholog alignment
    orths = res_mouse.properties.get("orthologs", {})
    assert "human" in orths
    assert orths["human"]["symbol"] == "TP53"
    assert orths["human"]["entrez_id"] == "7157"

    assert "mouse" in orths
    assert orths["mouse"]["symbol"] == "Trp53"
    assert orths["mouse"]["entrez_id"] == "22059"

    assert "rat" in orths
    assert orths["rat"]["symbol"] == "Tp53"

    props = res_mouse.properties
    assert "tumor suppressor" in props.get("summary", "").lower()
    assert any("p53" in p.lower() for p in props.get("pathways", []))
    assert "MDM2" in [p.upper() for p in props.get("ppi_partners", [])]

    # 2. Human input
    res_human = pa.enrich("KRAS_KO", online=False)
    assert res_human.identifiers.get("organism") == "human"
    kras_orths = res_human.properties.get("orthologs", {})
    assert kras_orths["mouse"]["symbol"] == "Kras"
    assert kras_orths["mouse"]["entrez_id"] == "16653"


def test_biologic_cytokine_enrichment():
    """Verify biologic/cytokine -> receptor + pathway."""
    res = pa.enrich("TNF-alpha 10ng/ml", online=False)
    assert res.modality == "biologic"
    assert res.dose == 10.0
    assert res.dose_unit == "ng/ml"
    assert res.identifiers.get("uniprot") == "P01375"

    props = res.properties
    assert "TNFRSF1A" in props.get("receptors", [])
    assert any("NF-kappa B" in p for p in props.get("pathways", []))


def test_environmental_stress_enrichment():
    """Verify environmental stress -> sensors + marker genes."""
    res = pa.enrich("Hypoxia 1% O2 24h", online=False)
    assert res.modality == "environmental"
    assert "HIF1A" in res.properties.get("sensor_transcription_factors", [])
    assert "VEGFA" in res.properties.get("marker_genes", [])
    assert "SLC2A1" in res.properties.get("marker_genes", [])


def test_enrich_batch_and_json():
    """Verify batch serialization and output."""
    queries = ["DMSO", "AZD-6244 100nM", "sgTrp53"]
    results = pa.enrich_batch(queries, online=False)
    assert len(results) == 3
    assert results[0].is_control is True
    assert results[1].modality == "small_molecule"
    assert results[2].modality == "genetic"

    json_str = results[1].to_json()
    assert "canonical_name" in json_str
    assert "Selumetinib" in json_str


@pytest.mark.network
def test_real_world_expanded_small_molecules_online():
    """Verify live online small molecule search."""
    res_ola = pa.enrich("Olaparib 10uM", online=True)
    assert res_ola.modality == "small_molecule"
    assert "PARP1" in res_ola.properties.get("targets", [])


def test_real_world_expanded_cytokines():
    """Verify expanded cytokines parse receptors and pathways."""
    # 1. IL1B
    res_il1b = pa.enrich("IL-1beta 50ng/ml", online=False)
    assert res_il1b.modality == "biologic"
    assert "IL1R1" in res_il1b.properties.get("receptors", [])
    assert any("IL-1" in p for p in res_il1b.properties.get("pathways", []))

    # 2. VEGFA
    res_vegf = pa.enrich("VEGFA 20ng/ml", online=False)
    assert "FLT1" in res_vegf.properties.get("receptors", []) or "KDR" in res_vegf.properties.get("receptors", [])
    assert any("angiogenesis" in p.lower() or "vegf" in p.lower() for p in res_vegf.properties.get("pathways", []))

    # 3. TGFB1
    res_tgf = pa.enrich("TGF-beta1", online=False)
    assert "TGFBR1" in res_tgf.properties.get("receptors", [])


def test_real_world_expanded_microenvironments():
    """Verify MSigDB Hallmark stress, oxidative stress, starvation."""
    # 1. starvation
    res_starv = pa.enrich("Glucose starvation 48h", online=False)
    assert res_starv.modality == "environmental"
    assert "ATF4" in res_starv.properties.get("sensor_transcription_factors", [])
    assert "DDIT3" in res_starv.properties.get("marker_genes", [])

    # 2. oxidative stress / ROS
    res_ox = pa.enrich("Oxidative stress H2O2", online=False)
    assert "NFE2L2" in res_ox.properties.get("sensor_transcription_factors", [])
    assert "HMOX1" in res_ox.properties.get("marker_genes", [])

    # 3. Acidosis
    res_acid = pa.enrich("Extracellular acidosis pH 6.5", online=False)
    assert "SLC9A1" in res_acid.properties.get("marker_genes", []) or "CA9" in res_acid.properties.get("marker_genes", [])


def test_scientific_honesty_unannotated_no_fake_data():
    """Verify scientific integrity: unknown compound does not fabricate data."""
    from perturbalign.compound import get_l1000_profile, get_drugbank_annotations

    # 1. Compound L1000: no fake genes
    l1k = get_l1000_profile("TotallyUnknownCompoundXYZ_999", online=False)
    assert l1k["has_profile"] is False
    assert l1k["top_upregulated_genes"] == []
    assert l1k["top_downregulated_genes"] == []
    assert l1k["landmark_signature_vector"] is None

    # 2. Compound DrugBank: no fake targets
    db_info = get_drugbank_annotations("TotallyUnknownCompoundXYZ_999", online=False)
    assert "Cellular Target" not in db_info.get("targets", [])
    assert db_info.get("targets") == []
    assert db_info.get("moa") is None
