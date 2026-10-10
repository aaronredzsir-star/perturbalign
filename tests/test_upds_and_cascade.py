"""Tests for UPDS specification, DoRothEA causal cascades, pharma code disambiguation, and MCP protocol services."""

import math
import pandas as pd
import pytest

from perturbalign.cascade import infer_causal_cascade
from perturbalign.compound import resolve_investigational_code, get_pubchem_compound
from perturbalign.facade import (
    enrich,
    standardize_anndata,
    validate_upds,
    UPDS_REQUIRED_OBS_COLUMNS,
    UPDS_ALLOWED_MODALITIES,
)
from perturbalign.mcp_server import PerturbAlignMCPServer


class MockAnnData:
    """Lightweight mock AnnData object for testing."""
    def __init__(self, obs_df: pd.DataFrame):
        self.obs = obs_df.copy()
        self.uns = {}

    def copy(self):
        new_obj = MockAnnData(self.obs.copy())
        new_obj.uns = dict(self.uns)
        return new_obj


# --------------------------------------------------------------------------
# 1. Pharma investigational code resolution and prefix disambiguation tests
# --------------------------------------------------------------------------
def test_investigational_code_resolution():
    # Hyphenated and compact form equivalent recognition
    res1 = resolve_investigational_code("AZD-6244")
    res2 = resolve_investigational_code("AZD6244")
    res3 = resolve_investigational_code("azd 6244")

    assert res1 is not None
    assert res1["canonical_name"] == "Selumetinib"
    assert "MAP2K1" in res1["targets"]
    assert res1["developer"] == "AstraZeneca"
    assert res1["canonical_name"] == res2["canonical_name"] == res3["canonical_name"]

    # Canonical investigational code resolution
    pd_res = resolve_investigational_code("PD-0332991")
    assert pd_res is not None
    assert pd_res["canonical_name"] == "Palbociclib"
    assert "CDK4" in pd_res["targets"]

    gdc_res = resolve_investigational_code("GDC-0941")
    assert gdc_res is not None
    assert gdc_res["canonical_name"] == "Pictilisib"

    bms_res = resolve_investigational_code("BMS-354825")
    assert bms_res is not None
    assert bms_res["canonical_name"] == "Dasatinib"

    abt_res = resolve_investigational_code("ABT-263")
    assert abt_res is not None
    assert abt_res["canonical_name"] == "Navitoclax"

    bay_res = resolve_investigational_code("BAY 43-9006")
    assert bay_res is not None
    assert bay_res["canonical_name"] == "Sorafenib"


def test_compound_pubchem_resolution_via_lab_code():
    # Offline query of AZD-6244 automatically redirects to canonical Selumetinib chemical structure
    pub = get_pubchem_compound("AZD-6244", online=False)
    assert pub is not None
    assert "canonical_smiles" in pub
    assert len(pub["canonical_smiles"]) > 10
    assert pub["molecular_weight"] > 400


# --------------------------------------------------------------------------
# 2. DoRothEA causal cascade inference tests
# --------------------------------------------------------------------------
def test_causal_cascade_inference():
    # Kinase targets: MAP2K1/MAP2K2 -> converge to AP-1/MYC/EGR1 -> regulate DoRothEA grade A response regulons
    cascade = infer_causal_cascade(["MAP2K1", "MAP2K2"])
    assert "MAP2K1" in cascade["direct_targets"]
    assert any("MAPK/ERK" in pw for pw in cascade["pathway_names"])
    assert "JUN" in cascade["downstream_tfs"]
    assert "FOS" in cascade["downstream_tfs"]
    assert "DUSP1" in cascade["response_regulon"] or "CCND1" in cascade["response_regulon"]
    assert cascade["confidence"] == "high"

    # Direct transcription factor target: TP53
    p53_cascade = infer_causal_cascade(["TP53"])
    assert "TP53" in p53_cascade["downstream_tfs"]
    assert "CDKN1A" in p53_cascade["response_regulon"]
    assert "BAX" in p53_cascade["response_regulon"]

    # Empty input fault tolerance
    empty_cascade = infer_causal_cascade([])
    assert empty_cascade["confidence"] == "unspecified"
    assert len(empty_cascade["downstream_tfs"]) == 0


def test_enrichment_includes_causal_cascade():
    # Small molecule enrichment includes causal_cascade
    res = enrich("Trametinib 10nM", online=False)
    assert "causal_cascade" in res.properties
    cas = res.properties["causal_cascade"]
    assert "JUN" in cas["downstream_tfs"] or "FOS" in cas["downstream_tfs"]

    # Genetic knockout enrichment includes causal_cascade
    gene_res = enrich("Trp53 KO", organism="mouse", online=False)
    assert "causal_cascade" in gene_res.properties
    assert "CDKN1A" in gene_res.properties["causal_cascade"]["response_regulon"]


# --------------------------------------------------------------------------
# 3. UPDS single-cell metadata standardization tests
# --------------------------------------------------------------------------
def test_standardize_anndata_and_upds_validation():
    # Construct test dataframe containing multimodal perturbations
    df = pd.DataFrame({
        "perturbation": [
            "AZD-6244 10uM 24h",
            "shTrp53",
            "TNF-alpha 50ng/ml 6h",
            "Hypoxia 48h",
            "DMSO control",
        ],
        "batch": ["b1", "b1", "b2", "b2", "b1"],
    })
    adata = MockAnnData(df)

    # Execute standardization
    std_adata = standardize_anndata(adata, obs_col="perturbation", online=False)

    obs = std_adata.obs
    # 1. Verify 10 required UPDS standard columns exist
    for col in UPDS_REQUIRED_OBS_COLUMNS:
        assert col in obs.columns, f"Missing UPDS standard column: {col}"

    # 2. Verify controlled vocabularies
    for m in obs["pert_modality"]:
        assert m in UPDS_ALLOWED_MODALITIES, f"Invalid modality controlled vocabulary: {m}"

    assert obs["pert_modality"].iloc[0] == "small_molecule"
    assert obs["pert_modality"].iloc[1] == "rnai" or obs["pert_modality"].iloc[1] == "crispr_ko"
    assert obs["pert_modality"].iloc[2] == "biologic"
    assert obs["pert_modality"].iloc[3] == "environmental"
    assert obs["pert_modality"].iloc[4] == "control"

    # 3. Verify control flags and dose normalization
    assert not obs["pert_is_control"].iloc[0]
    assert obs["pert_is_control"].iloc[4]  # DMSO is True

    # 10uM normalized micromolar concentration should be 10.0
    assert obs["pert_dose_norm_um"].iloc[0] == 10.0
    assert obs["pert_duration_h"].iloc[0] == 24.0
    assert obs["pert_duration_h"].iloc[3] == 48.0

    # 4. Verify gene alignment to human HGNC
    assert obs["pert_human_ortholog"].iloc[1] == "TP53"

    # 5. Verify uns audit dictionary structure
    assert "perturbalign" in std_adata.uns
    assert std_adata.uns["perturbalign"]["schema_version"] == "1.0.0"
    assert std_adata.uns["perturbalign"]["audit_summary"]["n_cells"] == 5

    # 6. Verify UPDS validator
    val_report = validate_upds(std_adata)
    assert val_report["is_valid"] is True
    assert len(val_report["errors"]) == 0


# --------------------------------------------------------------------------
# 4. MCP Server protocol and tool tests
# --------------------------------------------------------------------------
def test_mcp_server_new_tools():
    server = PerturbAlignMCPServer()

    # Check that tools list contains new tools
    tool_names = [t["name"] for t in server._tools]
    assert "perturbalign_enrich" in tool_names
    assert "perturbalign_resolve_lab_code" in tool_names
    assert "perturbalign_align_ortholog" in tool_names
    assert "perturbalign_standardize_metadata" in tool_names

    # Call perturbalign_resolve_lab_code
    lab_res = server.call_tool("perturbalign_resolve_lab_code", {"code": "AZD-6244"})
    assert lab_res["canonical_name"] == "Selumetinib"
    assert lab_res["developer"] == "AstraZeneca"

    # Call perturbalign_enrich
    enrich_res = server.call_tool("perturbalign_enrich", {"query": "Palbociclib 1uM", "online": False})
    assert enrich_res["canonical_name"] == "Palbociclib"
    assert "causal_cascade" in enrich_res["properties"]
    assert "E2F1" in enrich_res["properties"]["causal_cascade"]["downstream_tfs"]

    # Call perturbalign_standardize_metadata
    records = [
        {"perturbation": "Gefitinib 1uM 24h"},
        {"perturbation": "Vehicle control"},
    ]
    meta_res = server.call_tool("perturbalign_standardize_metadata", {"records": records})
    assert "records" in meta_res
    assert len(meta_res["records"]) == 2
    assert meta_res["records"][0]["pert_canonical_name"] == "Gefitinib"
    assert meta_res["records"][0]["pert_dose_norm_um"] == 1.0
    assert meta_res["records"][1]["pert_is_control"] is True


# --------------------------------------------------------------------------
# 5. 2: cell line causal pruning confidence Test
# --------------------------------------------------------------------------
def test_context_specific_causal_cascade_pruning():
    # Test K562 (TP53 /mutation ): TOP2A target
    # TOP2A TP53
    # in K562 TP53 pruning ，confidence context_attenuated
    k562_res = infer_causal_cascade(["TOP2A"], cell_line="K562")
    assert "TP53" in k562_res["downstream_tfs"]
    assert "TP53" not in k562_res["active_tfs"]
    assert len(k562_res["pruned_tfs"]) == 1
    assert k562_res["pruned_tfs"][0]["tf"] == "TP53"
    assert "mutant" in k562_res["pruned_tfs"][0]["reason"].lower() or "null" in k562_res["pruned_tfs"][0]["reason"].lower()
    assert k562_res["confidence"] == "context_attenuated"
    assert k562_res["cascade_confidence_score"] == 0.65

    # Test A549 (TP53-WT, KRAS mutation ): MAP2K1/MAP2K2 (MEK)
    # A549 TP53 is WT，MAPK row AP-1 (JUN/FOS) MYC
    a549_res = infer_causal_cascade(["MAP2K1", "MAP2K2"], cell_line="A549")
    assert "JUN" in a549_res["active_tfs"]
    assert "MYC" in a549_res["active_tfs"]
    assert len(a549_res["pruned_tfs"]) == 0
    assert a549_res["confidence"] == "high"
    assert a549_res["cascade_confidence_score"] >= 0.85

    # all no
    global_res = infer_causal_cascade(["MAP2K1"])
    assert global_res["cell_line_context"] is None
    assert global_res["cascade_confidence_score"] == 0.85


# --------------------------------------------------------------------------
# 6. pharma code offline SMILES extract
# --------------------------------------------------------------------------
def test_investigational_code_offline_smiles_enrichment():
    from perturbalign.compound import query_compound, get_pubchem_compound

    # 1. PubChem parse (offline from investigational TSV )
    p_res = get_pubchem_compound("AZD-6244", online=False)
    assert p_res is not None
    assert p_res["canonical_name"].lower() == "selumetinib"
    assert "c1c" in p_res["canonical_smiles"].lower()

    # 2. all query_compound parse
    c_res = query_compound("AZD-6244", online=False)
    assert c_res is not None
    assert c_res["pubchem"]["canonical_name"].lower() == "selumetinib"
    assert "c1c" in c_res["pubchem"]["canonical_smiles"].lower()
    assert "MAP2K1" in c_res["drugbank_annotations"]["targets"]

    # offline parse GDC-0941
    gdc_res = get_pubchem_compound("GDC0941", online=False)
    assert gdc_res is not None
    assert gdc_res["canonical_name"].lower() == "pictilisib"
    assert "c1" in gdc_res["canonical_smiles"].lower()


# --------------------------------------------------------------------------
# 7. standardize_anndata cell_line column
# --------------------------------------------------------------------------
def test_standardize_anndata_cell_line_propagation():
    obs_df = pd.DataFrame({
        "perturbation": ["Doxorubicin 1uM", "DMSO"],
        "cell_line": ["K562", "K562"],
    })
    adata = MockAnnData(obs_df)
    std_adata = standardize_anndata(adata, obs_col="perturbation", online=False)

    # Verify data
    assert "pert_canonical_name" in std_adata.obs.columns
    assert std_adata.obs["pert_canonical_name"].iloc[0] == "Doxorubicin"
    assert std_adata.obs["pert_canonical_name"].iloc[1] == "DMSO"

    # Verify uns host_context
    assert "perturbalign" in std_adata.uns
    entity_kb = std_adata.uns["perturbalign"]["entity_knowledge"]
    dox_kb = entity_kb.get("Doxorubicin 1uM")
    assert dox_kb is not None
    # Doxorubicin targets TOP2A, in K562 (TP53 null) pruning
    cascade = dox_kb["properties"]["causal_cascade"]
    assert "TP53" in cascade["downstream_tfs"]
    assert "TP53" not in cascade["active_tfs"]
    assert len(cascade["pruned_tfs"]) >= 1

