"""Tests for the 6-dimensional architectural optimizations in PerturbAlign.

Covers:
1. Unified facade (`perturbalign.process`) across strings, lists, DataFrames, and AnnData.
2. Holistic context model (`PerturbationContext`, `build_context`).
3. Declarative knowledge graph registration (`register_declarative_kg`).
4. Structured domain exception hierarchy (`PerturbAlignError` and subclasses).
5. Graph pathway search optimization (memoization cache & invalidation).
"""
import pytest
import pandas as pd
import numpy as np

import perturbalign as pa
from perturbalign.exceptions import (
    PerturbAlignError,
    NormalizationError,
    AmbiguousEntityError,
    UnmatchedEntityError,
    SpatiotemporalParseError,
    DimensionalMismatchError,
    RegistryError,
    GraphInferenceError,
)
from perturbalign.specification import (
    PerturbationContext,
    PerturbationUnit,
    PerturbationDataset,
    build_context,
)
from perturbalign.kg_yaml import register_declarative_kg
from perturbalign.hetero_kg import HeterogeneousKnowledgeGraph, HeteroNode, HeteroEdge


class DummyAnnData:
    """Lightweight AnnData test stub with obs, obsm, and uns."""
    def __init__(self, obs_dict):
        self.obs = pd.DataFrame(obs_dict)
        self.obsm = {}
        self.uns = {}

    def copy(self):
        d = DummyAnnData({})
        d.obs = self.obs.copy()
        d.obsm = dict(self.obsm)
        d.uns = dict(self.uns)
        return d


# ============================================================================
# 1. Unified Industrial Facade (perturbalign.process)
# ============================================================================

def test_process_single_string():
    """Test process with a single messy perturbation string returning a UPDS unit."""
    res = pa.process("Doxorubicin 10nM 24h")
    assert isinstance(res, PerturbationUnit)
    assert len(res.atoms) >= 1
    atom = res.atoms[0]
    assert "doxorubicin" in atom.name.lower()
    assert atom.dose == 10.0
    assert atom.dose_unit == "nM"
    assert atom.duration == 24.0


def test_process_list_of_strings():
    """Test process with a batch list of strings returning a PerturbationDataset."""
    inputs = ["DMSO control", "Trametinib 50nM", "TP53 KO"]
    ds = pa.process(inputs)
    assert isinstance(ds, PerturbationDataset)
    assert len(ds.units) == 3
    # Check control detection
    assert ds.units[0].is_control is True
    # Check drug
    assert "trametinib" in ds.units[1].atoms[0].name.lower()
    # Check genetic
    assert "TP53" in ds.units[2].atoms[0].name


def test_process_dataframe():
    """Test process with a pandas DataFrame."""
    df = pd.DataFrame({
        "raw_treatment": ["Doxorubicin 1uM", "DMSO", "Palbociclib 500nM"],
        "cell_line": ["A549", "MCF7", "PC9"],
    })
    annotated_df = pa.process(df, pert_col="raw_treatment", cell_line="A549")
    assert isinstance(annotated_df, pd.DataFrame)
    assert "perturbalign_drug_name" in annotated_df.columns
    assert "perturbalign_modality" in annotated_df.columns
    assert "perturbalign_is_control" in annotated_df.columns
    # Check cell line annotation injection
    assert "perturbalign_host_cell_line" in annotated_df.columns
    assert "perturbalign_host_primary_disease" in annotated_df.columns
    assert annotated_df.loc[0, "perturbalign_host_cell_line"] == "A549"


def test_process_anndata():
    """Test process with an AnnData object stub."""
    adata = DummyAnnData({
        "perturbation": ["Trametinib 10nM", "DMSO control", "Vorinostat 2uM"],
    })

    res_adata = pa.process(
        adata,
        pert_col="perturbation",
        cell_line="A549",
        add_embeddings=True,
        add_profiles=True,
    )
    assert hasattr(res_adata, "obs")
    assert "perturbalign_status" in res_adata.obs.columns
    assert "perturbalign_modality" in res_adata.obs.columns
    assert "perturbalign_host_cell_line" in res_adata.obs.columns
    assert res_adata.obs.loc[0, "perturbalign_host_cell_line"] == "A549"
    assert "X_pert_embedding" in res_adata.obsm
    assert "perturbalign_cell_line" in res_adata.uns


# ============================================================================
# 2. Holistic Context Model (PerturbationContext)
# ============================================================================

def test_build_context_holistic():
    """Test building a 3-in-1 PerturbationContext integrating agent, host, and microenvironment."""
    ctx = build_context(
        perturbation="Trametinib 100nM 48h",
        cell_line="A549",
        microenvironment={"oxygen": "1%", "matrix": "Matrigel"},
        experiment_id="EXP-2026-001"
    )
    assert isinstance(ctx, PerturbationContext)
    assert ctx.agent is not None
    assert len(ctx.agent.atoms) >= 1
    atom = ctx.agent.atoms[0]
    assert atom.name == "Trametinib"
    assert atom.dose == 100.0
    assert atom.duration == 48.0
    assert ctx.host_cell_line is not None
    assert ctx.host_cell_line.name == "A549"
    assert ctx.host_cell_line.depmap_id == "ACH-000681"
    assert ctx.microenvironment.get("matrix") == "Matrigel"
    assert ctx.microenvironment.get("experiment_id") == "EXP-2026-001"

    # Test serialization to dict
    d = ctx.to_dict()
    assert "agent" in d
    assert "host_cell_line" in d
    assert d["host_cell_line"]["depmap_id"] == "ACH-000681"
    assert d["microenvironment"]["oxygen"] == "1%"

    # Test system summary
    summary = ctx.summarize_system()
    assert "A549" in summary
    assert "Trametinib" in summary


# ============================================================================
# 3. Declarative Knowledge Graph Registration (register_declarative_kg)
# ============================================================================

def test_register_declarative_kg():
    """Test zero-code registration of custom KG entities and relations."""
    kg = HeterogeneousKnowledgeGraph(load_curated=False)
    decl_config = {
        "namespace": "oncology_trial_2026",
        "name": "oncology_trial",
        "description": "Custom trial nodes and resistance mechanisms",
        "nodes": [
            {
                "id": "Compound:PA_999",
                "node_type": "Compound",
                "name": "SuperInhibitor-999",
                "properties": {"phase": 3, "sponsor": "GlobalPharma"}
            },
            {
                "id": "Gene:TARGET_XYZ",
                "node_type": "Gene",
                "name": "XYZ1",
                "properties": {"entrez_id": 99999}
            },
            {
                "id": "Pathway:RESISTANCE_1",
                "node_type": "Pathway",
                "name": "Bypass Survival Pathway",
                "properties": {}
            }
        ],
        "edges": [
            {
                "source": "Compound:PA_999",
                "target": "Gene:TARGET_XYZ",
                "relation": "inhibits",
                "effect_sign": -1,
                "weight": 0.95
            },
            {
                "source": "Gene:TARGET_XYZ",
                "target": "Pathway:RESISTANCE_1",
                "relation": "activates",
                "effect_sign": 1,
                "weight": 0.88
            }
        ]
    }

    n_nodes, n_edges = register_declarative_kg(decl_config, kg=kg)
    assert n_nodes == 3
    assert n_edges == 2

    # Verify query
    node = kg.get_node("Compound:PA_999")
    assert node is not None
    assert node.name == "SuperInhibitor-999"

    # Search pathways
    paths = kg.search_pathways("Compound:PA_999", "Pathway:RESISTANCE_1", max_hops=3, min_score=0.1)
    assert len(paths) >= 1
    assert paths[0].overall_effect_sign == -1  # (-1) * (+1) = -1


# ============================================================================
# 4. Structured Domain Exceptions
# ============================================================================

def test_domain_exceptions_hierarchy():
    """Test that all domain-specific exceptions inherit from PerturbAlignError."""
    exc1 = NormalizationError("Compound not found", entity_name="UnknownDrug_X")
    assert isinstance(exc1, PerturbAlignError)
    assert exc1.entity_name == "UnknownDrug_X"
    assert exc1.error_code == "NORMALIZATION_ERROR"
    assert "UnknownDrug_X" in str(exc1)

    exc2 = AmbiguousEntityError("Multiple matches found", entity_name="ERK", candidates=["MAPK1", "MAPK3"])
    assert isinstance(exc2, PerturbAlignError)
    assert exc2.candidates == ["MAPK1", "MAPK3"]

    exc3 = DimensionalMismatchError("10 mg/kg", "10 uM", reason="Incompatible dose conversion")
    assert isinstance(exc3, PerturbAlignError)
    assert exc3.error_code == "DIMENSIONAL_MISMATCH_ERROR"

    exc4 = GraphInferenceError("DrugA", "GeneB", reason="Graph query failed")
    assert isinstance(exc4, PerturbAlignError)
    assert exc4.error_code == "GRAPH_INFERENCE_ERROR"


# ============================================================================
# 5. Graph Search Optimization & Invalidation
# ============================================================================

def test_graph_search_caching_and_invalidation():
    """Test pathway search memoization and cache invalidation upon graph mutation."""
    kg = HeterogeneousKnowledgeGraph(load_curated=False)
    kg.add_node("A", "DrugA", "Compound")
    kg.add_node("B", "GeneB", "Gene")
    kg.add_node("C", "PathC", "Pathway")

    kg.add_edge("A", "B", "target", effect_sign=-1, weight=1.0)
    kg.add_edge("B", "C", "activates", effect_sign=1, weight=1.0)

    # Initial search populates cache
    paths1 = kg.search_pathways("A", "C", max_hops=3, min_score=0.1, max_paths=10)
    assert len(paths1) == 1
    cache_key = ("A", "C", 3, 0.1, 10)
    assert cache_key in kg._path_cache

    # Subsequent search hits cache
    paths2 = kg.search_pathways("A", "C", max_hops=3, min_score=0.1, max_paths=10)
    assert len(paths2) == 1

    # Adding a new node invalidates cache
    kg.add_node("D", "GeneD", "Gene")
    assert len(kg._path_cache) == 0  # Invalidation verified!

    kg.add_edge("A", "D", "off_target", effect_sign=1, weight=1.0)
    assert len(kg._path_cache) == 0
    kg.add_edge("D", "C", "suppresses", effect_sign=-1, weight=1.0)

    # Re-searching finds 2 paths now
    paths3 = kg.search_pathways("A", "C", max_hops=3, min_score=0.1, max_paths=10)
    assert len(paths3) == 2
    assert cache_key in kg._path_cache
