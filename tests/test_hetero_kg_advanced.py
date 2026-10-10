"""all    knowledge graph       Test    (HeteroKG Advanced & Causal Pathways).

  ：
1. causal   effect_sign           overall_effect_sign (+1, -1, 0)
2.           (PrimeKG, Hetionet, BioGRID, DoRothEA, STITCH, SynLethDB, PROGENy)
3. knowledge graph JSON / Gzip  column   offline     
4.    Meta-path         (Compound -> Gene -> Pathway etc. )
5. perturbation     pathway        (predict_pathway_impact)
6.       -             
"""
import gzip
import json
import os
import tempfile
import pytest

import perturbalign
from perturbalign import (
    HeteroNode,
    HeteroEdge,
    HeteroPath,
    HeterogeneousKnowledgeGraph,
    global_hetero_kg,
    search_hetero_pathways,
    meta_path_search,
    predict_pathway_impact,
    load_all_curated_kg,
    load_primekg,
    load_synlethdb,
    load_progeny_pathways,
)


def test_causal_effect_signs():
    # Test causal
    e1 = HeteroEdge(source="Compound:T", target="Gene:M", relation="inhibits", weight=0.99, effect_sign=-1)
    e2 = HeteroEdge(source="Gene:M", target="Gene:E", relation="phosphorylates", weight=0.99, effect_sign=1)
    e3 = HeteroEdge(source="Gene:E", target="Gene:X", relation="inhibits", weight=0.95, effect_sign=-1)

    # (-1) * (+1) = (-1)
    p1 = HeteroPath(
        source_id="Compound:T",
        target_id="Gene:E",
        hops=2,
        nodes=[HeteroNode("Compound:T", "T", "Compound"), HeteroNode("Gene:M", "M", "Gene"), HeteroNode("Gene:E", "E", "Gene")],
        edges=[e1, e2],
        confidence_score=0.98,
    )
    assert p1.overall_effect_sign == -1

    # (-1) * (+1) * (-1) = (+1) ( causal )
    p2 = HeteroPath(
        source_id="Compound:T",
        target_id="Gene:X",
        hops=3,
        nodes=[HeteroNode("Compound:T", "T", "Compound"), HeteroNode("Gene:M", "M", "Gene"), HeteroNode("Gene:E", "E", "Gene"), HeteroNode("Gene:X", "X", "Gene")],
        edges=[e1, e2, e3],
        confidence_score=0.93,
    )
    assert p2.overall_effect_sign == 1
    d = p2.to_dict()
    assert d["overall_effect_sign"] == 1


def test_curated_database_loading():
    kg = HeterogeneousKnowledgeGraph(load_curated=True)
    assert len(kg.nodes) >= 40
    # Verify in
    node_types = {n.node_type for n in kg.nodes.values()}
    assert "Compound" in node_types
    assert "Gene" in node_types
    assert "Pathway" in node_types
    assert "Disease" in node_types


def test_primekg_clinical_indications():
    # Verify Osimertinib -> NSCLC, Trametinib -> Melanoma, Imatinib -> CML
    paths_osi = search_hetero_pathways("Osimertinib", "NSCLC", max_hops=1)
    assert len(paths_osi) > 0
    assert paths_osi[0].edges[0].relation == "indicated_for"

    paths_tra = search_hetero_pathways("Trametinib", "Melanoma", max_hops=1)
    assert len(paths_tra) > 0
    assert paths_tra[0].edges[0].relation == "indicated_for"

    paths_ima = search_hetero_pathways("Imatinib", "CML", max_hops=1)
    assert len(paths_ima) > 0
    assert paths_ima[0].edges[0].relation == "indicated_for"


def test_synlethdb_synthetic_lethality():
    # Verify
    paths_brca = search_hetero_pathways("BRCA1", "PARP1", max_hops=1)
    assert len(paths_brca) > 0
    assert paths_brca[0].edges[0].relation == "synthetic_lethal_with"

    paths_tp53 = search_hetero_pathways("TP53", "CHEK2", max_hops=1)
    assert len(paths_tp53) > 0
    assert any(e.relation == "synthetic_lethal_with" for e in paths_tp53[0].edges)


def test_progeny_14_pathways():
    kg = HeterogeneousKnowledgeGraph(load_curated=True)
    # Verify all 14 pathway
    expected = ["EGFR", "MAPK", "NFkB", "p53", "PI3K", "TGFb", "TNFa", "Trail", "VEGF", "WNT", "JAK_STAT", "Androgen", "Estrogen", "Hypoxia"]
    for pw in expected:
        node_id = f"Pathway:{pw}"
        assert node_id in kg.nodes, f"Missing PROGENy pathway: {node_id}"


def test_meta_path_search():
    # Compound -> Gene -> Pathway
    paths = meta_path_search("Trametinib", meta_path=["Compound", "Gene", "Pathway"], max_paths=5)
    assert len(paths) > 0
    top = paths[0]
    assert len(top.nodes) == 3
    assert top.nodes[0].node_type == "Compound"
    assert top.nodes[1].node_type == "Gene"
    assert top.nodes[2].node_type == "Pathway"

    # Compound -> Gene -> TF -> Gene (Doxorubicin -> TOP2A -> TP53 -> CDKN1A)
    paths_tf = meta_path_search("Doxorubicin", meta_path=["Compound", "Gene", "TF", "Gene"], max_paths=5)
    assert len(paths_tf) > 0
    top_tf = paths_tf[0]
    assert [n.node_type for n in top_tf.nodes] == ["Compound", "Gene", "TF", "Gene"]

    # Compound -> Gene -> Gene -> TF -> Gene (Trametinib -> MAP2K1 -> MAPK1 -> MYC -> CDKN1A)
    paths_long = meta_path_search("Trametinib", meta_path=["Compound", "Gene", "Gene", "TF", "Gene"], max_paths=5)
    assert len(paths_long) > 0
    top_long = paths_long[0]
    assert [n.node_type for n in top_long.nodes] == ["Compound", "Gene", "Gene", "TF", "Gene"]


def test_predict_pathway_impact():
    # Trametinib is MEK ， MAPK pathway has
    pred = predict_pathway_impact("Trametinib", pathway="Pathway:MAPK_Cascade")
    assert "MAPK/ERK Signaling Pathway" in pred["pathways"]
    mapk_info = pred["pathways"]["MAPK/ERK Signaling Pathway"]
    assert mapk_info["impact_score"] < 0
    assert mapk_info["direction"] == "inhibited"
    assert len(mapk_info["evidence_paths"]) > 0

    # TP53 perturbation Apoptosis pathway has
    pred_p53 = predict_pathway_impact("TP53", pathway="Pathway:Apoptosis")
    assert "Programmed Cell Death / Apoptosis" in pred_p53["pathways"]
    apop_info = pred_p53["pathways"]["Programmed Cell Death / Apoptosis"]
    assert apop_info["impact_score"] > 0
    assert apop_info["direction"] == "activated"


def test_json_gzip_serialization():
    kg = HeterogeneousKnowledgeGraph(load_curated=True)
    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        kg.save_to_json(tmp_path)
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 100

        # 
        new_kg = HeterogeneousKnowledgeGraph(load_curated=False)
        count = new_kg.load_from_json(tmp_path)
        assert count > 0
        assert len(new_kg.nodes) >= len(kg.nodes)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
