"""  data pathway knowledge graph      (HeteroKG)   Test   。

  ：
1. heterogeneous        (HeteroNode, HeteroEdge)
2.   causal pathway    (Multi-hop Meta-path Search):
   - small molecule     ->      ->      ->   gene (Trametinib -> CDKN1A)
   -     -> DNA      ->   pathway ->      (Doxorubicin -> Apoptosis)
   - gene mutation /knockout ->        (TP53 -> CHEK2, BRCA1 -> PARP1)
3.       extract (Neighborhood Subgraph Extraction)
4. PyTorch Geometric (PyG) heterogeneous  data      (to_pyg_data)
5.      API   Verify (search_hetero_pathways, get_hetero_subgraph)
"""
import pytest
import perturbalign
from perturbalign import (
    HeteroNode,
    HeteroEdge,
    HeteroPath,
    HeteroSubgraph,
    HeterogeneousKnowledgeGraph,
    global_hetero_kg,
    search_hetero_pathways,
    get_hetero_subgraph,
)


def test_hetero_node_edge_model():
    node = HeteroNode(id="Compound:Aspirin", name="Aspirin", node_type="Compound", attributes={"mw": 180.16})
    assert node.id == "Compound:Aspirin"
    assert node.node_type == "Compound"
    assert node.to_dict()["attributes"]["mw"] == 180.16

    edge = HeteroEdge(source="Compound:Aspirin", target="Gene:PTGS1", relation="inhibits", weight=0.95, source_db="DrugBank")
    assert edge.weight == 0.95
    assert edge.source_db == "DrugBank"


def test_multi_hop_pathway_trametinib_to_cdkn1a():
    paths = search_hetero_pathways("Trametinib", "CDKN1A", max_hops=4, min_score=0.5)
    assert len(paths) > 0
    top_path = paths[0]
    assert isinstance(top_path, HeteroPath)
    assert top_path.source_id == "Compound:Trametinib"
    assert top_path.target_id == "Gene:CDKN1A"
    assert top_path.hops <= 4
    assert top_path.confidence_score > 0.5

    # Verify
    fmt = top_path.format_path()
    assert "Compound: Trametinib" in fmt
    assert "Gene: CDKN1A" in fmt
    # Verify MEK (MAP2K1) or ERK (MAPK1) or MYC
    node_names = [n.name for n in top_path.nodes]
    assert "MAP2K1" in node_names or "MAPK1" in node_names or "MYC" in node_names


def test_multi_hop_pathway_doxorubicin_to_apoptosis():
    paths = search_hetero_pathways("Doxorubicin", "Pathway:Apoptosis", max_hops=4, min_score=0.5)
    assert len(paths) > 0
    top = paths[0]
    fmt = top.format_path()
    assert "Doxorubicin" in fmt
    assert "Apoptosis" in fmt
    node_names = [n.name for n in top.nodes]
    assert "TOP2A" in node_names or "TP53" in node_names or "BAX" in node_names


def test_synthetic_lethality_pathways():
    # BRCA1 PARP1
    paths = search_hetero_pathways("BRCA1", "PARP1", max_hops=2)
    assert len(paths) > 0
    assert any("synthetic_lethal_with" in e.relation for e in paths[0].edges)


def test_get_hetero_subgraph_extraction():
    sub = get_hetero_subgraph("TP53", radius=2, max_nodes=20)
    assert isinstance(sub, HeteroSubgraph)
    assert sub.node_count >= 5
    assert sub.edge_count >= 4
    # Verify gene 、pathway or
    node_names = {n.name for n in sub.nodes.values()}
    assert "CDKN1A" in node_names or "BAX" in node_names or "MDM2" in node_names

    # 
    d = sub.to_dict()
    assert d["center_id"] == "TF:TP53"
    assert d["radius"] == 2


def test_subgraph_to_pyg_export():
    sub = get_hetero_subgraph("Trametinib", radius=2)
    pyg = sub.to_pyg_data()
    assert "node_counts" in pyg
    assert "edge_indices" in pyg
    # Verify heterogeneous
    assert "Compound" in pyg["node_counts"]
    assert "Gene" in pyg["node_counts"]
    assert len(pyg["edge_indices"]) > 0


def test_custom_kg_addition():
    kg = HeterogeneousKnowledgeGraph()
    kg.add_node("Gene:GENE_A", "GENE_A", "Gene")
    kg.add_node("Gene:GENE_B", "GENE_B", "Gene")
    kg.add_edge("Gene:GENE_A", "Gene:GENE_B", "custom_regulation", weight=0.88, source_db="UserCustom")

    paths = kg.search_pathways("GENE_A", "GENE_B")
    assert len(paths) == 1
    assert paths[0].edges[0].relation == "custom_regulation"
