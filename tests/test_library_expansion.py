"""Tests for PerturbAlign library expansion APIs.

Includes:
1. Gene perturbation lookups (NCBI, ESM2, STRING, DepMap, query_gene)
2. Small molecule lookups (PubChem, ChemBERTa embedding, L1000, PRISM, JUMP, query_compound)
3. Multimodal embedding (embed_perturbation, embed_batch)
4. Single-cell AnnData featurization (featurize_anndata)
5. Embedded SQLite graph store (SQLiteGraphStore, HeterogeneousKnowledgeGraph)
"""
import pytest
import numpy as np
import perturbalign


class DummyAnnData:
    """Mock AnnData class for testing."""
    def __init__(self, obs_dict):
        import pandas as pd
        self.obs = pd.DataFrame(obs_dict)
        self.obsm = {}
        self.uns = {}

    def copy(self):
        d = DummyAnnData({})
        d.obs = self.obs.copy()
        d.obsm = dict(self.obsm)
        d.uns = dict(self.uns)
        return d


def test_ncbi_gene_lookup():
    res = perturbalign.lookup_ncbi_gene("TP53", online=False)
    assert res is not None
    assert res["symbol"] == "TP53"
    assert "entrez_id" in res
    assert "description" in res
    assert len(res["description"]) > 0
    assert "chromosome" in res


def test_esm2_embedding():
    # Without model/weights, strictly raises RuntimeError
    with pytest.raises(RuntimeError):
        perturbalign.get_esm2_embedding("TP53", dimension=320)

    # When model/callable is provided, computes embedding
    dummy_model = lambda seq: [1.0 / np.sqrt(320)] * 320
    emb = perturbalign.get_esm2_embedding("TP53", dimension=320, model=dummy_model)
    assert isinstance(emb, list)
    assert len(emb) == 320
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, rel=1e-2) == 1.0


def test_string_network():
    net = perturbalign.get_string_network(["TP53", "MDM2", "CDKN1A"], online=False)
    assert "query_genes" in net
    assert "interactions" in net
    assert len(net["interactions"]) > 0
    assert "network_size" in net
    assert net["network_size"]["nodes"] >= 3


def test_depmap_dependency():
    dep = perturbalign.get_depmap_dependency("MYC", online=False)
    assert dep is not None
    assert dep["gene_symbol"] == "MYC"
    assert "chronos_mean" in dep
    assert dep["is_common_essential"] is True
    assert len(dep["dependency_vector"]) == 32


def test_query_gene_facade():
    res = perturbalign.query_gene("KRAS", online=False)
    assert res["gene"] == "KRAS"
    assert "ncbi" in res
    assert "esm2_embedding" in res
    assert "string_network" in res
    assert "depmap_dependency" in res


@pytest.mark.network
def test_pubchem_compound_deep():
    pc = perturbalign.get_pubchem_compound("Doxorubicin", online=True)
    assert pc is not None
    assert "pubchem_cid" in pc
    assert "canonical_smiles" in pc
    assert "molecular_weight" in pc
    assert "xlogp" in pc
    assert "tpsa" in pc


def test_chemberta_embedding():
    # Without model/weights, strictly raises RuntimeError
    with pytest.raises(RuntimeError):
        perturbalign.get_chemberta_embedding("Trametinib", dimension=384)

    dummy_model = lambda smiles: [1.0 / np.sqrt(384)] * 384
    emb = perturbalign.get_chemberta_embedding("Trametinib", dimension=384, model=dummy_model)
    assert isinstance(emb, list)
    assert len(emb) == 384
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, rel=1e-2) == 1.0


def test_l1000_profile():
    l1k = perturbalign.get_l1000_profile("Doxorubicin", online=False)
    assert "top_upregulated_genes" in l1k
    assert "top_downregulated_genes" in l1k
    assert "landmark_signature_vector" in l1k


def test_prism_profile():
    prism = perturbalign.get_prism_profile("Trametinib", online=False)
    assert "mean_viability_auc" in prism
    assert "ic50_estimate_uM" in prism
    assert "viability_profile_vector" in prism


def test_jump_profile():
    jump = perturbalign.get_jump_profile("Belinostat", online=False)
    assert "top_phenotypic_changes" in jump
    assert "replicate_correlation" in jump
    assert "morphological_profile_vector" in jump


def test_query_compound_facade():
    res = perturbalign.query_compound("Doxorubicin", online=False)
    assert res["compound"] == "Doxorubicin"
    assert "pubchem" in res
    assert "chemberta_embedding" in res
    assert "l1000_profile" in res
    assert "prism_profile" in res
    assert "jump_profile" in res


def test_embed_perturbation_multimodal():
    dummy_chem = lambda s: [0.1] * 384
    dummy_esm = lambda s: [0.2] * 320

    # 1. Controls -> Baseline zero vector (allowed without model)
    v_ctrl = perturbalign.embed_perturbation("DMSO", dimension=512)
    assert len(v_ctrl) == 512
    assert all(x == 0.0 for x in v_ctrl)

    # 2. Without model weights, non-controls raise RuntimeError
    with pytest.raises(RuntimeError):
        perturbalign.embed_perturbation("Trametinib 10nM", dimension=512)
    with pytest.raises(RuntimeError):
        perturbalign.embed_perturbation("sgTP53", dimension=512)

    # 3. With explicit model weights passed
    v_sm = perturbalign.embed_perturbation("Trametinib 10nM", dimension=512, chemberta_model=dummy_chem)
    assert len(v_sm) == 512

    v_gen = perturbalign.embed_perturbation("sgTP53", dimension=512, esm2_model=dummy_esm)
    assert len(v_gen) == 512

    v_combo = perturbalign.embed_perturbation(
        "Trametinib 10nM + Pictilisib 1uM",
        dimension=512,
        chemberta_model=dummy_chem,
    )
    assert len(v_combo) == 512
    assert v_sm != v_gen


def test_embed_batch():
    # Controls work out of the box
    labels = ["DMSO", "sgCtrl"]
    matrix = perturbalign.embed_batch(labels, dimension=256, as_numpy=True)
    assert matrix.shape == (2, 256)
    assert np.all(matrix[0] == 0.0)
    assert np.all(matrix[1] == 0.0)

    # With model weights
    dummy_chem = lambda s: [0.1] * 384
    dummy_esm = lambda s: [0.2] * 320
    labels_mixed = ["Doxorubicin 10nM", "sgMYC", "DMSO"]
    matrix_mixed = perturbalign.embed_batch(
        labels_mixed,
        dimension=256,
        esm2_model=dummy_esm,
        chemberta_model=dummy_chem,
        as_numpy=True,
    )
    assert matrix_mixed.shape == (3, 256)
    assert np.all(matrix_mixed[2] == 0.0)


def test_get_profile_unified():
    p1 = perturbalign.get_profile("Doxorubicin", assay="l1000")
    assert "landmark_signature_vector" in p1
    p2 = perturbalign.get_profile("Trametinib", assay="prism")
    assert "viability_profile_vector" in p2
    p3 = perturbalign.get_profile("Belinostat", assay="jump")
    assert "morphological_profile_vector" in p3
    p4 = perturbalign.get_profile("sgTP53", assay="depmap")
    assert "dependency_vector" in p4


def test_sqlite_graph_store_and_traversal(tmp_path):
    db_file = str(tmp_path / "test_kg.sqlite")
    kg = perturbalign.HeterogeneousKnowledgeGraph()
    kg.to_sql_database(db_file)
    store = perturbalign.SQLiteGraphStore(db_file)
    nodes = store.query_nodes(limit=100)
    edges = store.query_edges(limit=100)
    assert len(nodes) > 0
    assert len(edges) > 0

    # Test SQL recursive CTE path traversal
    paths = store.find_paths_sql("Compound:Trametinib", "Gene:MAP2K1", max_hops=2)
    assert isinstance(paths, list)

    # Test SQL neighborhood
    sub = store.get_neighborhood_sql("Compound:Trametinib", radius=1)
    assert sub.node_count > 0

    # Test reload from SQL
    loaded_kg = perturbalign.HeterogeneousKnowledgeGraph(db_path=db_file)
    assert len(loaded_kg.nodes) == len(kg.nodes)
    assert len(loaded_kg.adj) > 0


def test_featurize_anndata():
    adata = DummyAnnData({
        "perturbation": ["Trametinib 10nM", "sgTP53", "DMSO", "Doxorubicin 100nM 24h", "sgMYC"]
    })
    perturbalign.featurize_anndata(
        adata,
        drug_col="perturbation",
        embed_dim=256,
        add_embeddings=True,
        add_profiles=True,
        build_ppi_network=True,
        inplace=True,
    )
    # 1. Verify obs
    assert "perturbalign_status" in adata.obs.columns
    assert "perturbalign_modality" in adata.obs.columns
    # 2. Verify obsm features
    assert "X_pert_embedding" in adata.obsm
    assert adata.obsm["X_pert_embedding"].shape == (5, 256)
    assert "X_pert_profile" in adata.obsm
    assert adata.obsm["X_pert_profile"].shape == (5, 64)
    # 3. Verify uns knowledge graph network
    assert "perturbalign_ppi_network" in adata.uns
    ppi = adata.uns["perturbalign_ppi_network"]
    assert "network_size" in ppi


def test_token_bucket_limiter():
    limiter = perturbalign.TokenBucketLimiter(rate=10.0, capacity=2.0)
    assert limiter.acquire(1.0) is True
    assert limiter.acquire(1.0) is True


def test_run_concurrent_tasks():
    items = ["TP53", "MYC", "KRAS", "EGFR"]
    results = perturbalign.run_concurrent_tasks(lambda g: f"gene_{g}", items, max_workers=2)
    assert results == ["gene_TP53", "gene_MYC", "gene_KRAS", "gene_EGFR"]


def test_reactome_pathways():
    pw = perturbalign.lookup_reactome_pathways("TP53", online=False)
    assert len(pw) > 0
    assert any("p53" in p["name"].lower() for p in pw)


def test_gene_ontology():
    terms = perturbalign.lookup_gene_ontology("TP53", online=False)
    assert len(terms) > 0
    assert any("GO:" in t["go_id"] for t in terms)


@pytest.mark.network
def test_chembl_clinical_info():
    info = perturbalign.get_chembl_clinical_info("Doxorubicin", online=True)
    assert info is not None
    assert info["max_phase"] == 4
    assert info["is_approved"] is True
