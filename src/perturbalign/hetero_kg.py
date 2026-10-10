"""Heterogeneous Pathway & Knowledge Graph Search Engine (HeteroKG).

Integrates multi-source biological networks from PrimeKG, Hetionet, BioGRID, DoRothEA, Reactome, KEGG, STITCH, etc.
Supports:
1. Heterogeneous node types: Compound, Gene, TF, Pathway, Disease, CellType.
2. Heterogeneous causal edges: targets, inhibits, activates, interacts_with, synthetic_lethal_with, regulates, participates_in, marker_for, indicated_for.
3. Multi-hop heterogeneous meta-path search.
4. Perturbation neighborhood subgraph extraction.
5. Export to NetworkX / PyTorch Geometric HeteroData formats for graph deep learning.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
import gzip
import heapq
import json
import os
from typing import Any, Iterable, Optional, Union


# --------------------------------------------------------------------------
# 1. Heterogeneous graph node and edge data contracts
# --------------------------------------------------------------------------
@dataclass
class HeteroNode:
    """Heterogeneous knowledge graph node."""
    id: str                  # Unique identifier (e.g. 'Compound:Doxorubicin', 'Gene:TP53')
    name: str                # Normalized name (e.g. 'Doxorubicin', 'TP53')
    node_type: str           # 'Compound', 'Gene', 'TF', 'Pathway', 'Disease', 'CellType'
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type,
            "attributes": self.attributes,
        }


@dataclass
class HeteroEdge:
    """Heterogeneous knowledge graph edge."""
    source: str              # Source node ID
    target: str              # Target node ID
    relation: str            # Relation predicate (e.g. 'targets', 'regulates', 'synthetic_lethal_with')
    weight: float = 1.0      # Confidence weight (0.0 ~ 1.0)
    source_db: str = "curated"  # Source database (PrimeKG, BioGRID, DoRothEA, STITCH, etc.)
    effect_sign: int = 0     # Causal effect sign: +1 (activation/upregulation), -1 (inhibition/downregulation), 0 (binding/neutral)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "source_db": self.source_db,
            "effect_sign": self.effect_sign,
            "attributes": self.attributes,
        }


@dataclass
class HeteroPath:
    """Heterogeneous multi-hop pathway between source and target entities."""
    source_id: str
    target_id: str
    hops: int
    nodes: list[HeteroNode]
    edges: list[HeteroEdge]
    confidence_score: float

    @property
    def overall_effect_sign(self) -> int:
        """Compute cascaded cumulative causal effect sign (+1 activation/enhancement, -1 inhibition/repression, 0 neutral/unknown)."""
        sign = 1
        has_signed = False
        for e in self.edges:
            if e.effect_sign != 0:
                sign *= e.effect_sign
                has_signed = True
        return sign if has_signed else 0

    def format_path(self) -> str:
        """Format as an intuitive causal chain string representation."""
        if not self.nodes:
            return ""
        chain = [f"[{self.nodes[0].node_type}: {self.nodes[0].name}]"]
        for i, edge in enumerate(self.edges):
            nxt = self.nodes[i + 1]
            chain.append(f" --({edge.relation}|{edge.source_db})--> [{nxt.node_type}: {nxt.name}]")
        return "".join(chain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "hops": self.hops,
            "overall_effect_sign": self.overall_effect_sign,
            "path_string": self.format_path(),
            "confidence_score": round(self.confidence_score, 4),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


@dataclass
class HeteroSubgraph:
    """Heterogeneous neighborhood knowledge subgraph surrounding a perturbation."""
    center_id: str
    radius: int
    nodes: dict[str, HeteroNode] = field(default_factory=dict)
    edges: list[HeteroEdge] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_id": self.center_id,
            "radius": self.radius,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_pyg_data(self) -> dict[str, Any]:
        """Export as structured tensor index dictionary compatible with PyTorch Geometric HeteroData."""
        # Collect node index mapping
        type_to_nodes: dict[str, list[str]] = collections.defaultdict(list)
        node_to_idx: dict[str, int] = {}
        for nid, node in self.nodes.items():
            type_to_nodes[node.node_type].append(nid)
            node_to_idx[nid] = len(type_to_nodes[node.node_type]) - 1

        # Collect edge indices (src_type, rel, dst_type)
        edge_index_dict: dict[tuple[str, str, str], list[list[int]]] = collections.defaultdict(lambda: [[], []])
        for e in self.edges:
            src_node = self.nodes.get(e.source)
            dst_node = self.nodes.get(e.target)
            if src_node and dst_node:
                src_type = src_node.node_type
                dst_type = dst_node.node_type
                key = (src_type, e.relation, dst_type)
                edge_index_dict[key][0].append(node_to_idx[e.source])
                edge_index_dict[key][1].append(node_to_idx[e.target])

        return {
            "node_counts": {k: len(v) for k, v in type_to_nodes.items()},
            "edge_indices": {f"{k[0]}__{k[1]}__{k[2]}": v for k, v in edge_index_dict.items()},
        }


# --------------------------------------------------------------------------
# 2. Global high-confidence heterogeneous knowledge base (HeteroKG Knowledge Base)
# --------------------------------------------------------------------------
class HeterogeneousKnowledgeGraph:
    """Unified heterogeneous graph integrating PrimeKG / Hetionet / DoRothEA / BioGRID / STITCH."""

    def __init__(self, load_curated: bool = True, db_path: Optional[str] = None) -> None:
        self.nodes: dict[str, HeteroNode] = {}
        # adjacency: src_id -> list of (dst_id, HeteroEdge)
        self.adj: dict[str, list[tuple[str, HeteroEdge]]] = collections.defaultdict(list)
        # reverse adjacency for bidirectional search
        self.rev_adj: dict[str, list[tuple[str, HeteroEdge]]] = collections.defaultdict(list)
        # High-frequency causal pathway search query cache (LRU / Memoization)
        self._path_cache: dict[tuple[str, str, int, float, int], list[HeteroPath]] = {}
        if db_path is not None and os.path.exists(db_path):
            self.from_sql_database(db_path)
        else:
            self._build_seed_graph()
            if load_curated:
                self.load_curated_databases()
            if db_path is not None:
                self.to_sql_database(db_path)

    def to_sql_database(self, db_path: str) -> dict[str, int]:
        """Export and persist the in-memory graph to an embedded SQLite graph database."""
        from .graph_store import SQLiteGraphStore
        with SQLiteGraphStore(db_path) as store:
            return store.save_graph(self)

    def from_sql_database(self, db_path: str) -> dict[str, int]:
        """Load nodes and edges from an embedded SQLite graph database."""
        from .graph_store import SQLiteGraphStore
        with SQLiteGraphStore(db_path) as store:
            return store.load_graph(self)

    def find_paths_sql(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        min_weight: float = 0.0,
        limit: int = 10,
        db_path: str = ":memory:",
    ) -> list[HeteroPath]:
        """Find multi-hop causal paths using SQLite Recursive CTE graph traversal."""
        from .graph_store import SQLiteGraphStore
        with SQLiteGraphStore(db_path) as store:
            store.save_graph(self)
            return store.find_paths_sql(source_id, target_id, max_hops=max_hops, min_weight=min_weight, limit=limit)

    def get_subgraph_sql(
        self,
        center_id: str,
        radius: int = 1,
        db_path: str = ":memory:",
    ) -> HeteroSubgraph:
        """Extract neighborhood subgraph using SQLite Recursive CTE."""
        from .graph_store import SQLiteGraphStore
        with SQLiteGraphStore(db_path) as store:
            store.save_graph(self)
            return store.get_neighborhood_sql(center_id, radius=radius)

    def load_curated_databases(self) -> dict[str, Any]:
        """Ingest all registered knowledge base providers via KGRegistry."""
        from .kg_registry import default_kg_registry
        import perturbalign.kg_providers  # noqa: F401
        counts = default_kg_registry.load_all(self)
        total_edges = sum(len(el) for el in self.adj.values())
        return {
            "total_nodes": len(self.nodes),
            "total_edges": total_edges,
            "provider_counts": counts,
            "registered_providers": list(default_kg_registry.list_providers().keys()),
        }

    def list_available_providers(self) -> dict[str, dict[str, Any]]:
        """List all currently registered knowledge base providers and their metadata."""
        from .kg_registry import default_kg_registry
        import perturbalign.kg_providers  # noqa: F401
        return default_kg_registry.list_providers()

    def load_dataset(self, name: str, **kwargs) -> int:
        """Dynamically load a specific knowledge base dataset from KGRegistry by name."""
        from .kg_registry import default_kg_registry
        import perturbalign.kg_providers  # noqa: F401
        return default_kg_registry.load_provider(name, self, **kwargs)

    def add_node(self, id: str, name: str, node_type: str, **attrs) -> HeteroNode:
        if id not in self.nodes:
            node = HeteroNode(id=id, name=name, node_type=node_type, attributes=attrs)
            self.nodes[id] = node
            self._path_cache.clear()
        return self.nodes[id]

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        source_db: str = "curated",
        effect_sign: Optional[int] = None,
        bidirectional: bool = False,
        **attrs,
    ) -> None:
        self._path_cache.clear()
        if effect_sign is None:
            rel_lower = relation.lower()
            if any(k in rel_lower for k in ["inhib", "repress", "antagon", "block", "suppress", "degrad", "downregulat"]):
                effect_sign = -1
            elif any(k in rel_lower for k in ["activat", "induc", "agon", "stimulat", "upregulat", "phospho", "stabiliz"]):
                effect_sign = 1
            else:
                effect_sign = 0
        edge = HeteroEdge(source=source, target=target, relation=relation, weight=weight, source_db=source_db, effect_sign=effect_sign, attributes=attrs)
        self.adj[source].append((target, edge))
        self.rev_adj[target].append((source, edge))
        if bidirectional:
            rev_edge = HeteroEdge(source=target, target=source, relation=relation, weight=weight, source_db=source_db, effect_sign=effect_sign, attributes=attrs)
            self.adj[target].append((source, rev_edge))
            self.rev_adj[source].append((target, rev_edge))

    def _build_seed_graph(self) -> None:
        """Load seed knowledge graph covering oncogenic signaling pathways, regulons, targets, and synthetic lethality."""
        # 1. Core compounds
        self.add_node("Compound:Trametinib", "Trametinib", "Compound", smiles="CC1=C2C(=O)N...", class_="MEK_inhibitor")
        self.add_node("Compound:Doxorubicin", "Doxorubicin", "Compound", smiles="CC1C(C...", class_="Anthracycline")
        self.add_node("Compound:Imatinib", "Imatinib", "Compound", class_="BCR_ABL_inhibitor")
        self.add_node("Compound:Belinostat", "Belinostat", "Compound", class_="HDAC_inhibitor")
        self.add_node("Compound:Pictilisib", "Pictilisib", "Compound", class_="PI3K_inhibitor")

        # 2. Core kinases and target proteins
        for g in ["MAP2K1", "MAP2K2", "MAPK1", "BRAF", "KRAS", "EGFR", "TOP2A", "TOP2B", "ABL1", "KIT", "HDAC1", "PIK3CA"]:
            self.add_node(f"Gene:{g}", g, "Gene")

        # 3. Core transcription factors
        for tf in ["TP53", "MYC", "GATA3", "STAT3", "RELA", "FOXA1"]:
            self.add_node(f"TF:{tf}", tf, "TF")

        # 4. Effector and downstream marker genes
        for eg in ["CDKN1A", "MDM2", "BAX", "ATM", "CHEK2", "BRCA1", "PARP1", "BCL2", "MCL1", "CASP3"]:
            self.add_node(f"Gene:{eg}", eg, "Gene")

        # 5. Biological pathways (Reactome / KEGG)
        self.add_node("Pathway:MAPK_Cascade", "MAPK/ERK Signaling Pathway", "Pathway", source_db="KEGG")
        self.add_node("Pathway:DNA_Damage_Checkpoint", "p53-Dependent DNA Damage Checkpoint", "Pathway", source_db="Reactome")
        self.add_node("Pathway:Apoptosis", "Programmed Cell Death / Apoptosis", "Pathway", source_db="Reactome")
        self.add_node("Pathway:Cell_Cycle_Arrest", "G1/S DNA Damage Cell Cycle Arrest", "Pathway", source_db="Reactome")

        # 6. Disease indications and cell types (PrimeKG & CellMarker)
        self.add_node("Disease:Melanoma", "Cutaneous Melanoma", "Disease", source_db="PrimeKG")
        self.add_node("Disease:Leukemia", "Acute Myeloid Leukemia", "Disease", source_db="PrimeKG")
        self.add_node("CellType:T_Cell", "CD8+ Cytotoxic T Cell", "CellType", source_db="CellMarker")
        self.add_node("CellType:Melanocyte", "Malignant Melanocyte", "CellType", source_db="CellMarker")

        # ---------------- Connect edge relations ----------------
        # A. Compounds -> targets (targets / inhibits) [STITCH & DrugBank]
        self.add_edge("Compound:Trametinib", "Gene:MAP2K1", "inhibits", weight=0.99, source_db="BindingDB/STITCH")
        self.add_edge("Compound:Trametinib", "Gene:MAP2K2", "inhibits", weight=0.98, source_db="BindingDB/STITCH")
        self.add_edge("Compound:Doxorubicin", "Gene:TOP2A", "intercalates_and_inhibits", weight=0.99, source_db="DrugBank")
        self.add_edge("Compound:Doxorubicin", "Gene:TOP2B", "inhibits", weight=0.95, source_db="DrugBank")
        self.add_edge("Compound:Imatinib", "Gene:ABL1", "inhibits", weight=0.99, source_db="DrugBank")
        self.add_edge("Compound:Belinostat", "Gene:HDAC1", "inhibits", weight=0.98, source_db="DrugBank")
        self.add_edge("Compound:Pictilisib", "Gene:PIK3CA", "inhibits", weight=0.97, source_db="BindingDB")

        # B. Gene and protein interactions (interacts_with / phosphorylates) [STRING & BioGRID]
        self.add_edge("Gene:MAP2K1", "Gene:MAPK1", "phosphorylates", weight=0.99, source_db="BioGRID", bidirectional=True)
        self.add_edge("Gene:BRAF", "Gene:MAP2K1", "activates", weight=0.99, source_db="STRING")
        self.add_edge("Gene:KRAS", "Gene:BRAF", "activates", weight=0.98, source_db="STRING")
        self.add_edge("Gene:TOP2A", "Gene:ATM", "activates_damage_response", weight=0.92, source_db="BioGRID")
        self.add_edge("Gene:TOP2A", "TF:TP53", "triggers_checkpoint_response", weight=0.95, source_db="Reactome")
        self.add_edge("Gene:ATM", "Gene:CHEK2", "phosphorylates", weight=0.99, source_db="BioGRID", bidirectional=True)
        self.add_edge("Gene:CHEK2", "TF:TP53", "activates", weight=0.98, source_db="BioGRID")
        self.add_edge("Gene:MDM2", "TF:TP53", "inhibits_ubiquitinates", weight=0.99, source_db="STRING", bidirectional=True)

        # C. Synthetic lethal relations (synthetic_lethal_with) [SynLethDB & BioGRID]
        self.add_edge("Gene:BRCA1", "Gene:PARP1", "synthetic_lethal_with", weight=0.99, source_db="SynLethDB", bidirectional=True)
        self.add_edge("TF:TP53", "Gene:CHEK2", "synthetic_lethal_with", weight=0.92, source_db="SynLethDB", bidirectional=True)
        self.add_edge("Gene:KRAS", "Gene:BCL2", "synthetic_lethal_with", weight=0.88, source_db="SynLethDB", bidirectional=True)

        # D. Transcription factor -> target gene regulons (regulates) [DoRothEA Level A & JASPAR]
        self.add_edge("TF:TP53", "Gene:CDKN1A", "transcriptional_activation", weight=0.99, source_db="DoRothEA_A")
        self.add_edge("TF:TP53", "Gene:BAX", "transcriptional_activation", weight=0.99, source_db="DoRothEA_A")
        self.add_edge("TF:TP53", "Gene:MDM2", "transcriptional_activation", weight=0.98, source_db="DoRothEA_A")
        self.add_edge("TF:MYC", "Gene:CDKN1A", "transcriptional_repression", weight=0.95, source_db="DoRothEA_A")
        self.add_edge("Gene:MAPK1", "TF:MYC", "stabilizes", weight=0.92, source_db="BioGRID")

        # E. Genes -> biological pathways (participates_in) [Reactome & KEGG]
        self.add_edge("Gene:MAPK1", "Pathway:MAPK_Cascade", "participates_in", weight=1.0, source_db="KEGG")
        self.add_edge("Gene:MAP2K1", "Pathway:MAPK_Cascade", "participates_in", weight=1.0, source_db="KEGG")
        self.add_edge("TF:TP53", "Pathway:DNA_Damage_Checkpoint", "participates_in", weight=1.0, source_db="Reactome")
        self.add_edge("TF:TP53", "Pathway:Apoptosis", "induces_pathway", weight=0.99, source_db="Reactome")
        self.add_edge("Gene:CDKN1A", "Pathway:Cell_Cycle_Arrest", "participates_in", weight=1.0, source_db="Reactome")
        self.add_edge("Gene:BAX", "Pathway:Apoptosis", "participates_in", weight=1.0, source_db="Reactome")
        self.add_edge("Gene:CASP3", "Pathway:Apoptosis", "participates_in", weight=1.0, source_db="Reactome")

        # F. Compounds & genes -> diseases & cell types [PrimeKG & CellMarker]
        self.add_edge("Compound:Trametinib", "Disease:Melanoma", "indicated_for", weight=0.99, source_db="PrimeKG")
        self.add_edge("Compound:Doxorubicin", "Disease:Leukemia", "indicated_for", weight=0.98, source_db="PrimeKG")
        self.add_edge("Gene:BRAF", "Disease:Melanoma", "mutated_in", weight=0.95, source_db="OpenTargets")
        self.add_edge("Gene:CDKN1A", "CellType:Melanocyte", "marker_for_senescence", weight=0.88, source_db="CellMarker")

    def resolve_node_id(self, query: str) -> Optional[str]:
        """Resolve arbitrary name or prefixed ID to exact heterogeneous graph node ID."""
        q = query.strip()
        # 1. Exact ID match
        if q in self.nodes:
            return q
        # 2. Case-insensitive prefix/name match
        upper = q.upper()
        for nid, node in self.nodes.items():
            if node.name.upper() == upper or nid.upper() == upper:
                return nid
        # 3. Try main entity type prefixes
        for prefix in ["Compound", "Gene", "TF", "Pathway", "Disease", "CellType"]:
            cand = f"{prefix}:{q}"
            for nid in self.nodes:
                if nid.upper() == cand.upper():
                    return nid
        return None

    def get_node(self, query: str) -> Optional[HeteroNode]:
        """Retrieve heterogeneous graph node object by ID or name."""
        nid = self.resolve_node_id(query)
        if nid:
            return self.nodes.get(nid)
        return self.nodes.get(query)


    def search_pathways(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
        min_score: float = 0.4,
        max_paths: int = 10,
    ) -> list[HeteroPath]:
        """Search all causal pathways between two entities in the heterogeneous knowledge graph (Multi-hop Pathway Finding).

        Parameters:
          source: Source entity name (e.g., 'Trametinib' or 'TP53')
          target: Target entity name (e.g., 'CDKN1A' or 'Apoptosis')
          max_hops: Maximum number of hops (default 3)
          min_score: Minimum cumulative confidence score
          max_paths: Maximum number of paths to return

        Returns:
          List of HeteroPath objects sorted by confidence score descending.
        """
        src_id = self.resolve_node_id(source)
        tgt_id = self.resolve_node_id(target)
        if not src_id or not tgt_id or src_id == tgt_id:
            return []

        cache_key = (src_id, tgt_id, max_hops, min_score, max_paths)
        if cache_key in self._path_cache:
            return list(self._path_cache[cache_key])

        found_paths: list[HeteroPath] = []
        # DFS / Backtracking search for all simple paths
        # State: (curr_node_id, visited_nodes, visited_set, current_edges, current_score)
        stack: list[tuple[str, list[str], set[str], list[HeteroEdge], float]] = [(src_id, [src_id], {src_id}, [], 1.0)]

        while stack:
            curr, visited, v_set, edges, score = stack.pop()
            if curr == tgt_id:
                if score >= min_score:
                    node_objs = [self.nodes[nid] for nid in visited]
                    path_obj = HeteroPath(
                        source_id=src_id,
                        target_id=tgt_id,
                        hops=len(edges),
                        nodes=node_objs,
                        edges=list(edges),
                        confidence_score=score,
                    )
                    found_paths.append(path_obj)
                    if len(found_paths) >= max_paths * 2:
                        break
                continue

            if len(edges) >= max_hops:
                continue

            for next_node, edge_obj in self.adj.get(curr, []):
                if next_node not in v_set:
                    new_score = score * edge_obj.weight
                    if new_score >= min_score * 0.8:  # Topological pruning
                        stack.append((next_node, visited + [next_node], v_set | {next_node}, edges + [edge_obj], new_score))

        # Sort by confidence score
        found_paths.sort(key=lambda p: (p.confidence_score, -p.hops), reverse=True)
        res = found_paths[:max_paths]
        self._path_cache[cache_key] = res
        return res

    def get_subgraph(
        self,
        center_query: str,
        radius: int = 2,
        max_nodes: int = 50,
    ) -> HeteroSubgraph:
        """Extract heterogeneous neighborhood knowledge subgraph around a perturbation (Neighborhood Subgraph).

        Parameters:
          center_query: Perturbation entity name (e.g., 'Trametinib', 'TP53')
          radius: Neighborhood radius (default 2)
          max_nodes: Maximum number of subgraph nodes
        """
        center_id = self.resolve_node_id(center_query)
        if not center_id:
            # Dynamically create temporary entity
            center_id = f"Entity:{center_query}"
            self.add_node(center_id, center_query, "Entity")

        subgraph_nodes: dict[str, HeteroNode] = {center_id: self.nodes[center_id]}
        subgraph_edges: list[HeteroEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()

        # BFS expansion
        queue = collections.deque([(center_id, 0)])
        visited_depth = {center_id: 0}

        while queue and len(subgraph_nodes) < max_nodes:
            curr, depth = queue.popleft()
            if depth >= radius:
                continue

            for nxt, edge in self.adj.get(curr, []):
                if nxt not in subgraph_nodes and len(subgraph_nodes) < max_nodes:
                    subgraph_nodes[nxt] = self.nodes[nxt]
                    visited_depth[nxt] = depth + 1
                    queue.append((nxt, depth + 1))

                edge_key = (edge.source, edge.target, edge.relation)
                if edge_key not in seen_edges and nxt in subgraph_nodes:
                    seen_edges.add(edge_key)
                    subgraph_edges.append(edge)

        return HeteroSubgraph(
            center_id=center_id,
            radius=radius,
            nodes=subgraph_nodes,
            edges=subgraph_edges,
        )

    def meta_path_search(
        self,
        source: str,
        target: Optional[str] = None,
        meta_path: Optional[list[str]] = None,
        max_paths: int = 10,
        min_score: float = 0.2,
    ) -> list[HeteroPath]:
        """Execute constrained path search following specified heterogeneous node type sequence pattern (Meta-path)."""
        if not meta_path or len(meta_path) < 2:
            return []
        src_id = self.resolve_node_id(source)
        if not src_id or src_id not in self.nodes:
            return []
        tgt_id = self.resolve_node_id(target) if target else None

        found: list[HeteroPath] = []
        target_depth = len(meta_path) - 1
        stack = [(src_id, 0, [src_id], [], 1.0)]

        while stack:
            curr, step, visited, edges, score = stack.pop()
            if step == target_depth:
                if tgt_id is None or curr == tgt_id:
                    if score >= min_score:
                        node_objs = [self.nodes[nid] for nid in visited]
                        found.append(HeteroPath(
                            source_id=src_id,
                            target_id=curr,
                            hops=len(edges),
                            nodes=node_objs,
                            edges=list(edges),
                            confidence_score=score,
                        ))
                        if len(found) >= max_paths * 2:
                            break
                continue

            next_type = meta_path[step + 1].upper()
            for nxt, edge in self.adj.get(curr, []):
                if nxt not in visited:
                    nxt_node = self.nodes.get(nxt)
                    if nxt_node and nxt_node.node_type.upper() == next_type:
                        new_score = score * edge.weight
                        if new_score >= min_score * 0.7:
                            stack.append((nxt, step + 1, visited + [nxt], edges + [edge], new_score))

        found.sort(key=lambda p: p.confidence_score, reverse=True)
        return found[:max_paths]

    def save_to_json(self, filepath: str) -> None:
        """Export heterogeneous knowledge graph to local JSON / Gzip file."""
        seen_edges = set()
        dedup_edges = []
        for edge_list in self.adj.values():
            for _, e in edge_list:
                key = (e.source, e.target, e.relation)
                if key not in seen_edges:
                    seen_edges.add(key)
                    dedup_edges.append(e.to_dict())
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": dedup_edges,
        }
        opener = gzip.open if filepath.endswith(".gz") else open
        mode = "wt" if filepath.endswith(".gz") else "w"
        with opener(filepath, mode, encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_json(self, filepath: str) -> int:
        """Load and merge heterogeneous knowledge graph from JSON / Gzip file; returns number of added nodes and edges."""
        if not os.path.exists(filepath):
            return 0
        opener = gzip.open if filepath.endswith(".gz") else open
        mode = "rt" if filepath.endswith(".gz") else "r"
        with opener(filepath, mode, encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for nd in data.get("nodes", []):
            self.add_node(nd["id"], nd["name"], nd["node_type"], **nd.get("attributes", {}))
            count += 1
        for ed in data.get("edges", []):
            self.add_edge(
                ed["source"],
                ed["target"],
                ed["relation"],
                weight=ed.get("weight", 1.0),
                source_db=ed.get("source_db", "curated"),
                effect_sign=ed.get("effect_sign", 0),
                **ed.get("attributes", {}),
            )
            count += 1
        return count

    def predict_pathway_impact(
        self,
        perturbation: str,
        pathway: Optional[str] = None,
        max_hops: int = 3,
    ) -> dict[str, Any]:
        """Automatically infer activation/inhibition causal impact and evidence chain on downstream biological pathways from perturbation entity."""
        src_id = self.resolve_node_id(perturbation)
        if not src_id or src_id not in self.nodes:
            return {"perturbation": perturbation, "pathways": {}, "top_impacted": []}

        target_pathway_ids = []
        if pathway:
            tgt_id = self.resolve_node_id(pathway)
            if tgt_id:
                target_pathway_ids.append(tgt_id)
        else:
            target_pathway_ids = [nid for nid, node in self.nodes.items() if node.node_type == "Pathway"]

        results = {}
        for pw_id in target_pathway_ids:
            pw_node = self.nodes.get(pw_id)
            if not pw_node:
                continue
            paths = self.search_pathways(src_id, pw_id, max_hops=max_hops, min_score=0.2, max_paths=10)
            if not paths:
                continue

            signed_weights = [p.overall_effect_sign * p.confidence_score for p in paths]
            total_weight = sum(p.confidence_score for p in paths)
            raw_score = sum(signed_weights) / total_weight if total_weight > 0 else 0.0
            score = round(raw_score, 4)
            direction = "activated" if score > 0.05 else ("inhibited" if score < -0.05 else "neutral")

            results[pw_node.name] = {
                "pathway_id": pw_id,
                "pathway_name": pw_node.name,
                "impact_score": score,
                "direction": direction,
                "path_count": len(paths),
                "evidence_paths": [p.to_dict() for p in paths[:3]],
            }

        sorted_impacts = sorted(results.values(), key=lambda x: abs(x["impact_score"]), reverse=True)
        return {
            "perturbation": self.nodes[src_id].name,
            "perturbation_id": src_id,
            "pathways": results,
            "top_impacted": sorted_impacts,
        }


# Global singleton heterogeneous knowledge graph
global_hetero_kg = HeterogeneousKnowledgeGraph()


# --------------------------------------------------------------------------
# 3. High-level public facade APIs
# --------------------------------------------------------------------------
def search_hetero_pathways(
    source: str,
    target: str,
    max_hops: int = 3,
    min_score: float = 0.4,
    max_paths: int = 10,
    kg: Optional[HeterogeneousKnowledgeGraph] = None,
) -> list[HeteroPath]:
    """Search multi-hop causal regulatory pathways between two entities in the heterogeneous knowledge graph."""
    graph = kg or global_hetero_kg
    return graph.search_pathways(source=source, target=target, max_hops=max_hops, min_score=min_score, max_paths=max_paths)


def meta_path_search(
    source: str,
    target: Optional[str] = None,
    meta_path: Optional[list[str]] = None,
    max_paths: int = 10,
    min_score: float = 0.2,
    kg: Optional[HeterogeneousKnowledgeGraph] = None,
) -> list[HeteroPath]:
    """Execute constrained path search following specified heterogeneous node type sequence pattern (Meta-path)."""
    graph = kg or global_hetero_kg
    return graph.meta_path_search(source=source, target=target, meta_path=meta_path, max_paths=max_paths, min_score=min_score)


def predict_pathway_impact(
    perturbation: str,
    pathway: Optional[str] = None,
    max_hops: int = 3,
    kg: Optional[HeterogeneousKnowledgeGraph] = None,
) -> dict[str, Any]:
    """Automatically infer activation/inhibition causal impact and evidence chain on downstream biological pathways from perturbation entity."""
    graph = kg or global_hetero_kg
    return graph.predict_pathway_impact(perturbation=perturbation, pathway=pathway, max_hops=max_hops)


def get_hetero_subgraph(
    perturbation_or_name: str,
    radius: int = 2,
    max_nodes: int = 50,
    kg: Optional[HeterogeneousKnowledgeGraph] = None,
) -> HeteroSubgraph:
    """Extract heterogeneous neighborhood knowledge subgraph around a perturbation (including upstream/downstream targets, regulons, and pathways)."""
    graph = kg or global_hetero_kg
    return graph.get_subgraph(center_query=perturbation_or_name, radius=radius, max_nodes=max_nodes)

