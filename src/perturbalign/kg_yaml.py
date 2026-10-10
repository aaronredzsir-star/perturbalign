"""Declarative zero-code knowledge graph ingestion engine.

Allows research groups or third-party users to register private or community knowledge graphs
via declarative YAML / JSON / dict configurations without writing Python code or subclassing
BaseKGProvider, achieving zero-overhead pluggable graph extensions.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional, Union, TYPE_CHECKING

from .kg_registry import BaseKGProvider, KGRegistry, default_kg_registry
from .exceptions import RegistryError

if TYPE_CHECKING:
    from .hetero_kg import HeterogeneousKnowledgeGraph


def parse_declarative_config(source: Union[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Parse declarative configuration (.yaml, .yml, .json file paths or native dicts)."""
    if isinstance(source, Mapping):
        return dict(source)

    if not isinstance(source, str):
        raise RegistryError("ConfigParser", f"Unsupported declarative config input type: {type(source)}")

    if not os.path.exists(source):
        # Attempt to parse as raw JSON text
        try:
            return json.loads(source)
        except Exception:
            raise RegistryError(source, f"Configuration file does not exist: {source}")

    content = ""
    with open(source, "r", encoding="utf-8") as f:
        content = f.read()

    # Try YAML parse (if PyYAML is installed)
    try:
        import yaml
        parsed = yaml.safe_load(content)
        if isinstance(parsed, dict):
            return parsed
    except ImportError:
        pass

    # Fallback to JSON parse
    try:
        return json.loads(content)
    except Exception as e:
        raise RegistryError(source, f"Failed to parse config file (install pyyaml or provide valid JSON): {str(e)}")


def register_declarative_kg(
    config: Union[str, Mapping[str, Any]],
    registry: Optional[KGRegistry] = None,
    kg: Optional[HeterogeneousKnowledgeGraph] = None,
) -> Any:
    """Create and register a knowledge graph Provider via declarative configuration.

    Parameters
    ----------
    config : str or dict
        Configuration file path (YAML/JSON) or configuration dictionary.
    registry : KGRegistry, optional
        Target registry; defaults to global default_kg_registry.
    kg : HeterogeneousKnowledgeGraph, optional
        If provided, directly loads nodes and edges into this graph and returns (node_count, edge_count).

    Returns
    -------
    str or tuple[int, int]
        If kg is None, returns registered provider_id;
        If kg is provided, returns (node_count, edge_count) tuple.
    """
    cfg = parse_declarative_config(config)
    target_registry = registry or default_kg_registry

    provider_id = cfg.get("id") or cfg.get("name", "custom_kg").strip().lower()
    name = cfg.get("name", provider_id)
    version = str(cfg.get("version", "1.0.0"))
    description = cfg.get("description", "Declarative Community Knowledge Graph")
    source_url = cfg.get("source_url", "")
    node_types = tuple(cfg.get("node_types", ("Gene", "Compound")))
    edge_types = tuple(cfg.get("edge_types", ("interacts_with",)))
    nodes_data = cfg.get("nodes", [])
    edges_data = cfg.get("edges", [])
    tsv_file = cfg.get("file")

    _name = name
    _version = version
    _description = description
    _source_url = source_url
    _node_types = node_types
    _edge_types = edge_types

    class DeclarativeProvider(BaseKGProvider):
        name = _name
        version = _version
        description = _description
        source_url = _source_url
        node_types = _node_types
        edge_types = _edge_types

        def load_into(self, target_kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
            # 0. Inject static nodes list (if present)
            for n in nodes_data:
                nid = str(n.get("id", "")).strip()
                nname = str(n.get("name", nid)).strip()
                ntype = str(n.get("node_type", "Gene")).strip()
                nprops = n.get("properties", {})
                if nid:
                    target_kg.add_node(nid, nname, ntype, **nprops)

            edge_count = 0
            # 1. Inject static edges list
            for e in edges_data:
                src = str(e.get("source", "")).strip()
                tgt = str(e.get("target", "")).strip()
                rel = str(e.get("relation", "interacts_with")).strip()
                if not src or not tgt:
                    continue
                w = float(e.get("weight", 1.0))
                sign = int(e.get("effect_sign", 0))

                s_name = src.split(":")[-1]
                s_type = src.split(":")[0] if ":" in src else "Gene"
                t_name = tgt.split(":")[-1]
                t_type = tgt.split(":")[0] if ":" in tgt else "Gene"

                target_kg.add_node(src, s_name, s_type)
                target_kg.add_node(tgt, t_name, t_type)
                target_kg.add_edge(src, tgt, rel, weight=w, source_db=name, effect_sign=sign)
                edge_count += 1

            # 2. Ingest external TSV/CSV tabular edge file if specified
            if tsv_file and os.path.exists(tsv_file):
                from .kg_loaders import ingest_tabular_edges
                c_file = ingest_tabular_edges(
                    target_kg,
                    filepath=tsv_file,
                    source_col=cfg.get("source_col", "source"),
                    target_col=cfg.get("target_col", "target"),
                    relation_col=cfg.get("relation_col", "relation"),
                    weight_col=cfg.get("weight_col", "weight"),
                    sign_col=cfg.get("sign_col", "effect_sign"),
                    source_db=name,
                    delimiter=cfg.get("delimiter", "\t"),
                )
                edge_count += c_file

            return edge_count

    # Register into registry
    target_registry.register(provider_id, DeclarativeProvider)

    if kg is not None:
        p = DeclarativeProvider()
        e_cnt = p.load_into(kg)
        n_cnt = len(nodes_data)
        return n_cnt, e_cnt

    return provider_id
