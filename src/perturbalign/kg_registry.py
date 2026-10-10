"""Pluggable heterogeneous knowledge graph abstraction layer and registry.

Design Goals:
1. Decouple data sources from graph engine: Abstract biomedical databases into independent Providers.
2. Zero-overhead extensibility: Adding a new data source (e.g. OmniPath, SIGNOR, DisGeNET)
   only requires implementing BaseKGProvider and decorating with @register_kg_provider.
3. Lazy loading and metadata governance: Uniformly manage provider names, versions, node/edge types,
   and confidence rules.
"""
from __future__ import annotations

import abc
from typing import Any, Callable, Optional, Type, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .hetero_kg import HeterogeneousKnowledgeGraph


class BaseKGProvider(abc.ABC):
    """Knowledge graph data provider base class."""
    name: str = "base"
    version: str = "1.0.0"
    description: str = ""
    source_url: str = ""
    node_types: tuple[str, ...] = ()
    edge_types: tuple[str, ...] = ()

    @abc.abstractmethod
    def load_into(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        """Inject entities and causal edges from this data source into the target KG. Returns record count."""
        raise NotImplementedError

    def get_metadata(self) -> dict[str, Any]:
        """Return standardized provider metadata dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "source_url": self.source_url,
            "node_types": list(self.node_types),
            "edge_types": list(self.edge_types),
        }


class KGRegistry:
    """Global knowledge graph provider registry."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseKGProvider] = {}

    def register(self, name_or_provider: Union[str, BaseKGProvider, Type[BaseKGProvider]], maybe_cls: Optional[Type[BaseKGProvider]] = None) -> Any:
        """Register a data provider (supports instance, class, or decorator syntax)."""
        if isinstance(name_or_provider, str) and maybe_cls is not None:
            instance = maybe_cls() if isinstance(maybe_cls, type) else maybe_cls
            self._providers[name_or_provider.lower()] = instance
            return maybe_cls
        if isinstance(name_or_provider, str) and maybe_cls is None:
            # Decorator syntax
            def decorator(cls: Type[BaseKGProvider]) -> Type[BaseKGProvider]:
                self._providers[name_or_provider.lower()] = cls()
                return cls
            return decorator
        if isinstance(name_or_provider, BaseKGProvider):
            self._providers[name_or_provider.name.lower()] = name_or_provider
            return name_or_provider
        if isinstance(name_or_provider, type) and issubclass(name_or_provider, BaseKGProvider):
            inst = name_or_provider()
            self._providers[inst.name.lower()] = inst
            return name_or_provider
        raise ValueError(f"Invalid provider registration: {name_or_provider}")

    def get(self, name: str) -> Optional[BaseKGProvider]:
        return self._providers.get(name.lower())

    def list_providers(self) -> dict[str, dict[str, Any]]:
        return {name: p.get_metadata() for name, p in self._providers.items()}

    def load_provider(self, name: str, kg: HeterogeneousKnowledgeGraph, **kwargs) -> int:
        p = self.get(name)
        if not p:
            raise KeyError(f"KG Provider '{name}' not found. Available: {list(self._providers.keys())}")
        return p.load_into(kg, **kwargs)

    def load_all(self, kg: HeterogeneousKnowledgeGraph, **kwargs) -> dict[str, int]:
        results = {}
        for name, p in self._providers.items():
            results[name] = p.load_into(kg, **kwargs)
        return results


# Global singleton registry and shortcut decorator
default_kg_registry = KGRegistry()
register_kg_provider = default_kg_registry.register
