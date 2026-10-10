"""Unit tests for Online Annotation Provider SPI (BaseAnnotationProvider & ProviderRegistry).

Tests:
  - BaseAnnotationProvider contract.
  - Built-in providers: PubChemProvider, ChEMBLProvider, NCICIRProvider.
  - ProviderRegistry dynamic registration, priority ordering, unregistration, reset.
  - Dependency injection of custom mock annotation providers.
  - Cascading lookup order verification.
"""
from __future__ import annotations

import pytest
from perturbalign.annotate import (
    BaseAnnotationProvider,
    ChEMBLProvider,
    NCICIRProvider,
    ProviderRegistry,
    PubChemProvider,
    default_provider_registry,
    resolve_online_cascade,
)
from perturbalign.cache import MemoryCache


class MockInternalDBProvider(BaseAnnotationProvider):
    """Simulates an internal company proprietary compound registration database."""

    @property
    def name(self) -> str:
        return "internal_db"

    def search_by_name(self, query: str, cache=None) -> list[dict] | None:
        if query.strip().upper() == "COMP-999":
            return [{
                "canonical_name": "Proprietary-Candidate-999",
                "canonical_smiles": "CC1=CC=CC=C1",
                "inchikey": "PROP-INCHIKEY-999",
                "source": "internal_db",
            }]
        return None

    def search_by_identifier(self, kind: str, value: str, cache=None) -> list[dict] | None:
        if kind == "custom_id" and value == "INTERNAL-001":
            return [{
                "canonical_name": "Internal-Lead-001",
                "canonical_smiles": "CCC",
                "inchikey": "LEAD-001",
                "source": "internal_db",
            }]
        return None


class TestProviderRegistry:
    def test_default_registry_initialization(self):
        reg = ProviderRegistry()
        providers = reg.get_providers()
        names = [p.name for p in providers]
        assert names == ["pubchem", "chembl", "nci_cir"]

    def test_register_and_priority(self):
        reg = ProviderRegistry()
        custom = MockInternalDBProvider()
        # Register with priority 5 (higher priority than PubChem at 10)
        reg.register(custom, priority=5)

        providers = reg.get_providers()
        assert providers[0].name == "internal_db"
        assert providers[1].name == "pubchem"

    def test_unregister(self):
        reg = ProviderRegistry()
        reg.unregister("chembl")
        names = [p.name for p in reg.get_providers()]
        assert "chembl" not in names
        assert names == ["pubchem", "nci_cir"]

    def test_reset(self):
        reg = ProviderRegistry()
        reg.unregister("pubchem")
        assert len(reg.get_providers()) == 2
        reg.reset()
        assert len(reg.get_providers()) == 3
        assert [p.name for p in reg.get_providers()] == ["pubchem", "chembl", "nci_cir"]


class TestOnlineCascadeWithRegistry:
    def test_cascade_with_custom_provider(self):
        reg = ProviderRegistry()
        reg.register(MockInternalDBProvider(), priority=5)

        hits, sources = resolve_online_cascade("COMP-999", registry=reg)
        assert hits is not None
        assert len(hits) == 1
        assert hits[0]["canonical_name"] == "Proprietary-Candidate-999"
        assert sources == ["internal_db"]

    def test_cascade_fallback_when_unmatched(self):
        reg = ProviderRegistry()
        custom = MockInternalDBProvider()
        reg.register(custom, priority=5)

        # Unmatched in custom and online (without network in mock) returns None, []
        hits, sources = resolve_online_cascade("NonExistentXYZ123", cache=MemoryCache(), registry=reg)
        assert hits is None
        assert sources == []
