"""Unit tests for Identifier Detection & Offline IdentifierResolver.

Tests:
  - Offline pass-through of valid SMILES and InChI without requiring network or cache.
  - Prefix toleration: SMILES:, InChIKey:, CID:, CAS:, CHEMBL:, DrugBank:, Ensembl:, UniProt:.
  - Strict OpenSMILES validation preventing false positives on chemical names and gene tokens.
  - Multi-database identifier routing and candidate construction.
"""
from __future__ import annotations

import pytest
from perturbalign.align import align
from perturbalign.cache import MemoryCache
from perturbalign.identifiers import (
    detect_identifier,
    is_cas_number,
    is_chembl_id,
    is_drugbank_id,
    is_ensembl_id,
    is_inchikey,
    is_smiles,
    is_uniprot_accession,
)
from perturbalign.pipeline import AlignmentPipeline, IdentifierResolver
from perturbalign.schema import STATUS_EXACT, STATUS_UNMATCHED


class TestIdentifierSyntax:
    def test_strict_smiles_validation(self):
        # Valid SMILES
        assert is_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert is_smiles("c1ccccc1")
        assert is_smiles("CCO")
        assert is_smiles("[Na+].[Cl-]")
        assert is_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

        # Prefixed SMILES
        assert is_smiles("SMILES: CC(=O)O")
        assert is_smiles("smiles=CCO")

        # Invalid SMILES that look like drug names / codes
        assert not is_smiles("(S)-(+)-rolipram")
        assert not is_smiles("(S)-(+)-Ibuprofen")
        assert not is_smiles("(Deoxyribonucleotide)n+m")
        assert not is_smiles("(S)(+)-Allantoin")
        assert not is_smiles("S1001")
        assert not is_smiles("PCI-34051")
        assert not is_smiles("LAB-101")
        assert not is_smiles("sgTP53")
        assert not is_smiles("IL-6")

    def test_prefixed_identifiers(self):
        assert is_inchikey("InChIKey: BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        assert is_inchikey("InChIKey=BSYNRYMUTXBXSQ-UHFFFAOYSA-N")

        assert is_cas_number("CAS: 50-78-2")
        assert is_cas_number("cas: 50-78-2")
        assert not is_cas_number("50-78-9")  # wrong checksum

        assert is_chembl_id("CHEMBL25")
        assert is_chembl_id("chembl: CHEMBL25")

        assert is_drugbank_id("DB00316")
        assert is_drugbank_id("DrugBank: DB00316")

        assert is_ensembl_id("ENSG00000141510")
        assert is_ensembl_id("ensembl: ENSG00000141510")

        assert is_uniprot_accession("P04637")
        assert is_uniprot_accession("UniProt: P04637")

    def test_detect_identifier_dispatcher(self):
        assert detect_identifier("CC(=O)Oc1ccccc1C(=O)O") == ("smiles", "CC(=O)Oc1ccccc1C(=O)O")
        assert detect_identifier("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3") == ("inchi", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
        assert detect_identifier("BSYNRYMUTXBXSQ-UHFFFAOYSA-N") == ("inchikey", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        assert detect_identifier("CID: 2244") == ("pubchem_cid", 2244)
        assert detect_identifier("CAS: 50-78-2") == ("cas", "50-78-2")
        assert detect_identifier("CHEMBL25") == ("chembl_id", "CHEMBL25")
        assert detect_identifier("DB00316") == ("drugbank_id", "DB00316")
        assert detect_identifier("ENSG00000141510") == ("ensembl_id", "ENSG00000141510")
        assert detect_identifier("P04637") == ("uniprot_accession", "P04637")
        assert detect_identifier("Trametinib") is None


class TestOfflineIdentifierResolver:
    def test_offline_smiles_passthrough(self):
        # With online=False and empty cache, pure SMILES passes through to exact DrugEntity!
        cache = MemoryCache()
        res = align(["CC(=O)Oc1ccccc1C(=O)O"], online=False, cache=cache)[0]
        assert res.status == STATUS_EXACT
        assert res.entity is not None
        assert res.entity.canonical_smiles == "CC(=O)Oc1ccccc1C(=O)O"
        assert res.entity.confidence == 1.0
        assert res.entity.source == "raw_smiles"
        assert res.note == "resolved via SMILES identifier"

    def test_offline_inchi_passthrough(self):
        cache = MemoryCache()
        inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        res = align([inchi], online=False, cache=cache)[0]
        assert res.status == STATUS_EXACT
        assert res.entity is not None
        assert res.entity.canonical_name == inchi
        assert res.entity.source == "raw_inchi"

    def test_prefixed_smiles_offline(self):
        cache = MemoryCache()
        res = align(["SMILES: CCO 10uM"], online=False, cache=cache)[0]
        assert res.status == STATUS_EXACT
        assert res.entity is not None
        assert res.entity.canonical_smiles == "CCO"
        assert res.parsed_dose == 10.0
        assert res.parsed_unit == "µM"
