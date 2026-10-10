"""Tests for protocols expansion: JUMP-CP, MuData multimodal, sgRNA sequence lookup, OPEF export ( )."""
import pytest
from perturbalign import (
    align,
    PreprocessConfig,
    JUMPCPAdapter,
    from_jump_cp,
    annotate_mudata,
    lookup_sgrna,
    to_frictionless,
    to_jsonld,
    default_protocol_registry,
    MODALITY_GENETIC,
    MODALITY_DELIVERY,
)
from perturbalign.schema import STATUS_CONTROL, STATUS_UNMATCHED, STATUS_EXACT


def test_jump_cp_adapter_registered():
    assert "jump_cp" in default_protocol_registry.list_protocols()
    adapter = default_protocol_registry.get_adapter("jump_cp")
    assert isinstance(adapter, JUMPCPAdapter)


def test_jump_cp_records():
    # 1. Negative control
    rec_neg = {
        "Metadata_JCP2022": "JCP2022_999999",
        "Metadata_pert_type": "negcon",
        "Metadata_pert_name": "DMSO",
    }
    r_neg = from_jump_cp(rec_neg)
    assert r_neg.status == STATUS_CONTROL
    assert r_neg.is_control is True
    assert "DMSO" in r_neg.parsed_name

    # 2. Positive control
    rec_pos = {
        "Metadata_JCP2022": "JCP2022_888888",
        "Metadata_pert_type": "poscon",
    }
    r_pos = from_jump_cp(rec_pos)
    assert r_pos.status == STATUS_CONTROL
    assert r_pos.is_control is True

    # 3. CRISPR KO screen record
    rec_crispr = {
        "Metadata_JCP2022": "JCP2022_CRISPR_01",
        "Metadata_pert_type": "crispr",
        "Metadata_gene_name": "TP53",
    }
    r_cr = from_jump_cp(rec_crispr)
    assert r_cr.modality == MODALITY_GENETIC
    assert r_cr.genetic_target == "TP53"
    assert r_cr.genetic_action == "KO"

    # 4. Small molecule screen record with InChIKey & dose
    rec_cp = {
        "Metadata_JCP2022": "JCP2022_012345",
        "Metadata_InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        "Metadata_pert_type": "compound",
        "Metadata_dose": 10.0,
    }
    r_cp = from_jump_cp(rec_cp, online=False)
    assert r_cp.status == STATUS_EXACT
    assert r_cp.parsed_dose == 10.0
    assert r_cp.parsed_unit == "µM"
    assert "JCP2022_012345" in r_cp.note


class DummyModality:
    def __init__(self, obs_dict):
        try:
            import pandas as pd
            self.obs = pd.DataFrame(obs_dict)
        except ImportError:
            # Fallback stub with .columns
            class ObsStub:
                def __init__(self, d):
                    self._d = {k: list(v) for k, v in d.items()}
                    self.columns = list(d.keys())
                def __getitem__(self, col):
                    return self._d[col]
                def __setitem__(self, col, vals):
                    self._d[col] = vals
                    if col not in self.columns:
                        self.columns.append(col)
                def copy(self):
                    return ObsStub(self._d)
            self.obs = ObsStub(obs_dict)
        self.uns = {}


class DummyMuData:
    def __init__(self, mod_dict):
        self.mod = mod_dict
        self.uns = {}


def test_annotate_mudata():
    rna_mod = DummyModality({"perturbation": ["Trametinib 10nM", "DMSO"]})
    prot_mod = DummyModality({"antibody_target": ["anti-CD3", "Control_IgG"]})
    mdata = DummyMuData({"rna": rna_mod, "prot": prot_mod})

    out = annotate_mudata(
        mdata,
        modal_cols={"rna": "perturbation", "prot": "antibody_target"},
        online=False,
    )
    assert "perturbalign_status" in out.mod["rna"].obs.columns
    assert "perturbalign_status" in out.mod["prot"].obs.columns
    assert "perturbalign" in out.uns
    assert "rna" in out.uns["perturbalign"]["modalities"]
    assert "prot" in out.uns["perturbalign"]["modalities"]


def test_lookup_sgrna():
    # 1. TP53 protospacer
    tp53_seq = "CCCGGACGATATTGAACAAT"
    hit = lookup_sgrna(tp53_seq)
    assert hit is not None
    assert hit.gene_symbol == "TP53"
    assert hit.action == "KO"
    assert hit.guide_id == tp53_seq

    # 2. Non-targeting control protospacer
    nt_seq = "GACCGGAACGATCTCGCGTA"
    hit_nt = lookup_sgrna(nt_seq)
    assert hit_nt is not None
    assert hit_nt.gene_symbol == "NT_CONTROL"
    assert hit_nt.action == "control"

    # 3. Invalid sequence (wrong characters / length)
    assert lookup_sgrna("NOT_A_SEQUENCE") is None
    assert lookup_sgrna("ACTG") is None  # too short

    # 4. Custom sgRNA
    cfg = PreprocessConfig().with_extra_sgrna(GGGGCCCCAAAATTTTGGGG="MY_CUSTOM_GENE")
    hit_custom = lookup_sgrna("GGGGCCCCAAAATTTTGGGG", config=cfg)
    assert hit_custom is not None
    assert hit_custom.gene_symbol == "MY_CUSTOM_GENE"


def test_open_perturbation_exchange_format_opef():
    results = align([
        "Doxorubicin 10nM 24h",
        "sgTP53",
        "LNP-mRNA",
        "Pulse 2h Doxorubicin followed by 48h chase in fresh media",
    ], online=False)

    # 1. Frictionless Data Package export
    frictionless = to_frictionless(results)
    assert frictionless["profile"] == "tabular-data-resource"
    assert "schema" in frictionless
    assert len(frictionless["schema"]["fields"]) >= 10
    assert len(frictionless["data"]) == 4
    assert frictionless["data"][0]["input"] == "Doxorubicin 10nM 24h"
    assert frictionless["data"][0]["parsed_dose"] == 10.0
    assert frictionless["data"][3]["regime_type"] == "pulse_chase"

    # 2. JSON-LD / BioSchemas export
    jsonld = to_jsonld(results)
    assert "@context" in jsonld
    assert "bioschemas" in jsonld["@context"]
    assert "@graph" in jsonld
    graph = jsonld["@graph"]
    assert len(graph) == 4

    # Genetic node has bioschemas:Gene
    gene_node = graph[1]
    assert "bioschemas:Gene" in gene_node["@type"]
    assert gene_node["targetGene"] == "TP53"

    # Delivery node has DeliveryVehicle
    delivery_node = graph[2]
    assert "perturbalign:DeliveryVehicle" in delivery_node["@type"]
    assert delivery_node["perturbalign:cargo"] == ["mRNA"]
