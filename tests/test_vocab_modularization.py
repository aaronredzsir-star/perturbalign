"""Tests for Modular Vocabularies & Protocol Harmonization (v0.8.0).

Verifies that:
1. All modular TSV vocabulary files load with proper schema and non-empty content.
2. User overrides and custom config fields take effect dynamically without code modifications.
3. LINCS and scPerturb protocol metadata adapters classify perturbations accurately.
4. AnnData adapter writes multi-modal metadata columns cleanly into .obs.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from perturbalign import (
    align,
    annotate_anndata,
    from_lincs,
    from_scperturb,
    from_prism,
    classify_protocol,
    detect_control,
    detect_genetic_perturbation,
    detect_biologic,
    detect_environmental,
    PreprocessConfig,
)
from perturbalign.preprocess import strip_salt, clean_name, _symbol_quality_ok
from perturbalign.resources import (
    load_stereo_tokens,
    load_inorganic_compounds,
    load_metal_cations,
    load_genetic_actions,
    load_genetic_control_targets,
    load_biologic_suffixes,
    load_greek_transliterations,
    load_perturbation_protocols,
    VOCABULARY_VERSION,
)
from perturbalign.schema import (
    STATUS_CONTROL,
    STATUS_UNMATCHED,
    MODALITY_CONTROL,
    MODALITY_GENETIC,
    MODALITY_BIOLOGIC,
    MODALITY_SMALL_MOLECULE,
)


class TestModularResourceLoaders:
    """Verify  has 8      and   TSV            。"""

    def test_vocabulary_version_integrity(self):
        assert VOCABULARY_VERSION.startswith("0.9.0-")
        assert len(VOCABULARY_VERSION) == 14  # 0.9.0- + 8 hex chars

    def test_load_stereo_tokens(self):
        tokens = load_stereo_tokens()
        assert isinstance(tokens, set)
        assert len(tokens) >= 15
        assert "cis" in tokens
        assert "trans" in tokens
        assert "exo" in tokens
        assert "rac" in tokens
        assert "meso" in tokens

    def test_load_inorganic_compounds(self):
        compounds = load_inorganic_compounds()
        assert isinstance(compounds, set)
        assert len(compounds) >= 15
        assert "chloride" in compounds
        assert "carbonate" in compounds
        assert "sulfate" in compounds
        assert "nitrate" in compounds
        assert "phosphate" in compounds

    def test_load_metal_cations(self):
        cations = load_metal_cations()
        assert isinstance(cations, list)
        assert len(cations) >= 15
        assert "sodium" in cations
        assert "disodium" in cations
        assert "potassium" in cations
        assert "calcium" in cations
        assert "zinc" in cations
        # Verify column
        for i in range(len(cations) - 1):
            assert len(cations[i]) >= len(cations[i + 1])

    def test_load_genetic_actions(self):
        actions = load_genetic_actions()
        assert isinstance(actions, dict)
        assert len(actions) >= 10
        assert "ko" in actions
        assert actions["ko"]["canonical_action"] == "KO"
        assert "kd" in actions
        assert actions["kd"]["canonical_action"] == "KD"
        assert "oe" in actions
        assert actions["oe"]["canonical_action"] == "OE"
        assert "crispri" in actions
        assert actions["crispri"]["canonical_action"] == "CRISPRi"

    def test_load_genetic_control_targets(self):
        targets = load_genetic_control_targets()
        assert isinstance(targets, set)
        assert len(targets) >= 20
        assert "nt" in targets
        assert "scramble" in targets
        assert "safe_targeting" in targets
        assert "luciferase" in targets
        assert "aavs1" in targets
        assert "chr2" in targets

    def test_load_biologic_suffixes(self):
        suffixes = load_biologic_suffixes()
        assert isinstance(suffixes, list)
        assert len(suffixes) >= 5
        suffix_dict = dict(suffixes)
        assert "mab" in suffix_dict
        assert suffix_dict["mab"] == "antibody"
        assert "cept" in suffix_dict
        assert suffix_dict["cept"] == "fusion_protein"
        assert "kin" in suffix_dict
        assert suffix_dict["kin"] == "cytokine"

    def test_load_greek_transliterations(self):
        greeks = load_greek_transliterations()
        assert isinstance(greeks, dict)
        assert len(greeks) >= 24
        assert greeks["α"] == "alpha"
        assert greeks["β"] == "beta"
        assert greeks["γ"] == "gamma"
        assert greeks["ω"] == "omega"
        assert greeks["Α"] == "Alpha"

    def test_load_perturbation_protocols(self):
        protocols = load_perturbation_protocols()
        assert isinstance(protocols, dict)
        assert len(protocols) >= 15
        assert "trt_cp" in protocols
        assert protocols["trt_cp"]["canonical_modality"] == "small_molecule"
        assert "trt_sh" in protocols
        assert protocols["trt_sh"]["canonical_modality"] == "genetic"
        assert "ctl_vehicle" in protocols
        assert protocols["ctl_vehicle"]["canonical_modality"] == "control"
        assert "small molecule" in protocols
        assert protocols["small molecule"]["standard_source"] == "scPerturb"


class TestCustomConfigOverrides:
    """Verify   in zero        ，     or    vocabulary     row is    。"""

    def test_custom_inorganic_protection(self):
        # Lithium Carbonate fallback （Carbonate is inorganic ）
        assert strip_salt("Lithium Carbonate") == "Lithium Carbonate"
        # carbonate， is Carbonate
        cfg = PreprocessConfig(inorganic_compounds={"chloride"})
        assert strip_salt("Lithium Carbonate", cfg) == "Carbonate"

    def test_custom_metal_cation(self):
        # "Francium Phenylbutyrate" not "Francium"（not in ）
        assert strip_salt("Francium Phenylbutyrate") == "Francium Phenylbutyrate"
        # "francium" to metal
        cfg = PreprocessConfig(metal_cations=["francium", "sodium", "potassium"])
        assert strip_salt("Francium Phenylbutyrate", cfg) == "Phenylbutyrate"

    def test_custom_biologic_suffix(self):
        # has biologic "-body" -> nanobody
        assert detect_biologic("Caplacizumab-body") is None
        cfg = PreprocessConfig(biologic_suffixes=[("body", "nanobody"), ("mab", "antibody")])
        res = detect_biologic("Caplacizumab-body", cfg)
        assert res is not None
        assert res["category"] == "nanobody"

    def test_custom_genetic_control_target(self):
        # control "LabGeneX_Control"
        assert detect_control("LabGeneX_Control") is False
        cfg = PreprocessConfig(genetic_control_targets={"labgenex_control", "nt"})
        assert detect_control("LabGeneX_Control", cfg) is True
        assert detect_genetic_perturbation("sgLabGeneX_Control", cfg) is None  # is control

    def test_custom_stereo_token_in_combo(self):
        # in ，
        cfg = PreprocessConfig(stereo_tokens={"cis", "trans", "my_custom_isomer"})
        assert _symbol_quality_ok(["DrugA", "my_custom_isomer"], "/", cfg) is False


class TestProtocolHarmonization:
    """Verify LINCS L1000   scPerturb   protocol  data adapter 。"""

    def test_classify_protocol(self):
        cp_meta = classify_protocol("trt_cp")
        assert cp_meta is not None
        assert cp_meta["canonical_modality"] == "small_molecule"

        sh_meta = classify_protocol("trt_sh")
        assert sh_meta is not None
        assert sh_meta["canonical_modality"] == "genetic"
        assert sh_meta["default_action"] == "KD"

        ctrl_meta = classify_protocol("ctl_vehicle")
        assert ctrl_meta is not None
        assert ctrl_meta["canonical_modality"] == "control"
        assert ctrl_meta["default_action"] == "vehicle"

    def test_from_lincs_compound(self):
        # trt_cp: compound ， dose and duration
        res = from_lincs(
            pert_id="BRD-K81418486",
            pert_type="trt_cp",
            pert_iname="vorinostat",
            pert_dose=10.0,
            pert_dose_unit="uM",
            pert_time=24.0,
            pert_time_unit="h",
            online=False,
        )
        assert res.input == "BRD-K81418486"
        assert res.modality == MODALITY_SMALL_MOLECULE
        assert res.parsed_dose == 10.0
        assert res.parsed_unit == "uM"
        assert res.parsed_duration == 24.0
        assert res.parsed_duration_unit == "h"

    def test_from_lincs_genetic(self):
        # trt_sh: shRNA gene
        res = from_lincs(
            pert_id="TRCN0000000001",
            pert_type="trt_sh",
            pert_iname="TP53",
            pert_time=48.0,
            pert_time_unit="h",
        )
        assert res.input == "TRCN0000000001"
        assert res.modality == MODALITY_GENETIC
        assert res.genetic_target == "TP53"
        assert res.genetic_action == "KD"
        assert res.status == STATUS_UNMATCHED
        assert res.parsed_duration == 48.0
        assert res.parsed_duration_unit == "h"

    def test_from_lincs_control(self):
        # ctl_vehicle: control
        res = from_lincs(
            pert_id="DMSO",
            pert_type="ctl_vehicle",
            pert_iname="DMSO",
            pert_time=6.0,
            pert_time_unit="h",
        )
        assert res.input == "DMSO"
        assert res.status == STATUS_CONTROL
        assert res.is_control is True
        assert res.modality == MODALITY_CONTROL
        assert res.parsed_duration == 6.0
        assert res.parsed_duration_unit == "h"

    def test_from_scperturb_row(self):
        # scPerturb
        row_genetic = {
            "perturbation": "sgTP53",
            "perturbation_type": "genetic",
        }
        res = from_scperturb(row_genetic)
        assert res.modality == MODALITY_GENETIC
        assert res.genetic_target == "TP53"
        assert res.genetic_action == "KO"

        row_ctrl = {
            "perturbation": "ctrl",
            "perturbation_type": "control",
        }
        res_ctrl = from_scperturb(row_ctrl)
        assert res_ctrl.status == STATUS_CONTROL
        assert res_ctrl.is_control is True

    def test_from_lincs_crispr_2020(self):
        # trt_xpr: LINCS 2020 CMap standard for CRISPR knockout
        res = from_lincs(
            pert_id="CRISPR_MYC",
            pert_type="trt_xpr",
            pert_iname="MYC",
            pert_time=96.0,
            pert_time_unit="h",
        )
        assert res.input == "CRISPR_MYC"
        assert res.modality == MODALITY_GENETIC
        assert res.genetic_target == "MYC"
        assert res.genetic_action == "KO"
        assert res.status == STATUS_UNMATCHED
        assert res.parsed_duration == 96.0
        assert res.parsed_duration_unit == "h"
        assert "trt_xpr" in res.note

    def test_from_prism_composite(self):
        # DepMap PRISM standard : <broad_id>::<dose>::<screen_id>
        res = from_prism("BRD-A00077618-236-07-6::2.5::HTS", online=False)
        assert res.input == "BRD-A00077618-236-07-6::2.5::HTS"
        assert res.parsed_name == "BRD-A00077618-236-07-6"
        assert res.parsed_dose == 2.5
        assert res.parsed_unit == "µM"
        assert "PRISM screen: HTS" in res.note

    def test_from_prism_control(self):
        # PRISM /control
        res = from_prism("DMSO::0::HTS")
        assert res.status == STATUS_CONTROL
        assert res.is_control is True
        assert res.modality == MODALITY_CONTROL
        assert res.parsed_name == "DMSO"
        assert res.parsed_dose == 0.0
        assert "PRISM control (HTS)" in res.note

    def test_from_prism_drug_and_screen(self):
        # PRISM panel (PR500)
        res = from_prism("Bortezomib::0.002::PR500", online=False)
        assert res.parsed_name == "Bortezomib"
        assert res.parsed_dose == 0.002
        assert res.parsed_unit == "µM"
        assert "PR500" in res.note


class TestAnnDataAdapterMultiModal:
    """Verify annotate_anndata    obs  multimodal     。"""

    def test_annotate_anndata_multimodal_obs(self):
        class StubAdata:
            def __init__(self, obs_df):
                self.obs = obs_df
                self.uns = {}

        import pandas as pd
        obs_data = pd.DataFrame({
            "perturbation": ["Trametinib 10nM 24h", "sgTP53", "DMSO 6h", "IL-6 10ng/ml 2h"]
        })
        adata = StubAdata(obs_data)
        out = annotate_anndata(adata, drug_col="perturbation", online=False, inplace=True)

        cols = out.obs.columns.tolist()
        assert "perturbalign_modality" in cols
        assert "perturbalign_parsed_duration" in cols
        assert "perturbalign_parsed_duration_unit" in cols
        assert "perturbalign_genetic_target" in cols
        assert "perturbalign_genetic_action" in cols

        # empty
        modalities = out.obs["perturbalign_modality"].tolist()
        assert modalities[1] == MODALITY_GENETIC
        assert modalities[2] == MODALITY_CONTROL
        assert modalities[3] == MODALITY_BIOLOGIC

        durations = out.obs["perturbalign_parsed_duration"].tolist()
        assert durations[0] == 24.0
        assert durations[2] == 6.0
        assert durations[3] == 2.0

        targets = out.obs["perturbalign_genetic_target"].tolist()
        assert targets[1] == "TP53"
        actions = out.obs["perturbalign_genetic_action"].tolist()
        assert actions[1] == "KO"


class TestExpandedDomainVocabularies:
    """Verify         /      vocabulary      。"""

    def test_biologic_adc_and_rna_suffixes(self):
        # ADC suffix (vedotin, deruxtecan)
        adc1 = detect_biologic("Trastuzumab deruxtecan")
        assert adc1 is not None
        assert adc1["category"] == "adc"
        adc2 = detect_biologic("Enfortumab vedotin")
        assert adc2 is not None
        assert adc2["category"] == "adc"

        # RNAi / siRNA suffix (siran)
        rnai = detect_biologic("Patisiran")
        assert rnai is not None
        assert rnai["category"] == "rnai"

        # ASO suffix (nersen)
        aso = detect_biologic("Nusinersen")
        assert aso is not None
        assert aso["category"] == "aso"

        # suffix (tide)
        pep = detect_biologic("Semaglutide")
        assert pep is not None
        assert pep["category"] == "peptide"

    def test_expanded_biologics_interleukins_and_cytokines(self):
        # 、
        il17 = detect_biologic("IL-17A")
        assert il17 is not None
        assert il17["category"] == "cytokine"

        il33 = detect_biologic("IL-33")
        assert il33 is not None
        assert il33["category"] == "cytokine"

        cxcl = detect_biologic("CXCL12")
        assert cxcl is not None
        assert cxcl["category"] == "chemokine"

        vegf = detect_biologic("VEGF-A")
        assert vegf is not None
        assert vegf["category"] == "growth_factor"

        pembro = detect_biologic("Pembrolizumab")
        assert pembro is not None
        assert pembro["category"] == "antibody"

    def test_expanded_formulation_codes(self):
        # 、
        assert clean_name("Paclitaxel liposomal") == "Paclitaxel"
        assert clean_name("Doxorubicin inj") == "Doxorubicin"
        assert clean_name("Omeprazole DR") == "Omeprazole"
        assert clean_name("Metformin IR") == "Metformin"
        assert clean_name("Paroxetine CR") == "Paroxetine"

    def test_expanded_vehicles_and_transfection_controls(self):
        # 
        assert detect_control("Captisol") is True
        assert detect_control("Solutol HS15") is True
        assert detect_control("Cremophor EL") is True
        assert detect_control("PEG400") is True
        # 
        assert detect_control("RPMI-1640") is True
        assert detect_control("Opti-MEM") is True
        assert detect_control("Lipofectamine") is True

    def test_expanded_environmental_stress(self):
        # starvation 、
        env1 = detect_environmental("glucose starvation")
        assert env1 is not None
        assert env1["category"] == "nutrient"

        env2 = detect_environmental("shear stress")
        assert env2 is not None
        assert env2["category"] == "mechanical"

        env3 = detect_environmental("Erastin")
        assert env3 is not None
        assert env3["category"] == "cellular_stress"
        assert "ferroptosis" in env3["canonical_name"].lower()

    def test_atp_protection_and_salt_lookbehind(self):
        # (ATP) not is Adenosine
        assert clean_name("Adenosine Triphosphate") == "Adenosine Triphosphate"
        assert strip_salt("Adenosine Triphosphate") == "Adenosine Triphosphate"
        # salt form suffix
        assert strip_salt("Doxorubicin HCl") == "Doxorubicin"
        assert strip_salt("Sildenafil Citrate") == "Sildenafil"
