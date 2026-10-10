"""             Test  。

    :
1.          （   uva etc.         968          ）。
2.   small molecule stress      （Erastin, Tunicamycin etc.   small molecule  row      ， in   stress      ）。
3.   dose   dose combination     （    dose          or   ）。
4.       WHO INN suffix    （-rsen, -leucel，   -cide，    small molecule ）。
5.          EC           （<=>         not    is   combo）。
"""

import pytest
from perturbalign.align import align
from perturbalign.schema import (
    MODALITY_COMBO,
    MODALITY_CONTROL,
    MODALITY_ENVIRONMENTAL,
    MODALITY_SMALL_MOLECULE,
)
from perturbalign.preprocess import (
    detect_biologic,
    detect_environmental,
    detect_genetic_perturbation,
    split_combo,
)


class TestEnvironmentalWordBoundary:
    """Test        match ，      。"""

    def test_prevent_uva_substring_collision(self):
        # (Sodium pyruvate) "uva"，not is UVA
        assert detect_environmental("Sodium pyruvate") is None
        assert detect_environmental("Sodium pyruvate 1mM") is None

        # (Fluvastatin) "uva"，not is UVA
        assert detect_environmental("Fluvastatin") is None
        assert detect_environmental("Fluvastatin 10 uM") is None

        # (Adjuvant) "uva"
        assert detect_environmental("Adjuvant") is None

        # UVA
        env_uva1 = detect_environmental("UVA irradiation")
        assert env_uva1 is not None
        assert env_uva1["category"] == "radiation"

        env_uva2 = detect_environmental("UVA 20 J/m2")
        assert env_uva2 is not None
        assert env_uva2["category"] == "radiation"

        env_uva3 = detect_environmental("UVA")
        assert env_uva3 is not None
        assert env_uva3["category"] == "radiation"

    def test_align_pipeline_not_hijacked_by_uva(self):
        r = align(["Sodium pyruvate 1mM"], online=False)[0]
        assert r.modality != MODALITY_ENVIRONMENTAL
        assert r.parsed_name in ("pyruvate", "Sodium pyruvate")
        assert r.parsed_dose == 1.0


class TestStressInducerProtection:
    """Test small molecule    /      stress   。"""

    def test_erastin_pipeline_routing(self):
        # single agent and dose input ： small molecule ， parsed_name dose
        r_drug = align(["Erastin 10 uM"], online=False)[0]
        assert r_drug.modality == MODALITY_SMALL_MOLECULE
        assert r_drug.parsed_name == "Erastin"
        assert r_drug.parsed_dose == 10.0
        assert r_drug.parsed_unit in ("\xb5M", "uM")

        # stress ： environmental
        r_stress = align(["Erastin-induced ferroptosis"], online=False)[0]
        assert r_stress.modality == MODALITY_ENVIRONMENTAL
        assert "ferroptosis" in r_stress.parsed_name.lower()

    def test_tunicamycin_pipeline_routing(self):
        r_drug = align(["Tunicamycin 5 ug/mL"], online=False)[0]
        assert r_drug.modality == MODALITY_SMALL_MOLECULE
        assert r_drug.parsed_name == "Tunicamycin"
        assert r_drug.parsed_dose == 5.0

        r_stress = align(["Tunicamycin ER stress"], online=False)[0]
        assert r_stress.modality == MODALITY_ENVIRONMENTAL


class TestComboDoseAttribution:
    """Test combination       dose     。"""

    def test_leading_dose_attribution(self):
        # 10 uM Trametinib，Pictilisib no dose
        res = align(["10 uM Trametinib + Pictilisib"], online=False)[0]
        assert res.modality == MODALITY_COMBO
        assert len(res.parts) == 2
        p1, p2 = res.parts[0], res.parts[1]
        assert p1.parsed_name == "Trametinib"
        assert p1.parsed_dose == 10.0
        assert p1.parsed_unit in ("\xb5M", "uM")
        assert p2.parsed_name == "Pictilisib"
        assert p2.parsed_dose is None

    def test_both_doses_preserved(self):
        # 10 uM + 20 nM
        res = align(["10 uM Trametinib + 20 nM Pictilisib"], online=False)[0]
        assert res.modality == MODALITY_COMBO
        assert len(res.parts) == 2
        p1, p2 = res.parts[0], res.parts[1]
        assert p1.parsed_name == "Trametinib"
        assert p1.parsed_dose == 10.0
        assert p1.parsed_unit in ("\xb5M", "uM")
        assert p2.parsed_name == "Pictilisib"
        assert p2.parsed_dose == 20.0
        assert p2.parsed_unit == "nM"

    def test_trailing_doses_preserved(self):
        # dose
        res = align(["Trametinib 10 uM + Pictilisib 20 nM"], online=False)[0]
        assert res.modality == MODALITY_COMBO
        assert len(res.parts) == 2
        p1, p2 = res.parts[0], res.parts[1]
        assert p1.parsed_name == "Trametinib"
        assert p1.parsed_dose == 10.0
        assert p2.parsed_name == "Pictilisib"
        assert p2.parsed_dose == 20.0


class TestBiologicSuffixesRefinement:
    """Test     INN suffix   ，  small molecule   。"""

    def test_small_molecules_not_misidentified(self):
        # -sen match column Basen and Aquatensen
        assert detect_biologic("Basen") is None
        assert detect_biologic("Aquatensen") is None

        # -cide match / small molecule
        assert detect_biologic("Pesticide") is None
        assert detect_biologic("Spermicide") is None

        # -cel match 7 small molecule Doricel
        assert detect_biologic("Doricel") is None

    def test_valid_biologics_still_detected(self):
        # ASO specification suffix -rsen -nersen
        b_ete = detect_biologic("Eteplirsen")
        assert b_ete is not None
        assert b_ete["category"] == "aso"

        b_nus = detect_biologic("Nusinersen")
        assert b_nus is not None
        assert b_nus["category"] == "aso"

        # -leucel
        b_car = detect_biologic("Tisagenlecleucel")
        assert b_car is not None
        assert b_car["category"] == "cell_therapy"


class TestReactionAndEnzymeProtection:
    """Test                。"""

    def test_reaction_arrows_not_split_into_regime(self):
        from perturbalign.preprocess import split_sequential_regime
        reaction1 = "UDP-glucose + NAD+ <=> UDP-glucuronate + NADH + H+"
        assert split_sequential_regime(reaction1) is None

        reaction2 = "L-Cysteate + H2O <=> Pyruvate + HSO3- + Ammonia"
        assert split_sequential_regime(reaction2) is None

    def test_ec_enzyme_naming_not_split(self):
        enzyme1 = "methanol:NAD+ oxidoreductase"
        assert split_combo(enzyme1) is None

        enzyme2 = "ATP:protein phosphotransferase"
        assert split_combo(enzyme2) is None

    def test_normal_combos_still_split(self):
        combo = "Trametinib + Pictilisib"
        parts = split_combo(combo)
        assert parts == ["Trametinib", "Pictilisib"]


class TestGeneticPrefixGuard:
    """Test   prefix    ：    small molecule investigational code 、    and all     gene 。"""

    def test_chemical_compounds_and_genes_not_misidentified(self):
        # ( + )
        assert detect_genetic_perturbation("Shikonin") is None
        assert detect_genetic_perturbation("Shogaol") is None

        # investigational code ( 、 )
        assert detect_genetic_perturbation("SH-714") is None
        assert detect_genetic_perturbation("SG-2000") is None
        assert detect_genetic_perturbation("SGC-0946") is None
        assert detect_genetic_perturbation("SGI-1776") is None

        # all gene ( sg/sh )
        assert detect_genetic_perturbation("SHANK3") is None
        assert detect_genetic_perturbation("SH2B3") is None
        assert detect_genetic_perturbation("SGCB") is None

    def test_genuine_genetic_perturbations_detected(self):
        # sg/sh
        res_sg = detect_genetic_perturbation("sgTP53")
        assert res_sg == ("TP53", "KO")

        res_sh = detect_genetic_perturbation("shEGFR")
        assert res_sh == ("EGFR", "KD")

        # 
        res_sep_sg = detect_genetic_perturbation("sg-TP53")
        assert res_sep_sg == ("TP53", "KO")

        res_sep_sh = detect_genetic_perturbation("shRNA-KRAS")
        assert res_sep_sh == ("KRAS", "KD")

        # CRISPRi / CRISPRa
        res_ci = detect_genetic_perturbation("CRISPRi_MYC")
        assert res_ci == ("MYC", "CRISPRi")

        res_ca = detect_genetic_perturbation("CRISPRa_IL6")
        assert res_ca == ("IL6", "CRISPRa")


class TestEndogenousGeneControlProtection:
    """Test   gene    control   ： gene   not   is all      control 。"""

    def test_bare_genes_not_treated_as_control(self):
        from perturbalign.preprocess import detect_control
        assert not detect_control("CCR5")
        assert not detect_control("HPRT1")
        assert not detect_control("ROSA26")

        # in align parse not is control
        r_ccr5 = align(["CCR5"], online=False)[0]
        assert not r_ccr5.is_control
        assert r_ccr5.status != "control"
        assert r_ccr5.modality != MODALITY_CONTROL

        r_hprt1 = align(["HPRT1"], online=False)[0]
        assert not r_hprt1.is_control
        assert r_hprt1.status != "control"
        assert r_hprt1.modality != MODALITY_CONTROL

    def test_genetic_controls_with_prefixes_correctly_detected(self):
        from perturbalign.preprocess import detect_control
        # has sg/sh prefix or _sg suffix all control
        assert detect_control("sgCCR5")
        assert detect_control("sgHPRT1")
        assert detect_control("sgRosa26")
        assert detect_control("sgNT")
        assert detect_control("shControl")
        assert detect_control("aavs1_sg")

        r_sg_ccr5 = align(["sgCCR5"], online=False)[0]
        assert r_sg_ccr5.is_control
        assert r_sg_ccr5.modality == MODALITY_CONTROL


class TestCovalentPhosphateProtection:
    """Test desalt    ：                   ，    is   。"""

    def test_covalent_phosphate_metabolites_protected(self):
        from perturbalign.preprocess import strip_salt
        # /
        assert strip_salt("Adenosine diphosphate") == "Adenosine diphosphate"
        assert strip_salt("Pyridoxal phosphate") == "Pyridoxal phosphate"
        assert strip_salt("Creatine phosphate") == "Creatine phosphate"
        assert strip_salt("Carbamoyl phosphate") == "Carbamoyl phosphate"
        assert strip_salt("Thiamine pyrophosphate") == "Thiamine pyrophosphate"
        assert strip_salt("Isopentenyl pyrophosphate") == "Isopentenyl pyrophosphate"

    def test_organic_salts_regularly_stripped(self):
        from perturbalign.preprocess import strip_salt
        # has salt form
        assert strip_salt("Morphine Sulfate") == "Morphine"
        assert strip_salt("Atropine Sulfate") == "Atropine"
        assert strip_salt("Doxorubicin HCl") == "Doxorubicin"
        assert strip_salt("Sodium Phenylbutyrate") == "Phenylbutyrate"


class TestMolarDivisionAndMicroVolumeUnits:
    """Test     unit (umol/L) and      concentration (ng/ul)     parse 。"""

    def test_molar_division_units(self):
        from perturbalign.dose import classify_dose_dimension, standardize_dose
        r = align(["Trametinib 10 umol/L"], online=False)[0]
        assert r.parsed_name == "Trametinib"
        assert r.parsed_dose == 10.0
        assert r.parsed_unit in ("µM", "uM")
        assert classify_dose_dimension("umol/L") == "molar"
        assert standardize_dose(10.0, "umol/L") == (10.0, "µM")

        r_nmol = align(["Pictilisib 50 nmol/L"], online=False)[0]
        assert r_nmol.parsed_name == "Pictilisib"
        assert r_nmol.parsed_dose == 50.0
        assert r_nmol.parsed_unit == "nM"
        assert standardize_dose(50.0, "nmol/L") == (0.05, "µM")

    def test_micro_volume_concentration_units(self):
        from perturbalign.dose import classify_dose_dimension, standardize_dose
        r = align(["Total-RNA 100 ng/ul"], online=False)[0]
        assert r.parsed_name == "Total-RNA"
        assert r.parsed_dose == 100.0
        assert r.parsed_unit in ("ng/µl", "ng/ul")
        assert classify_dose_dimension("ng/ul") == "mass_concentration"
        # 100 ng/ul = 100 ug/ml
        val, unit = standardize_dose(100.0, "ng/ul")
        assert unit == "µg/ml"
        assert abs(val - 100.0) < 1e-5


class TestCanonicalIngestionAndPunctuation:
    """Test Phase 0 specification    ：all    、zero    、not  row empty  。"""

    def test_fullwidth_parentheses_and_alias(self):
        # all ：extract dose 10 µM， Trametinib
        r = align(["Trametinib 10uM（GSK1120212）"], online=False)[0]
        assert r.parsed_dose == 10.0
        assert r.parsed_unit in ("µM", "uM")
        assert "Trametinib" in r.parsed_name

    def test_fullwidth_plus_combination(self):
        # all is combination
        r = align(["Trametinib＋Pictilisib"], online=False)[0]
        assert r.modality == MODALITY_COMBO
        assert len(r.parts) == 2
        assert r.parts[0].parsed_name == "Trametinib"
        assert r.parts[1].parsed_name == "Pictilisib"

    def test_zero_width_and_non_breaking_spaces(self):
        # zero not row empty
        raw = "Trametinib\u200b\u00a010\u00a0uM"
        r = align([raw], online=False)[0]
        assert r.parsed_name == "Trametinib"
        assert r.parsed_dose == 10.0
        assert r.parsed_unit in ("µM", "uM")


class TestMolarAndViralTiterDimensions:
    """Test       M         gene      。"""

    def test_scientific_notation_molar(self):
        from perturbalign.dose import classify_dose_dimension, standardize_dose
        r = align(["Trametinib 1e-6 M"], online=False)[0]
        assert r.parsed_name == "Trametinib"
        assert r.parsed_dose == 1e-6
        assert r.parsed_unit == "M"
        assert classify_dose_dimension("M") == "molar"
        std_val, std_u = standardize_dose(1e-6, "M")
        assert abs(std_val - 1.0) < 1e-9
        assert std_u == "µM"

    def test_viral_vector_titer_dimension(self):
        from perturbalign.dose import classify_dose_dimension, standardize_dose
        r = align(["AAV-GFP 1e11 vg/ml"], online=False)[0]
        assert r.parsed_dose == 1e11
        assert r.parsed_unit == "vg/ml"
        assert classify_dose_dimension("vg/ml") == "viral_titer"
        std_val, std_u = standardize_dose(1e11, "vg/ml")
        assert std_val == 1e11
        assert std_u == "vg/ml"

        r_lenti = align(["Lenti-shRNA 2.5e8 TU/ml"], online=False)[0]
        assert r_lenti.parsed_dose == 2.5e8
        assert r_lenti.parsed_unit == "TU/ml"
        assert classify_dose_dimension("TU/ml") == "viral_titer"


class TestCompoundEpigenomeEffectors:
    """Test   epigenetic    （dCas9-KRAB-MeCP2 / CRISPRoff / CRISPRon）not    is Combo。"""

    def test_dcas9_krab_mecp2_not_split(self):
        from perturbalign.schema import MODALITY_GENETIC
        r = align(["dCas9-KRAB-MeCP2_MYC"], online=False)[0]
        assert r.modality == MODALITY_GENETIC
        assert r.genetic_target == "MYC"
        assert "dCas9-KRAB-MeCP2" in r.genetic_action

    def test_crisproff_and_crispron(self):
        from perturbalign.schema import MODALITY_GENETIC
        r_off = align(["CRISPRoff_MYC"], online=False)[0]
        assert r_off.modality == MODALITY_GENETIC
        assert r_off.genetic_target == "MYC"

        r_on = align(["CRISPRon_IL1B"], online=False)[0]
        assert r_on.modality == MODALITY_GENETIC
        assert r_on.genetic_target == "IL1B"

