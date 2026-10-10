"""Unit tests for Extended Genetic, Environmental, Combo, and Physical Dose Dimensions.

Tests:
  - Dual CRISPR (sgA+sgB, sgA/sgB, sgA_sgB).
  - Base editing (ABE, CBE) & Prime editing (PE, PE2, PE3).
  - Degron / Targeted Protein Degradation (dTAG, PROTAC, AID).
  - Dominant Negative (DN_gene, gene_DN).
  - HGVS mutations (gene_p.T790M, BRAF_p.V600E).
  - Environmental stress parameter extraction (pH, Osmolarity, Shear stress, Pressure, Radiation).
  - Combo ratio parsing in split_combo.
  - Physical dose dimension conversions & dynamic register_dose_dimension.
"""
from __future__ import annotations

import pytest
from perturbalign.dose import (
    DIMENSION_MULTIPLIER,
    DIMENSION_OSMOLARITY,
    DIMENSION_PRESSURE,
    DIMENSION_RADIATION,
    DIMENSION_SHEAR_STRESS,
    DIMENSION_TEMPERATURE,
    classify_dose_dimension,
    register_dose_dimension,
    standardize_dose,
)
from perturbalign.preprocess import (
    detect_environmental,
    detect_genetic_perturbation,
    split_combo,
)


class TestExtendedGeneticPerturbations:
    def test_dual_crispr(self):
        res1 = detect_genetic_perturbation("sgTP53+sgBRCA1")
        assert res1 == ("TP53+BRCA1", "dual_KO")

        res2 = detect_genetic_perturbation("sgEGFR/sgMYC")
        assert res2 == ("EGFR+MYC", "dual_KO")

        res3 = detect_genetic_perturbation("sgCDK4_sgCDK6")
        assert res3 == ("CDK4+CDK6", "dual_KO")

    def test_base_and_prime_editing(self):
        res_abe = detect_genetic_perturbation("ABE_TP53")
        assert res_abe == ("TP53", "editing:ABE")

        res_cbe = detect_genetic_perturbation("CBE_BRCA1")
        assert res_cbe == ("BRCA1", "editing:CBE")

        res_pe = detect_genetic_perturbation("PE3_HEK3")
        assert res_pe == ("HEK3", "editing:PE3")

    def test_degron_and_protac(self):
        res_dtag = detect_genetic_perturbation("BRD4_dTAG")
        assert res_dtag == ("BRD4", "degradation:DTAG")

        res_protac = detect_genetic_perturbation("SMARCA4_PROTAC")
        assert res_protac == ("SMARCA4", "degradation:PROTAC")

        res_aid = detect_genetic_perturbation("MYC_AID")
        assert res_aid == ("MYC", "degradation:AID")

    def test_dominant_negative(self):
        res_dn1 = detect_genetic_perturbation("RAS_DN")
        assert res_dn1 == ("RAS", "dominant_negative")

        res_dn2 = detect_genetic_perturbation("DN_TP53")
        assert res_dn2 == ("TP53", "dominant_negative")

    def test_hgvs_mutations(self):
        res_mut1 = detect_genetic_perturbation("EGFR_p.T790M")
        assert res_mut1 == ("EGFR", "mutant:p.T790M")

        res_mut2 = detect_genetic_perturbation("BRAF_p.V600E")
        assert res_mut2 == ("BRAF", "mutant:p.V600E")


class TestExtendedEnvironmentalStress:
    def test_ph_stress(self):
        res_ph = detect_environmental("pH 6.5")
        assert res_ph is not None
        assert res_ph["category"] == "ph"
        assert "6.5" in res_ph["parameter"]
        assert "acidic" in res_ph["parameter"]

        res_ph2 = detect_environmental("acidic pH (5.5)")
        assert res_ph2 is not None
        assert res_ph2["category"] == "ph"

    def test_osmolarity_stress(self):
        res = detect_environmental("Hyperosmotic stress 400 mOsm")
        assert res is not None
        assert res["category"] == "osmotic"
        assert "400 mOsm" in res["parameter"]

    def test_shear_stress(self):
        res = detect_environmental("Fluid shear stress 15 dyn/cm2")
        assert res is not None
        assert res["category"] == "mechanical"
        assert "15 dyn/cm2" in res["parameter"]

    def test_radiation_parameter(self):
        res = detect_environmental("Gamma radiation 4 Gy")
        assert res is not None
        assert res["category"] == "radiation"
        assert "4 Gy" in res["parameter"]

    def test_new_mechanical_tokens(self):
        assert detect_environmental("contact inhibition") is not None
        assert detect_environmental("high cell density") is not None


class TestComboRatioParsing:
    def test_molar_ratio_stripping(self):
        parts = split_combo("Trametinib + Dabrafenib (1:2 molar ratio)")
        assert parts == ["Trametinib", "Dabrafenib"]

    def test_mass_ratio_stripping(self):
        parts = split_combo("DrugA + DrugB 1:10 mass ratio")
        assert parts == ["DrugA", "DrugB"]


class TestExpandedDoseDimensions:
    def test_dimension_classification(self):
        assert classify_dose_dimension("Gy") == DIMENSION_RADIATION
        assert classify_dose_dimension("cGy") == DIMENSION_RADIATION
        assert classify_dose_dimension("rad") == DIMENSION_RADIATION

        assert classify_dose_dimension("dyn/cm2") == DIMENSION_SHEAR_STRESS
        assert classify_dose_dimension("Pa") == DIMENSION_SHEAR_STRESS

        assert classify_dose_dimension("mOsm") == DIMENSION_OSMOLARITY
        assert classify_dose_dimension("Osm") == DIMENSION_OSMOLARITY

        assert classify_dose_dimension("atm") == DIMENSION_PRESSURE
        assert classify_dose_dimension("bar") == DIMENSION_PRESSURE
        assert classify_dose_dimension("kPa") == DIMENSION_PRESSURE

        assert classify_dose_dimension("°C") == DIMENSION_TEMPERATURE
        assert classify_dose_dimension("x") == DIMENSION_MULTIPLIER
        assert classify_dose_dimension("-fold") == DIMENSION_MULTIPLIER

    def test_radiation_standardization(self):
        # 100 cGy = 1 Gy
        val, u = standardize_dose(100.0, "cGy")
        assert u == "Gy"
        assert abs(val - 1.0) < 1e-9

        # 200 rad = 2 Gy
        val2, u2 = standardize_dose(200.0, "rad")
        assert u2 == "Gy"
        assert abs(val2 - 2.0) < 1e-9

    def test_shear_stress_standardization(self):
        # 1 Pa = 10 dyn/cm2
        val, u = standardize_dose(2.0, "Pa")
        assert u == "dyn/cm2"
        assert abs(val - 20.0) < 1e-9

    def test_osmolarity_standardization(self):
        # 0.5 Osm = 500 mOsm
        val, u = standardize_dose(0.5, "Osm")
        assert u == "mOsm"
        assert abs(val - 500.0) < 1e-9

    def test_custom_dimension_registration(self):
        # Register sound pressure dimension (dB, base = SPL in Pa)
        register_dose_dimension(
            "sound_pressure",
            {"db": (1.0, "dB"), "kdb": (1000.0, "kdB")},
            default_unit="dB",
        )
        assert classify_dose_dimension("dB") == "sound_pressure"
        val, u = standardize_dose(0.08, "kdB")
        assert u == "dB"
        assert abs(val - 80.0) < 1e-9
