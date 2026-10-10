"""Tests for Multi-Dimensional Dose Classification and Standardization (P1).

Verify       、        、        /       and pConc   。
"""
from __future__ import annotations

import pytest
from perturbalign.dose import (
    classify_dose_dimension,
    standardize_dose,
    to_pconc,
    DIMENSION_MOLAR,
    DIMENSION_MASS_CONCENTRATION,
    DIMENSION_BODY_WEIGHT,
    DIMENSION_ACTIVITY,
    DIMENSION_PERCENTAGE,
    DIMENSION_INFECTION_RATIO,
    DIMENSION_AMOUNT,
    DIMENSION_VOLUME,
)


class TestDoseDimensionClassification:
    """Verify            no  。"""

    def test_molar_dimensions(self):
        assert classify_dose_dimension("nM") == DIMENSION_MOLAR
        assert classify_dose_dimension("uM") == DIMENSION_MOLAR
        assert classify_dose_dimension("µM") == DIMENSION_MOLAR
        assert classify_dose_dimension("mM") == DIMENSION_MOLAR
        assert classify_dose_dimension("pM") == DIMENSION_MOLAR
        assert classify_dose_dimension("fM") == DIMENSION_MOLAR
        assert classify_dose_dimension("M") == DIMENSION_MOLAR

    def test_mass_concentration_dimensions(self):
        assert classify_dose_dimension("mg/ml") == DIMENSION_MASS_CONCENTRATION
        assert classify_dose_dimension("mg/mL") == DIMENSION_MASS_CONCENTRATION
        assert classify_dose_dimension("ug/ml") == DIMENSION_MASS_CONCENTRATION
        assert classify_dose_dimension("µg/ml") == DIMENSION_MASS_CONCENTRATION
        assert classify_dose_dimension("ng/ml") == DIMENSION_MASS_CONCENTRATION
        assert classify_dose_dimension("pg/ml") == DIMENSION_MASS_CONCENTRATION
        assert classify_dose_dimension("g/L") == DIMENSION_MASS_CONCENTRATION

    def test_body_weight_dimensions(self):
        assert classify_dose_dimension("mg/kg") == DIMENSION_BODY_WEIGHT
        assert classify_dose_dimension("ug/kg") == DIMENSION_BODY_WEIGHT
        assert classify_dose_dimension("µg/kg") == DIMENSION_BODY_WEIGHT
        assert classify_dose_dimension("g/kg") == DIMENSION_BODY_WEIGHT

    def test_activity_dimensions(self):
        assert classify_dose_dimension("U/ml") == DIMENSION_ACTIVITY
        assert classify_dose_dimension("mU/ml") == DIMENSION_ACTIVITY
        assert classify_dose_dimension("IU/ml") == DIMENSION_ACTIVITY
        assert classify_dose_dimension("kU/ml") == DIMENSION_ACTIVITY

    def test_percentage_and_infection_ratio(self):
        assert classify_dose_dimension("%") == DIMENSION_PERCENTAGE
        assert classify_dose_dimension("% v/v") == DIMENSION_PERCENTAGE
        assert classify_dose_dimension("% w/v") == DIMENSION_PERCENTAGE
        assert classify_dose_dimension("MOI") == DIMENSION_INFECTION_RATIO

    def test_amount_and_volume(self):
        assert classify_dose_dimension("nmol") == DIMENSION_AMOUNT
        assert classify_dose_dimension("µmol") == DIMENSION_AMOUNT
        assert classify_dose_dimension("mL") == DIMENSION_VOLUME
        assert classify_dose_dimension("µL") == DIMENSION_VOLUME

    def test_unknown_dimension(self):
        assert classify_dose_dimension("arbitrary_units") is None
        assert classify_dose_dimension("") is None
        assert classify_dose_dimension(None) is None


class TestStandardizeDose:
    """Verify             。"""

    def test_molar_standardization(self):
        # to µM
        val, unit = standardize_dose(100.0, "nM")
        assert unit == "µM"
        assert abs(val - 0.1) < 1e-9

        val2, unit2 = standardize_dose(0.005, "mM")
        assert unit2 == "µM"
        assert abs(val2 - 5.0) < 1e-9

        # unit nM
        val3, unit3 = standardize_dose(2.5, "µM", target_unit="nM")
        assert unit3 == "nM"
        assert abs(val3 - 2500.0) < 1e-9

    def test_mass_conc_standardization(self):
        # to µg/ml
        val, unit = standardize_dose(500.0, "ng/ml")
        assert unit == "µg/ml"
        assert abs(val - 0.5) < 1e-9

        val2, unit2 = standardize_dose(2.0, "mg/ml")
        assert unit2 == "µg/ml"
        assert abs(val2 - 2000.0) < 1e-9

    def test_body_weight_standardization(self):
        # to mg/kg
        val, unit = standardize_dose(2500.0, "ug/kg")
        assert unit == "mg/kg"
        assert abs(val - 2.5) < 1e-9

    def test_activity_standardization(self):
        val, unit = standardize_dose(5000.0, "mU/ml")
        assert unit == "U/ml"
        assert abs(val - 5.0) < 1e-9

    def test_cross_dimension_with_mw(self):
        # small molecule MW = 180.16 g/mol
        # 1.8016 µg/ml = 1.8016 mg/L = (1.8016 / 180.16) mmol/L = 0.01 mM = 10 µM
        val, unit = standardize_dose(1.8016, "µg/ml", target_unit="µM", mw=180.16)
        assert unit == "µM"
        assert abs(val - 10.0) < 1e-6

        # : 10 µM -> µg/ml
        val_mass, unit_mass = standardize_dose(10.0, "µM", target_unit="µg/ml", mw=180.16)
        assert unit_mass == "µg/ml"
        assert abs(val_mass - 1.8016) < 1e-6

    def test_cross_dimension_error_without_mw(self):
        # Converting mass concentration to molar without MW must raise error
        with pytest.raises(ValueError, match="must provide positive molecular weight"):
            standardize_dose(10.0, "ng/ml", target_unit="µM")

    def test_incompatible_dimension_error(self):
        # Converting body weight dose (mg/kg) to molar (µM) must raise incompatible dimension error
        with pytest.raises(ValueError, match="Cannot convert dose between incompatible physical dimensions"):
            standardize_dose(5.0, "mg/kg", target_unit="µM")


class TestToPConc:
    """Verify        concentration   and        。"""

    def test_pconc_valid_molar(self):
        # 10 µM = 10^-5 M -> 5.0
        assert to_pconc(10.0, "µM") == 5.0
        # 100 nM = 10^-7 M -> 7.0
        assert to_pconc(100.0, "nM") == 7.0
        # 1 mM = 10^-3 M -> 3.0
        assert to_pconc(1.0, "mM") == 3.0

    def test_pconc_safety_guard_for_non_molar(self):
        # concentration not pConc
        assert to_pconc(10.0, "ng/ml") is None
        # not pConc
        assert to_pconc(5.0, "mg/kg") is None
        # not pConc
        assert to_pconc(1000.0, "IU/ml") is None
        # not pConc
        assert to_pconc(0.1, "%") is None
        # or empty
        assert to_pconc(0.0, "µM") is None
        assert to_pconc(-5.0, "µM") is None
        assert to_pconc(None, "µM") is None
