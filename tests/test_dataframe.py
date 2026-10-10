"""Tests for Native DataFrame Annotation & Combo Explosion (P0).

Verify annotate_dataframe dataframe alignment 、 has structure  column   、combination structure      explode_combos explode 。
"""
from __future__ import annotations

import pandas as pd
import pytest
from perturbalign import (
    annotate_dataframe,
    annotate_anndata,
    explode_combos,
    align,
)


class TestAnnotateDataFrame:
    """Verify annotate_dataframe   Pandas dataframe    alignment    。"""

    def test_basic_dataframe_annotation(self):
        df = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3"],
            "perturbation": ["Trametinib 10nM 24h", "DMSO", "sgTP53"],
        })

        out = annotate_dataframe(df, pert_col="perturbation", online=False, inplace=False)
        assert len(out) == 3
        assert "perturbalign_status" in out.columns
        assert "perturbalign_modality" in out.columns
        assert "perturbalign_parsed_dose" in out.columns
        assert "perturbalign_parsed_unit" in out.columns
        assert "perturbalign_is_control" in out.columns

        assert out["perturbalign_is_control"].tolist() == [False, True, False]
        assert out["perturbalign_parsed_dose"].iloc[0] == 10.0
        assert out["perturbalign_parsed_unit"].iloc[0] == "nM"
        assert out["perturbalign_parsed_duration"].iloc[0] == 24.0

        # DataFrame
        assert "perturbalign_status" not in df.columns

    def test_structured_columns_fusion(self):
        # data ，compound 、dose and in column
        df = pd.DataFrame({
            "Compound": ["Doxorubicin", "Bortezomib"],
            "Dose": [50.0, 100.0],
            "Unit": ["nM", "nM"],
            "Time": [48.0, 72.0],
            "TimeUnit": ["h", "h"],
        })

        out = annotate_dataframe(
            df,
            pert_col="Compound",
            dose_col="Dose",
            unit_col="Unit",
            time_col="Time",
            time_unit_col="TimeUnit",
            online=False,
            inplace=False,
        )

        assert out["perturbalign_parsed_dose"].tolist() == [50.0, 100.0]
        assert out["perturbalign_parsed_unit"].tolist() == ["nM", "nM"]
        assert out["perturbalign_parsed_duration"].tolist() == [48.0, 72.0]
        assert out["perturbalign_parsed_duration_unit"].tolist() == ["h", "h"]

    def test_standardize_doses_in_dataframe(self):
        # dose to µM
        df = pd.DataFrame({
            "perturbation": ["DrugA 100nM", "DrugB 0.05mM", "DrugC 10uM"],
        })
        out = annotate_dataframe(df, pert_col="perturbation", standardize_doses=True, online=False)

        doses = out["perturbalign_parsed_dose"].tolist()
        units = out["perturbalign_parsed_unit"].tolist()

        assert units == ["µM", "µM", "µM"]
        assert abs(doses[0] - 0.1) < 1e-6
        assert abs(doses[1] - 50.0) < 1e-6
        assert abs(doses[2] - 10.0) < 1e-6

    def test_audit_dictionary_integration(self):
        df = pd.DataFrame({"perturbation": ["DMSO", "Trametinib 10nM"]})
        out, audit = annotate_dataframe(df, return_audit=True, online=False)
        assert audit is not None
        assert audit["n_total"] == 2
        assert "stats" in audit
        assert hasattr(out, "attrs") and "perturbalign" in out.attrs


class TestComboStructuredSupport:
    """Verify combination /   perturbation structure      explode_combos。"""

    def test_combo_structured_columns_in_dataframe(self):
        df = pd.DataFrame({
            "perturbation": [
                "Trametinib 10nM",
                "Paclitaxel 10nM + Carboplatin 50uM",
            ]
        })
        out = annotate_dataframe(df, online=False)

        assert "perturbalign_combo_names" in out.columns
        assert "perturbalign_combo_doses" in out.columns
        assert "perturbalign_combo_units" in out.columns

        # single agent row combo column is None
        assert pd.isna(out["perturbalign_combo_names"].iloc[0])

        # combination row combo column
        combo_names = out["perturbalign_combo_names"].iloc[1]
        assert "Paclitaxel" in combo_names
        assert "Carboplatin" in combo_names

        combo_doses = out["perturbalign_combo_doses"].iloc[1]
        assert "10.0" in combo_doses
        assert "50.0" in combo_doses

    def test_combo_structured_columns_in_anndata(self):
        class StubAdata:
            def __init__(self, obs_df):
                self.obs = obs_df
                self.uns = {}

        obs = pd.DataFrame({
            "perturbation": ["Paclitaxel 10nM + Carboplatin 50uM", "DMSO"]
        })
        adata = StubAdata(obs)
        out = annotate_anndata(adata, online=False, inplace=True)

        assert "perturbalign_combo_names" in out.obs.columns
        assert "perturbalign_combo_doses" in out.obs.columns
        combo_names = out.obs["perturbalign_combo_names"].iloc[0]
        assert "Paclitaxel" in combo_names
        assert "Carboplatin" in combo_names

    def test_explode_combos(self):
        df = pd.DataFrame({
            "plate_id": ["P1", "P2"],
            "perturbation": [
                "Trametinib 10nM",
                "Paclitaxel 10nM + Carboplatin 50uM",
            ]
        })
        annotated = annotate_dataframe(df, online=False)
        exploded = explode_combos(annotated)

        # 2 row data ， 1 row is single agent 1 row ， 2 row is explode is 2 row ， 3 row
        assert len(exploded) == 3

        # Verify explode properties
        assert "component_name" in exploded.columns
        assert "component_index" in exploded.columns
        assert "is_combo_component" in exploded.columns

        # 0 row (single agent )
        assert exploded["component_name"].iloc[0] == "Trametinib"
        assert exploded["component_index"].iloc[0] == 0
        assert bool(exploded["is_combo_component"].iloc[0]) is False

        # 1 row ( 1)
        assert exploded["component_name"].iloc[1] == "Paclitaxel"
        assert exploded["component_index"].iloc[1] == 0
        assert exploded["component_dose"].iloc[1] == 10.0
        assert exploded["component_unit"].iloc[1] == "nM"
        assert bool(exploded["is_combo_component"].iloc[1]) is True

        # 2 row ( 2)
        assert exploded["component_name"].iloc[2] == "Carboplatin"
        assert exploded["component_index"].iloc[2] == 1
        assert exploded["component_dose"].iloc[2] == 50.0
        assert exploded["component_unit"].iloc[2] == "µM"
        assert bool(exploded["is_combo_component"].iloc[2]) is True
