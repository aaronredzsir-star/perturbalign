"""Unit tests for In-Memory Chainable Configuration (PreprocessConfig.with_extra_*).

Tests:
  - Chainable configuration creation without disk I/O.
  - Verification that DEFAULT_CONFIG remains immutable.
  - Adding custom biologics, environmental stressors, inorganic compounds, metal cations,
    salt suffixes, combo separators, dose units, duration units, genetic actions,
    and control targets.
"""
from __future__ import annotations

import pytest
from perturbalign.preprocess import (
    DEFAULT_CONFIG,
    PreprocessConfig,
    clean_name,
    detect_biologic,
    detect_control,
    detect_environmental,
    detect_genetic_perturbation,
    parse_dose,
    parse_duration,
    split_combo,
)


class TestInMemoryConfig:
    def test_with_extra_controls_chaining(self):
        cfg = DEFAULT_CONFIG.with_extra_controls(["my_special_ctrl", "water_control"])
        assert detect_control("my_special_ctrl", cfg) is True
        assert detect_control("water_control", cfg) is True
        # Original default config remains unmutated
        assert detect_control("my_special_ctrl", DEFAULT_CONFIG) is False

    def test_with_extra_biologics(self):
        cfg = DEFAULT_CONFIG.with_extra_biologics(
            {"custom-factor-x": {"canonical_name": "Custom Factor X", "category": "growth_factor"}},
            custom_protein="Custom Protein Alpha",
        )
        b1 = detect_biologic("custom-factor-x", cfg)
        assert b1 is not None
        assert b1["canonical_name"] == "Custom Factor X"

        b2 = detect_biologic("custom_protein", cfg)
        assert b2 is not None
        assert b2["canonical_name"] == "Custom Protein Alpha"

        assert detect_biologic("custom-factor-x", DEFAULT_CONFIG) is None

    def test_with_extra_environmental(self):
        cfg = DEFAULT_CONFIG.with_extra_environmental(["microgravity", "acoustic_levitation"])
        e1 = detect_environmental("microgravity", cfg)
        assert e1 is not None
        assert e1["canonical_name"] == "microgravity"

        assert detect_environmental("microgravity", DEFAULT_CONFIG) is None

    def test_with_extra_inorganics_and_metals(self):
        # Adding custom metal cation and inorganic compound
        cfg = DEFAULT_CONFIG.with_extra_metal_cations("mycation").with_extra_inorganics("borate")
        # clean_name protects inorganic compounds from prefix stripping
        cleaned = clean_name("mycation borate", cfg)
        assert cleaned == "mycation borate"

    def test_with_extra_salt_suffixes(self):
        cfg = DEFAULT_CONFIG.with_extra_salt_suffixes("mycustomsalt", "polysilicate")
        cleaned = clean_name("CompoundX mycustomsalt", cfg)
        assert cleaned == "CompoundX"
        assert clean_name("CompoundX mycustomsalt", DEFAULT_CONFIG) == "CompoundX mycustomsalt"

    def test_with_extra_combo_separators(self):
        cfg = DEFAULT_CONFIG.with_extra_combo_separators(" COMBINED_WITH ", (" ## ", 50))
        parts = split_combo("DrugA COMBINED_WITH DrugB", cfg)
        assert parts == ["DrugA", "DrugB"]
        assert split_combo("DrugA COMBINED_WITH DrugB", DEFAULT_CONFIG) is None

    def test_with_extra_dose_and_duration_units(self):
        cfg = (
            DEFAULT_CONFIG
            .with_extra_dose_units(("parts_per_million", "ppm"), ("ppm", "ppm"))
            .with_extra_duration_units(("weeks", "w"), ("wk", "w"))
        )
        name, dose, unit = parse_dose("Aspirin 50ppm", cfg)
        assert name == "Aspirin"
        assert dose == 50.0
        assert unit == "ppm"

        name, dur, dur_u = parse_duration("Aspirin 2weeks", cfg)
        assert name == "Aspirin"
        assert dur == 2.0
        assert dur_u == "w"

    def test_with_extra_genetic_actions_and_controls(self):
        cfg = (
            DEFAULT_CONFIG
            .with_extra_genetic_actions(overexpression="OE", degradation="PROTAC")
            .with_extra_genetic_control_targets("my_safe_site", "ctrl_guide_1")
        )
        target, act = detect_genetic_perturbation("TP53_overexpression", cfg)
        assert target == "TP53"
        assert act == "OE"

        # Safe site is treated as control rather than target
        assert detect_genetic_perturbation("sgmy_safe_site", cfg) is None

    def test_fluent_pipeline_chaining(self):
        # Full multi-step fluent chain
        cfg = (
            PreprocessConfig()
            .with_extra_controls(["custom_neg"])
            .with_extra_biologics(["AlphaFoldProtein"])
            .with_extra_environmental(["hyperbaric"])
            .with_extra_salt_suffixes("co-crystal")
        )
        assert detect_control("custom_neg", cfg) is True
        assert detect_biologic("AlphaFoldProtein", cfg) is not None
        assert detect_environmental("hyperbaric", cfg) is not None
        assert clean_name("DrugY co-crystal", cfg) == "DrugY"
