"""Tests for sequential treatments, pulse-chase protocols, and treatment DAG schedules ( .3)."""
import pytest
from perturbalign import (
    align,
    PreprocessConfig,
    AlignResult,
    TreatmentStage,
    split_sequential_regime,
)
from perturbalign.schema import STATUS_COMBO, MODALITY_COMBO


def test_split_sequential_regime_arrow():
    text = "Pre-treatment with IFN-g (24h) -> followed by Trametinib (10nM, 12h)"
    stages = split_sequential_regime(text)
    assert stages is not None
    assert len(stages) == 2
    assert stages[0]["action_type"] == "pretreat"
    assert "IFN-g" in stages[0]["text"]
    assert stages[0]["duration"] == 24.0
    assert stages[0]["duration_unit"] == "h"

    assert stages[1]["action_type"] == "treatment"
    assert "Trametinib" in stages[1]["text"]
    assert stages[1]["dose"] == 10.0
    assert stages[1]["unit"] == "nM"
    assert stages[1]["duration"] == 12.0
    assert stages[1]["duration_unit"] == "h"


def test_split_sequential_regime_pulse_chase():
    text = "Pulse 2h Doxorubicin followed by 48h chase in fresh media"
    stages = split_sequential_regime(text)
    assert stages is not None
    assert len(stages) == 2
    assert stages[0]["action_type"] == "pulse"
    assert "Doxorubicin" in stages[0]["text"]
    assert stages[0]["duration"] == 2.0
    assert stages[0]["duration_unit"] == "h"

    assert stages[1]["action_type"] == "chase"
    assert "fresh media" in stages[1]["text"]
    assert stages[1]["duration"] == 48.0
    assert stages[1]["duration_unit"] == "h"


def test_split_sequential_regime_three_stages_washout():
    text = "Doxorubicin 1uM (4h) -> Washout -> Media (24h)"
    stages = split_sequential_regime(text)
    assert stages is not None
    assert len(stages) == 3
    assert stages[0]["action_type"] == "treatment"
    assert stages[0]["dose"] == 1.0
    assert stages[0]["unit"] == "µM"
    assert stages[0]["duration"] == 4.0
    assert stages[1]["action_type"] == "washout"
    assert stages[2]["action_type"] == "treatment"
    assert stages[2]["duration"] == 24.0


def test_sequential_alignment_pipeline():
    inputs = [
        "Pre-treatment with IFN-g (24h) -> followed by Trametinib (10nM, 12h)",
        "Pulse 2h Doxorubicin followed by 48h chase in fresh media",
        "Stage 1: TGF-b1 (48h) -> Stage 2: BMP4 (24h)",
    ]
    results = align(inputs, online=False)
    assert len(results) == 3

    # Check stage 1
    r0 = results[0]
    assert r0.status == STATUS_COMBO
    assert r0.modality == MODALITY_COMBO
    assert r0.regime_type == "sequential"
    assert len(r0.stages) == 2
    assert len(r0.parts) == 2
    assert r0.stages[0].action_type == "pretreat"
    assert r0.stages[0].duration == 24.0
    assert r0.stages[1].action_type == "treatment"
    assert r0.stages[1].result.parsed_dose == 10.0
    assert r0.stages[1].result.parsed_unit == "nM"
    assert r0.stages[1].duration == 12.0
    assert len(r0.stage_names) == 2
    assert (24.0, "h") in r0.stage_durations

    # Check pulse-chase
    r1 = results[1]
    assert r1.status == STATUS_COMBO
    assert r1.regime_type == "pulse_chase"
    assert len(r1.stages) == 2
    assert r1.stages[0].action_type == "pulse"
    assert r1.stages[0].duration == 2.0
    assert r1.stages[1].action_type == "chase"
    assert r1.stages[1].duration == 48.0
    # Media in chase is control!
    assert r1.stages[1].result.is_control is True

    # Check as_dict contains regime_type and stages
    d0 = r0.as_dict()
    assert d0["regime_type"] == "sequential"
    assert len(d0["stages"]) == 2
    assert d0["stages"][0]["action_type"] == "pretreat"
    assert d0["stages"][0]["result"] is not None


def test_sequential_regime_non_conflicting_smiles_or_inchi():
    # InChI strings with slashes or hyphens must not be broken
    inchi = "InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3-,4+,5-,6?/m1/s1"
    assert split_sequential_regime(inchi) is None

    # SMILES with reaction or arrows
    res = align([inchi], online=False)
    assert res[0].regime_type == "single"
