"""        perturbation data         Test   。

        single-cell     perturbation   ：
1. scPerturb / Norman et al. 2019:  target / gene knockout       (DUSP9+ETS2, Control)
2. scPerturb / Srivatsan et al. 2020 (sci-Plex 3):   small molecule compound   、     concentration (BMS-345541_10.0_uM, Vehicle)
3. scPerturb / Replogle et al. 2022: all gene   CRISPRi       control (non-targeting, sgTP53, RPE1_control)
4. LINCS L1000 (GEO GSE92742): Broad    data    (pert_iname, pert_type, pert_dose, pert_time)
5. DepMap PRISM (Broad Repurposing Hub):            (BRD-K...::dose::screen)
"""
import pytest
import pandas as pd
import perturbalign as pa
from perturbalign.schema import (
    MODALITY_CONTROL,
    MODALITY_GENETIC,
    MODALITY_SMALL_MOLECULE,
    MODALITY_COMBO,
    STATUS_CONTROL,
    STATUS_EXACT,
)


def test_norman_2019_crispr_combos():
    """Verify scPerturb   Norman et al. 2019  gene knockout /CRISPRa     parse 。"""
    # 1. scPerturb adapter ( perturbation_type)
    records = [
        {"perturbation": "Control", "perturbation_type": "CRISPRa"},
        {"perturbation": "DUSP9+ETS2", "perturbation_type": "CRISPRa"},
        {"perturbation": "KLF1", "perturbation_type": "CRISPRa"},
        {"perturbation": "CBL+CNN1", "perturbation_type": "CRISPRa"},
        {"perturbation": "neg_control_1", "perturbation_type": "CRISPRa"},
        {"perturbation": "PTPN12+SNAI1", "perturbation_type": "CRISPRa"},
    ]
    results = [pa.from_scperturb(r) for r in records]
    assert len(results) == 6

    # control
    assert results[0].is_control is True
    assert results[0].status == STATUS_CONTROL
    assert results[4].is_control is True

    # gene knockout
    assert results[2].modality == MODALITY_GENETIC
    assert results[2].genetic_target == "KLF1"

    # gene knockout
    res_dusp = pa.align(["DUSP9+ETS2"], online=False)[0]
    assert res_dusp.modality == MODALITY_COMBO
    assert res_dusp.parts is not None
    assert len(res_dusp.parts) == 2
    genes = [p.parsed_name for p in res_dusp.parts]
    assert "DUSP9" in genes
    assert "ETS2" in genes


def test_srivatsan_2020_sciplex_doses():
    """Verify scPerturb   Srivatsan et al. 2020 (sci-Plex 3)      compound  concentration   。"""
    raw_labels = [
        "Vehicle",
        "BMS-345541_10.0_uM",
        "dexamethasone_100_nM",
        "nutlin-3a_1.0_uM",
        "trametinib_10_nM",
        "DMSO",
    ]
    results = pa.align(raw_labels, online=False)
    assert len(results) == 6

    # 1. control
    assert results[0].is_control is True
    assert results[5].is_control is True

    # 2. concentration dose
    bms = results[1]
    assert bms.is_control is False
    assert bms.parsed_dose == 10.0
    assert bms.parsed_unit in ["uM", "µM"]
    assert "bms-345541" in bms.parsed_name.lower()

    dex = results[2]
    assert dex.parsed_dose == 100.0
    assert dex.parsed_unit == "nM"
    assert "dexamethasone" in dex.parsed_name.lower()


def test_replogle_2022_perturb_seq():
    """Verify Replogle et al. 2022 (K562/RPE1)      control     RNA   。"""
    raw_labels = [
        "non-targeting",
        "sgTP53",
        "neg_control",
        "sgMYC",
        "RPE1_control",
    ]
    results = pa.align(raw_labels, online=False)

    assert results[0].is_control is True
    assert results[2].is_control is True
    assert results[4].is_control is True

    # gene
    assert results[1].modality == MODALITY_GENETIC
    assert results[1].genetic_target == "TP53"
    assert results[3].genetic_target == "MYC"


def test_lincs_l1000_metadata_records():
    """Verify LINCS L1000    perturbagen  data         。"""
    records = [
        {
            "pert_id": "BRD-K12345678",
            "pert_iname": "vorinostat",
            "pert_type": "trt_cp",
            "pert_dose": 10.0,
            "pert_dose_unit": "uM",
            "pert_time": 24.0,
            "pert_time_unit": "h",
        },
        {
            "pert_id": "BRD-CONTROL",
            "pert_iname": "DMSO",
            "pert_type": "ctl_vehicle",
            "pert_dose": 0.1,
            "pert_dose_unit": "%",
            "pert_time": 24.0,
            "pert_time_unit": "h",
        },
        {
            "pert_id": "TRCN0000001234",
            "pert_iname": "TP53",
            "pert_type": "trt_sh",
            "pert_dose": None,
            "pert_dose_unit": None,
            "pert_time": 96.0,
            "pert_time_unit": "h",
        },
    ]

    adapter = pa.LINCSAdapter()
    res0 = adapter.adapt(records[0])
    assert res0.modality == MODALITY_SMALL_MOLECULE
    assert res0.parsed_name == "vorinostat"
    assert res0.parsed_dose == 10.0

    res1 = adapter.adapt(records[1])
    assert res1.is_control is True
    assert res1.modality == MODALITY_CONTROL

    res2 = adapter.adapt(records[2])
    assert res2.modality == MODALITY_GENETIC
    assert res2.genetic_target == "TP53"
    assert "shRNA" in res2.note or "trt_sh" in res2.note


def test_prism_composite_identifiers():
    """Verify PRISM / Repurposing Hub         parse 。"""
    adapter = pa.PRISMAdapter()
    res = adapter.adapt("BRD-A00145892::2.5::MTS001")
    assert res.parsed_name == "BRD-A00145892"
    assert res.parsed_dose == 2.5
