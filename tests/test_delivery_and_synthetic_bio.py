"""Tests for delivery systems & nanocarriers, synthetic biology, PROTACs, and immune cocktails ( )."""
import pytest
from perturbalign import (
    align,
    PreprocessConfig,
    DeliveryVehicleEntity,
    GeneticEntity,
    DrugEntity,
    detect_delivery_vehicle,
    detect_genetic_perturbation,
    detect_protac,
    MODALITY_DELIVERY,
)
from perturbalign.schema import MODALITY_GENETIC, STATUS_EXACT, STATUS_COMBO


def test_delivery_vehicle_standalone_and_cargo():
    # Standalone
    v_lnp = detect_delivery_vehicle("LNP")
    assert v_lnp is not None
    assert v_lnp["carrier_type"] == "lnp"

    v_mc3 = detect_delivery_vehicle("MC3")
    assert v_mc3 is not None
    assert "MC3" in v_mc3["formulation"]

    v_aav9 = detect_delivery_vehicle("AAV9")
    assert v_aav9 is not None
    assert v_aav9["serotype"] == "AAV9"

    v_aav_php = detect_delivery_vehicle("AAV-PHP.eB")
    assert v_aav_php is not None
    assert v_aav_php["serotype"] == "AAV-PHP.eB"

    v_lipo = detect_delivery_vehicle("Lipofectamine 3000")
    assert v_lipo is not None
    assert v_lipo["carrier_type"] == "transfection_reagent"

    # With Cargo
    v_lnp_mrna = detect_delivery_vehicle("LNP-mRNA")
    assert v_lnp_mrna is not None
    assert v_lnp_mrna["cargo"] == ["mRNA"]

    v_aav_cre = detect_delivery_vehicle("AAV-PHP.eB-Cre")
    assert v_aav_cre is not None
    assert v_aav_cre["serotype"] == "AAV-PHP.eB"
    assert v_aav_cre["cargo"] == ["Cre"]

    v_exo_mir = detect_delivery_vehicle("Exosome-miR-21")
    assert v_exo_mir is not None
    assert v_exo_mir["carrier_type"] == "exosome"
    assert v_exo_mir["cargo"] == ["miR-21"]


def test_delivery_vehicle_pipeline_alignment():
    inputs = ["LNP-mRNA", "AAV9-GFP", "SM-102"]
    results = align(inputs, online=False)
    assert len(results) == 3

    r0 = results[0]
    assert r0.status == STATUS_EXACT
    assert r0.modality == MODALITY_DELIVERY
    assert isinstance(r0.entity, DeliveryVehicleEntity)
    assert r0.entity.carrier_type == "lnp"
    assert r0.entity.cargo == ["mRNA"]

    r1 = results[1]
    assert r1.status == STATUS_EXACT
    assert r1.modality == MODALITY_DELIVERY
    assert isinstance(r1.entity, DeliveryVehicleEntity)
    assert r1.entity.serotype == "AAV9"
    assert r1.entity.cargo == ["GFP"]

    r2 = results[2]
    assert r2.status == STATUS_EXACT
    assert r2.modality == MODALITY_DELIVERY
    assert r2.entity.carrier_type == "lnp"

    # Lipofectamine as transfection reagent is properly recognized as vehicle control
    r_lipo = align(["Lipofectamine 3000"], online=False)[0]
    assert r_lipo.is_control is True


def test_synthetic_biology_and_epigenome_editing():
    # Epigenome editing
    res_krab = detect_genetic_perturbation("dCas9-KRAB_MYC")
    assert res_krab is not None
    assert res_krab == ("MYC", "epigenetic_edit:dCas9-KRAB")

    res_tet1 = detect_genetic_perturbation("dCas9-TET1_BRCA1")
    assert res_tet1 is not None
    assert res_tet1 == ("BRCA1", "epigenetic_edit:dCas9-TET1")

    # Chemogenetics / DREADD
    res_dreadd = detect_genetic_perturbation("hM3Dq")
    assert res_dreadd is not None
    assert res_dreadd[1] == "chemogenetics:hM3Dq"

    res_kord = detect_genetic_perturbation("KORD")
    assert res_kord is not None
    assert res_kord[1] == "chemogenetics:KORD"

    # Optogenetics
    res_opto = detect_genetic_perturbation("ChR2")
    assert res_opto is not None
    assert res_opto[1] == "optogenetics:ChR2"

    res_nphr = detect_genetic_perturbation("NpHR")
    assert res_nphr is not None
    assert res_nphr[1] == "optogenetics:NpHR"

    # Pipeline alignment
    align_res = align(["dCas9-KRAB_MYC", "hM3Dq", "ChR2"], online=False)
    assert len(align_res) == 3
    assert align_res[0].modality == MODALITY_GENETIC
    assert isinstance(align_res[0].entity, GeneticEntity)
    assert align_res[0].entity.system_type == "epigenome_edit"
    assert align_res[0].entity.effector == "dCas9-KRAB"
    assert align_res[0].genetic_target == "MYC"

    assert align_res[1].entity.system_type == "chemogenetics"
    assert align_res[1].entity.effector == "hM3Dq"

    assert align_res[2].entity.system_type == "optogenetics"
    assert align_res[2].entity.effector == "ChR2"


def test_protac_and_molecular_glues():
    p_arv = detect_protac("ARV-110")
    assert p_arv is not None
    assert p_arv["e3_ligase"] == "CRBN"
    assert p_arv["target_protein"] == "AR"
    assert p_arv["is_molecular_glue"] is False

    p_arv471 = detect_protac("ARV-471")
    assert p_arv471 is not None
    assert p_arv471["e3_ligase"] == "VHL"

    p_pattern = detect_protac("BRD4-PROTAC(VHL)")
    assert p_pattern is not None
    assert p_pattern["target_protein"] == "BRD4"
    assert p_pattern["e3_ligase"] == "VHL"

    p_glue = detect_protac("Thalidomide")
    assert p_glue is not None
    assert p_glue["is_molecular_glue"] is True
    assert p_glue["e3_ligase"] == "CRBN"

    p_indisulam = detect_protac("Indisulam")
    assert p_indisulam is not None
    assert p_indisulam["is_molecular_glue"] is True
    assert p_indisulam["e3_ligase"] == "DCAF15"


def test_immune_cocktails():
    inputs = [
        "M1 macrophage cocktail",
        "Th1 differentiation cocktail",
        "T-cell activation",
    ]
    results = align(inputs, online=False)
    assert len(results) == 3

    r0 = results[0]
    assert r0.status == STATUS_COMBO
    assert "M1 Macrophage" in r0.parsed_name
    assert len(r0.parts) == 2
    part_names = [p.parsed_name for p in r0.parts]
    assert "LPS" in part_names
    assert "IFN-gamma" in part_names

    r1 = results[1]
    assert r1.status == STATUS_COMBO
    assert len(r1.parts) == 2
    part_names1 = [p.parsed_name for p in r1.parts]
    assert "IL-12" in part_names1
    assert "anti-IL-4" in part_names1

    r2 = results[2]
    assert r2.status == STATUS_COMBO
    assert len(r2.parts) == 2
    part_names2 = [p.parsed_name for p in r2.parts]
    assert "anti-CD3" in part_names2
    assert "anti-CD28" in part_names2


def test_custom_cocktail_and_delivery_extension():
    cfg = (
        PreprocessConfig()
        .with_extra_cocktails(my_recipe="DrugA + DrugB + DrugC")
        .with_extra_delivery_vehicles(aav_my=dict(canonical_name="AAV-MyCustom", carrier_type="aav", serotype="AAV-Custom"))
    )
    res = align(["my_recipe", "aav_my"], config=cfg, online=False)
    assert len(res) == 2
    assert res[0].status == STATUS_COMBO
    assert len(res[0].parts) == 3
    assert res[1].status == STATUS_EXACT
    assert res[1].modality == MODALITY_DELIVERY
    assert res[1].entity.serotype == "AAV-Custom"
