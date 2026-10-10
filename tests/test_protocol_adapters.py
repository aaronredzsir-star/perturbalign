"""Unit tests for Dataset Protocol Adapter SPI (BaseProtocolAdapter & ProtocolRegistry).

Tests:
  - BaseProtocolAdapter contract.
  - Built-in adapters: LINCSAdapter, ScPerturbAdapter, PRISMAdapter.
  - ProtocolRegistry dynamic registration, unregistration, lookup, list_protocols.
  - Adding a custom dataset protocol adapter (e.g., Cell Painting profile adapter).
  - adapt_protocol unified API.
"""
from __future__ import annotations

import pytest
from perturbalign.adapter import (
    BaseProtocolAdapter,
    LINCSAdapter,
    PRISMAdapter,
    ProtocolRegistry,
    ScPerturbAdapter,
    adapt_protocol,
    from_lincs,
    from_prism,
    from_scperturb,
)
from perturbalign.cache import MemoryCache
from perturbalign.schema import (
    AlignResult,
    MODALITY_CONTROL,
    MODALITY_GENETIC,
    STATUS_CONTROL,
    STATUS_UNMATCHED,
)


class MockCellPaintingAdapter(BaseProtocolAdapter):
    """Custom adapter for Cell Painting Consortium morphological profiling datasets."""

    @property
    def protocol_key(self) -> str:
        return "cell_painting"

    def adapt(self, record, config=None, **kwargs) -> AlignResult:
        well_id = record.get("well_id", "")
        compound = record.get("broad_sample", "")
        is_dmso = compound.upper() == "DMSO" or record.get("is_control", False)

        if is_dmso:
            return AlignResult(
                input=well_id,
                status=STATUS_CONTROL,
                is_control=True,
                modality=MODALITY_CONTROL,
                parsed_name="DMSO",
                note="Cell Painting negative control",
            )
        return AlignResult(
            input=well_id,
            status=STATUS_UNMATCHED,
            is_control=False,
            modality="small_molecule",
            parsed_name=compound,
            parsed_dose=record.get("dose_uM"),
            parsed_unit="µM",
            note=f"Cell Painting compound {compound}",
        )


class TestProtocolRegistry:
    def test_default_protocols(self):
        from perturbalign import JUMPCPAdapter
        reg = ProtocolRegistry()
        protocols = reg.list_protocols()
        assert protocols == ["jump_cp", "lincs", "prism", "scperturb"]
        assert isinstance(reg.get_adapter("lincs"), LINCSAdapter)
        assert isinstance(reg.get_adapter("scperturb"), ScPerturbAdapter)
        assert isinstance(reg.get_adapter("prism"), PRISMAdapter)
        assert isinstance(reg.get_adapter("jump_cp"), JUMPCPAdapter)

    def test_custom_protocol_registration(self):
        reg = ProtocolRegistry()
        reg.register_adapter(MockCellPaintingAdapter())
        assert "cell_painting" in reg.list_protocols()

        rec = {"well_id": "A01", "broad_sample": "BRD-K12345678", "dose_uM": 10.0}
        res = reg.adapt_protocol("cell_painting", rec)
        assert res.parsed_name == "BRD-K12345678"
        assert res.parsed_dose == 10.0
        assert res.parsed_unit == "µM"
        assert "Cell Painting" in res.note

        ctrl_rec = {"well_id": "B01", "broad_sample": "DMSO", "is_control": True}
        res_ctrl = reg.adapt_protocol("cell_painting", ctrl_rec)
        assert res_ctrl.is_control is True
        assert res_ctrl.status == STATUS_CONTROL

    def test_unregister(self):
        reg = ProtocolRegistry()
        reg.unregister_adapter("lincs")
        assert "lincs" not in reg.list_protocols()

    def test_unknown_protocol_raises(self):
        reg = ProtocolRegistry()
        with pytest.raises(KeyError, match="Unregistered protocol adapter"):
            reg.adapt_protocol("non_existent_protocol", {})


class TestStandardProtocolAdapters:
    def test_from_lincs_via_adapter(self):
        res = from_lincs(
            pert_id="BRD-K81418486",
            pert_type="trt_cp",
            pert_iname="vorinostat",
            pert_dose=10.0,
            pert_dose_unit="uM",
            pert_time=24.0,
            pert_time_unit="h",
            online=False,
            cache=MemoryCache(),
        )
        assert res.input == "BRD-K81418486"
        assert res.parsed_dose == 10.0
        assert res.parsed_unit == "uM"
        assert res.parsed_duration == 24.0
        assert res.parsed_duration_unit == "h"

    def test_from_scperturb_via_adapter(self):
        row = {"perturbation": "sgTP53", "perturbation_type": "CRISPR"}
        res = from_scperturb(row)
        assert res.modality == MODALITY_GENETIC
        assert res.genetic_target == "TP53"
        assert res.genetic_action == "KO"

    def test_from_prism_via_adapter(self):
        res = from_prism("BRD-A00077618-236-07-6::2.5::HTS", online=False, cache=MemoryCache())
        assert res.parsed_name == "BRD-A00077618-236-07-6"
        assert res.parsed_dose == 2.5
        assert res.parsed_unit == "µM"
        assert "HTS" in res.note

    def test_adapt_protocol_unified_function(self):
        res = adapt_protocol(
            "prism",
            "DMSO::0::MTS",
        )
        assert res.is_control is True
        assert res.status == STATUS_CONTROL
