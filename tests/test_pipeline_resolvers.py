"""Unit tests for the AlignmentPipeline and BaseResolver architecture."""
import pytest
from perturbalign import (
    AlignResult,
    AlignStatus,
    AlignmentContext,
    AlignmentPipeline,
    BaseResolver,
    Candidate,
    DrugEntity,
    GeneticEntity,
    Modality,
    align,
)


class CustomLIMSResolver(BaseResolver):
    """        compound code parse  。"""

    name = "CustomLIMSResolver"

    def __init__(self, internal_db: dict[str, dict] | None = None) -> None:
        self.db = internal_db or {
            "LAB-101": {
                "name": "Bortezomib",
                "cid": 387447,
                "smiles": "CC(C)CC(NC(=O)C(Cc1ccccc1)NC(=O)c2nccnc2)B(O)O",
                "targets": ["PSMB5"],
            },
            "LAB-999": {
                "name": "Compound-Omega",
                "cid": 999999,
                "smiles": "C1=CC=CC=C1",
                "targets": ["SECRET_TARGET"],
            },
        }

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        code = ctx.cleaned.upper()
        if code in self.db:
            info = self.db[code]
            entity = DrugEntity(
                input_string=ctx.cleaned,
                canonical_name=info["name"],
                canonical_smiles=info["smiles"],
                pubchem_cid=info["cid"],
                targets=info["targets"],
                confidence=1.0,
                match_path=["custom:lims"],
                source="in_house_lims",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"internal_lims": 1.0},
                matched_by="CustomLIMSResolver",
                matched_surface=ctx.cleaned,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=AlignStatus.EXACT,
                modality=Modality.SMALL_MOLECULE,
                parsed_name=info["name"],
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note="matched via CustomLIMSResolver",
            )
        return None


def test_custom_resolver_via_extra_resolvers():
    """Verify    extra_resolvers      parse  no  parse     。"""
    lims_resolver = CustomLIMSResolver()
    results = align(
        ["LAB-101 10nM 24h", "LAB-999 5uM"],
        online=False,
        extra_resolvers=[lims_resolver],
    )
    assert len(results) == 2
    r1, r2 = results

    assert r1.status == "exact"
    assert r1.entity is not None
    assert r1.entity.canonical_name == "Bortezomib"
    assert r1.entity.pubchem_cid == 387447
    assert r1.parsed_dose == 10.0
    assert r1.parsed_unit == "nM"
    assert r1.parsed_duration == 24.0
    assert r1.entity.targets == ["PSMB5"]

    assert r2.status == "exact"
    assert r2.entity.canonical_name == "Compound-Omega"
    assert r2.parsed_dose == 5.0
    assert r2.parsed_unit in ("uM", "µM")


def test_custom_pipeline_lifecycle():
    """Verify AlignmentPipeline    、       。"""
    pipeline = AlignmentPipeline()
    initial_len = len(pipeline)

    res1 = CustomLIMSResolver()
    pipeline.add_resolver(res1, priority=0)
    assert len(pipeline) == initial_len + 1
    assert pipeline.resolvers[0] is res1

    # 
    removed = pipeline.remove_resolver("CustomLIMSResolver")
    assert removed is True
    assert len(pipeline) == initial_len

    # before
    pipeline.add_resolver(res1, before="CachedCorpusResolver")
    idx_cached = [i for i, r in enumerate(pipeline.resolvers) if r.name == "CachedCorpusResolver"][0]
    idx_lims = [i for i, r in enumerate(pipeline.resolvers) if r.name == "CustomLIMSResolver"][0]
    assert idx_lims == idx_cached - 1


def test_custom_resolver_in_combo():
    """Verify Combination drug parsing          parse  。"""
    lims_resolver = CustomLIMSResolver()
    pipeline = AlignmentPipeline()
    pipeline.add_resolver(lims_resolver, before="CachedCorpusResolver")

    results = align(
        ["LAB-101 10nM + LAB-999 5uM"],
        online=False,
        pipeline=pipeline,
    )
    assert len(results) == 1
    res = results[0]
    assert res.status == "combo"
    assert len(res.parts) == 2
    assert res.parts[0].entity.canonical_name == "Bortezomib"
    assert res.parts[0].parsed_dose == 10.0
    assert res.parts[1].entity.canonical_name == "Compound-Omega"
    assert res.parts[1].parsed_dose == 5.0
