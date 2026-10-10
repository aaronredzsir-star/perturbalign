"""perturbalign.pipeline — Flexible and extensible perturbation alignment pipeline & resolver system.

Supports Chain of Responsibility pattern:
Each resolver (BaseResolver) handles specific logic or modality (user mapping, controls,
combinations, genetics, biologics, environmental, cached corpus, online APIs, etc.).
Users can seamlessly extend the pipeline by registering custom resolvers (e.g. proprietary
LIMS databases, in-house compound ID resolvers, proprietary CRISPR libraries).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import normalize as _norm
from . import preprocess as _pre
from .cache import BaseCache
from .schema import (
    STATUS_AMBIGUOUS,
    STATUS_COMBO,
    STATUS_CONTROL,
    STATUS_EMPTY,
    STATUS_EXACT,
    STATUS_ONLINE,
    STATUS_UNMATCHED,
    MODALITY_BIOLOGIC,
    MODALITY_COMBO,
    MODALITY_CONTROL,
    MODALITY_EMPTY,
    MODALITY_ENVIRONMENTAL,
    MODALITY_GENETIC,
    MODALITY_SMALL_MOLECULE,
    MODALITY_DELIVERY,
    AlignResult,
    AlignStatus,
    BiologicEntity,
    Candidate,
    DeliveryVehicleEntity,
    DrugEntity,
    EnvironmentalEntity,
    GeneticEntity,
    Modality,
    PerturbationEntity,
    TreatmentStage,
)


def enrich_entity(
    entity: PerturbationEntity | None,
    cleaned: str,
    cache: BaseCache | None,
    enrich_rdkit: bool,
    enrich_chembl: bool,
) -> None:
    """Perform RDKit parent InChIKey, ChEMBL target/MoA, and PROTAC/molecular glue enrichment on demand."""
    if entity is None or not isinstance(entity, DrugEntity):
        return
    if enrich_rdkit:
        try:
            from .structure import parent_inchikey
            entity.parent_inchikey = parent_inchikey(entity.canonical_smiles)
        except Exception:  # noqa: BLE001
            pass
    if enrich_chembl:
        try:
            from .annotate import chembl_targets_lookup
            info = chembl_targets_lookup(entity.canonical_name or cleaned, cache=cache)
        except Exception:  # noqa: BLE001
            info = None
        if info:
            entity.chembl_id = info.get("chembl_id") or entity.chembl_id
            entity.targets = info.get("targets") or entity.targets
            if info.get("moa"):
                entity.moa = info["moa"][0]
            entity.match_path.append("enrich:chembl")

    # PROTAC and molecular glue E3 ligase information enrichment
    try:
        from .preprocess import detect_protac
        protac_info = detect_protac(entity.canonical_name or cleaned) or detect_protac(cleaned)
        if protac_info:
            entity.is_protac = protac_info.get("is_protac", not protac_info.get("is_molecular_glue", False))
            entity.is_molecular_glue = protac_info.get("is_molecular_glue", False)
            entity.e3_ligase = protac_info.get("e3_ligase")
            entity.e3_ligand = protac_info.get("e3_ligand")
            if protac_info.get("target_protein") and not entity.targets:
                entity.targets = [t.strip() for t in protac_info["target_protein"].split(",") if t.strip()]
            entity.match_path.append("enrich:protac")
    except Exception:
        pass


def build_online_candidates(
    cleaned: str,
    hits: list[dict],
) -> tuple[str, DrugEntity | None, list[Candidate]]:
    """Build candidates and status from online API hit list."""
    if not hits:
        return STATUS_UNMATCHED, None, []
    candidates = []
    for i, hit in enumerate(hits):
        src = hit.get("source") or "pubchem"
        entity = DrugEntity(
            input_string=cleaned,
            canonical_name=hit.get("canonical_name") or cleaned,
            canonical_smiles=hit.get("canonical_smiles"),
            inchikey=hit.get("inchikey"),
            pubchem_cid=hit.get("pubchem_cid"),
            chembl_id=hit.get("chembl_id"),
            source=src,
            confidence=0.7 if i == 0 else 0.4,
            match_path=[f"online:{src}"],
        )
        candidates.append(
            Candidate(
                entity=entity,
                score=entity.confidence,
                score_breakdown={"online_order": entity.confidence},
                matched_by=f"online {src}",
                matched_surface=cleaned,
            )
        )
    if len(candidates) == 1:
        return STATUS_ONLINE, candidates[0].entity, candidates
    # Multi-candidate ambiguity
    return STATUS_AMBIGUOUS, None, candidates


@dataclass
class AlignmentContext:
    """Alignment context for a single input, carrying raw data, extracted features, runtime configuration, and cache."""

    raw_input: Any
    config: _pre.PreprocessConfig
    cache: BaseCache | None = None
    online: bool = False
    fuzzy: bool = False
    fuzzy_threshold: int = 80
    enrich_chembl: bool = False
    enrich_rdkit: bool = False
    user_mapping: dict[str, str | PerturbationEntity] | None = None
    options: dict[str, Any] = field(default_factory=dict)

    # Spatiotemporal and textual features extracted via preprocessing
    text: str = ""
    base: str = ""
    cleaned: str = ""
    dose: float | None = None
    unit: str | None = None
    duration: float | None = None
    duration_unit: str | None = None
    is_empty: bool = False
    empty_note: str = ""

    pipeline: AlignmentPipeline | None = None

    @classmethod
    def create(
        cls,
        raw_input: Any,
        config: _pre.PreprocessConfig,
        cache: BaseCache | None = None,
        online: bool = False,
        fuzzy: bool = False,
        fuzzy_threshold: int = 80,
        enrich_chembl: bool = False,
        enrich_rdkit: bool = False,
        user_mapping: dict[str, str | PerturbationEntity] | None = None,
        options: dict[str, Any] | None = None,
        pipeline: AlignmentPipeline | None = None,
    ) -> AlignmentContext:
        ctx = cls(
            raw_input=raw_input,
            config=config,
            cache=cache,
            online=online,
            fuzzy=fuzzy,
            fuzzy_threshold=fuzzy_threshold,
            enrich_chembl=enrich_chembl,
            enrich_rdkit=enrich_rdkit,
            user_mapping=user_mapping,
            options=options if options is not None else {},
            pipeline=pipeline,
        )
        ctx._init_features()
        return ctx

    def _init_features(self) -> None:
        norm_text, is_empty, empty_note = _pre.normalize_input_text(self.raw_input)
        if is_empty:
            self.is_empty = True
            self.empty_note = empty_note
            return

        self.text = norm_text
        base, dose, unit, duration, duration_unit = _pre.parse_spatiotemporal(self.text, self.config)
        self.base = _pre.strip_leading_parens(base)
        self.dose = dose
        self.unit = unit
        self.duration = duration
        self.duration_unit = duration_unit
        self.cleaned = _pre.clean_name(self.base, self.config)


class BaseResolver(abc.ABC):
    """Abstract base class for resolvers forming links in the Chain of Responsibility."""

    name: str = "BaseResolver"

    @abc.abstractmethod
    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        """Attempt to resolve the input. If resolvable by this resolver, return AlignResult; otherwise return None to delegate to the next resolver."""
        pass


class EmptyInputResolver(BaseResolver):
    """Empty and missing value resolver."""

    name = "EmptyInputResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        if not ctx.is_empty:
            return None
        input_str = "None" if ctx.raw_input is None else str(ctx.raw_input)
        return AlignResult(
            input=input_str,
            status=STATUS_EMPTY,
            modality=MODALITY_EMPTY,
            note=ctx.empty_note,
        )


class ControlResolver(BaseResolver):
    """Control and vehicle detection resolver."""

    name = "ControlResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        if _pre.detect_control(ctx.base, ctx.config):
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_CONTROL,
                is_control=True,
                modality=MODALITY_CONTROL,
                parsed_name=ctx.base,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
            )
        return None


class SequentialRegimeResolver(BaseResolver):
    """Sequential treatment, pulse-chase, and washout recovery experiment regimen resolver."""

    name = "SequentialRegimeResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        raw_stages = _pre.split_sequential_regime(str(ctx.raw_input), ctx.config)
        if not raw_stages or len(raw_stages) < 2:
            raw_stages = _pre.split_sequential_regime(ctx.base, ctx.config)
        if not raw_stages or len(raw_stages) < 2:
            return None

        is_pulse_chase = any(s["action_type"] in ("pulse", "chase") for s in raw_stages)
        regime_type = "pulse_chase" if is_pulse_chase else "sequential"

        stages: list[TreatmentStage] = []
        sub_results: list[AlignResult] = []

        for s in raw_stages:
            st_text = s["text"]
            if ctx.pipeline:
                sub_res = ctx.pipeline.resolve_single(st_text, parent_ctx=ctx)
                if s["dose"] is not None and sub_res.parsed_dose is None:
                    sub_res.parsed_dose = s["dose"]
                    sub_res.parsed_unit = s["unit"]
                if s["duration"] is not None and sub_res.parsed_duration is None:
                    sub_res.parsed_duration = s["duration"]
                    sub_res.parsed_duration_unit = s["duration_unit"]
            else:
                sub_res = None

            t_stage = TreatmentStage(
                stage_index=s["stage_index"],
                action_type=s["action_type"],
                duration=s["duration"] or (sub_res.parsed_duration if sub_res else None),
                duration_unit=s["duration_unit"] or (sub_res.parsed_duration_unit if sub_res else None),
                result=sub_res,
                description=s["raw_stage"],
            )
            stages.append(t_stage)
            if sub_res:
                sub_results.append(sub_res)

        if ctx.dose is not None and sub_results and sub_results[-1].parsed_dose is None:
            sub_results[-1].parsed_dose = ctx.dose
            sub_results[-1].parsed_unit = ctx.unit
        if ctx.duration is not None and stages and stages[-1].duration is None:
            stages[-1].duration = ctx.duration
            stages[-1].duration_unit = ctx.duration_unit
            if sub_results and sub_results[-1].parsed_duration is None:
                sub_results[-1].parsed_duration = ctx.duration
                sub_results[-1].parsed_duration_unit = ctx.duration_unit

        return AlignResult(
            input=str(ctx.raw_input),
            status=STATUS_COMBO,
            modality=MODALITY_COMBO,
            parsed_name=ctx.base,
            parsed_duration=ctx.duration,
            parsed_duration_unit=ctx.duration_unit,
            regime_type=regime_type,
            stages=stages,
            parts=sub_results,
            note=f"{regime_type} regime: {len(stages)} stages",
        )


class ComboResolver(BaseResolver):
    """Combination therapy and cocktail splitting resolver."""

    name = "ComboResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        # 1. Predefined immune cocktails / differentiation recipes prioritized resolution
        tok = ctx.base.strip().lower()
        tok_clean = ctx.cleaned.strip().lower()
        cocktail_meta = ctx.config.cocktails.get(tok) or ctx.config.cocktails.get(tok_clean)
        if not cocktail_meta:
            import re
            tok_no_tail = re.sub(r"\s*(?:cocktail|polarization|differentiation|activation|recipe)\s*$", "", tok).strip()
            cocktail_meta = ctx.config.cocktails.get(tok_no_tail)

        if cocktail_meta:
            parts = cocktail_meta["components"]
            if ctx.pipeline:
                sub = [ctx.pipeline.resolve_single(p, parent_ctx=ctx) for p in parts]
            else:
                sub = []
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_COMBO,
                modality=MODALITY_COMBO,
                parsed_name=cocktail_meta["canonical_name"],
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                note=f"cocktail: {cocktail_meta['canonical_name']} ({cocktail_meta['category']})",
                parts=sub,
            )

        # 2. Syntax delimiter splitting
        base_parts = _pre.split_combo(ctx.base, ctx.config)
        parts = None
        if base_parts is not None:
            # Base name confirmed as combination; if raw text splits into identical number of parts with local doses, prioritize it
            text_parts = _pre.split_combo(ctx.text, ctx.config)
            if text_parts is not None and len(text_parts) == len(base_parts):
                parts = text_parts
            else:
                parts = base_parts
        elif ctx.dose is None:
            parts = _pre.split_underscore(ctx.base)
        if parts is not None:
            if ctx.pipeline:
                sub = [ctx.pipeline.resolve_single(p, parent_ctx=ctx) for p in parts]
            else:
                sub = []
            if ctx.dose is not None and sub:
                # Check whether dose is leading or trailing
                is_leading = bool(ctx.config._dose_leading_re.match(ctx.text))
                if is_leading and sub[0].parsed_dose is None:
                    sub[0].parsed_dose = ctx.dose
                    sub[0].parsed_unit = ctx.unit
                elif not is_leading and sub[-1].parsed_dose is None:
                    sub[-1].parsed_dose = ctx.dose
                    sub[-1].parsed_unit = ctx.unit
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_COMBO,
                modality=MODALITY_COMBO,
                parsed_name=ctx.base,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                note=f"cocktail: {len(sub)} components",
                parts=sub,
            )
class IdentifierResolver(BaseResolver):
    """Authoritative structural identifiers and accession numbers resolver (SMILES, InChIKey, InChI, CAS, CID, ChEMBL, DrugBank, Ensembl, UniProt)."""

    name = "IdentifierResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        from .identifiers import detect_identifier
        ident = (
            detect_identifier(ctx.base)
            or detect_identifier(ctx.cleaned)
            or detect_identifier(str(ctx.raw_input))
        )
        if not ident:
            return None

        kind, val = ident

        # 1. Direct SMILES structure pass-through
        if kind == "smiles":
            smiles_str = str(val)
            entity = DrugEntity(
                input_string=ctx.base,
                canonical_name=smiles_str,
                canonical_smiles=smiles_str,
                confidence=1.0,
                match_path=["identifier:smiles"],
                source="raw_smiles",
            )
            if ctx.cache:
                cached = ctx.cache.get(f"pubchem:smiles:{smiles_str}", vocab_version=None)
                if cached and cached.get("candidates"):
                    c0 = cached["candidates"][0]
                    entity.canonical_name = c0.get("canonical_name") or smiles_str
                    entity.pubchem_cid = c0.get("pubchem_cid")
                    entity.inchikey = c0.get("inchikey")
            if not getattr(entity, "pubchem_cid", None) and ctx.online:
                try:
                    from .annotate import pubchem_lookup_by_smiles
                    hits = pubchem_lookup_by_smiles(smiles_str, cache=ctx.cache)
                    if hits:
                        c0 = hits[0]
                        entity.canonical_name = c0.get("canonical_name") or smiles_str
                        entity.pubchem_cid = c0.get("pubchem_cid")
                        entity.inchikey = c0.get("inchikey")
                        if c0.get("canonical_smiles"):
                            entity.canonical_smiles = c0.get("canonical_smiles")
                        entity.source = "pubchem"
                except Exception:
                    pass
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"identifier_smiles": 1.0},
                matched_by="IdentifierResolver",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_EXACT,
                modality=MODALITY_SMALL_MOLECULE,
                parsed_name=entity.canonical_name,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note="resolved via SMILES identifier",
            )

        # 2. Direct InChI structure pass-through
        if kind == "inchi":
            inchi_str = str(val)
            entity = DrugEntity(
                input_string=ctx.base,
                canonical_name=inchi_str,
                confidence=1.0,
                match_path=["identifier:inchi"],
                source="raw_inchi",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"identifier_inchi": 1.0},
                matched_by="IdentifierResolver",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_EXACT,
                modality=MODALITY_SMALL_MOLECULE,
                parsed_name=inchi_str,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note="resolved via InChI identifier",
            )

        # 3. InChIKey
        if kind == "inchikey":
            ik_str = str(val).upper()
            hits = None
            if ctx.cache:
                cached = ctx.cache.get(f"pubchem:inchikey:{ik_str}", vocab_version=None)
                if cached and cached.get("candidates"):
                    hits = cached["candidates"]
            if not hits and ctx.online:
                try:
                    from .annotate import pubchem_lookup_by_inchikey
                    hits = pubchem_lookup_by_inchikey(ik_str, cache=ctx.cache)
                except Exception:
                    hits = None
            if hits:
                status, ent, cands = build_online_candidates(ik_str, hits)
                enrich_entity(ent, ik_str, ctx.cache, ctx.enrich_rdkit, ctx.enrich_chembl)
                return AlignResult(
                    input=str(ctx.raw_input),
                    status=status,
                    modality=MODALITY_SMALL_MOLECULE,
                    parsed_name=ent.canonical_name if ent else ik_str,
                    parsed_dose=ctx.dose,
                    parsed_unit=ctx.unit,
                    parsed_duration=ctx.duration,
                    parsed_duration_unit=ctx.duration_unit,
                    entity=ent,
                    candidates=cands,
                    sources_resolved=["pubchem:inchikey"],
                    note="resolved via InChIKey lookup",
                )
            entity = DrugEntity(
                input_string=ctx.base,
                canonical_name=ik_str,
                inchikey=ik_str,
                confidence=1.0,
                match_path=["identifier:inchikey"],
                source="raw_inchikey",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"identifier_inchikey": 1.0},
                matched_by="IdentifierResolver",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_EXACT,
                modality=MODALITY_SMALL_MOLECULE,
                parsed_name=ik_str,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note="resolved via InChIKey identifier",
            )

        # 4. PubChem CID
        if kind == "pubchem_cid":
            cid_int = int(val)
            hits = None
            if ctx.cache:
                cached = ctx.cache.get(f"pubchem:cid:{cid_int}", vocab_version=None)
                if cached and cached.get("candidates"):
                    hits = cached["candidates"]
            if not hits and ctx.online:
                try:
                    from .annotate import pubchem_lookup_by_cid
                    hits = pubchem_lookup_by_cid(cid_int, cache=ctx.cache)
                except Exception:
                    hits = None
            if hits:
                status, ent, cands = build_online_candidates(f"CID {cid_int}", hits)
                enrich_entity(ent, str(cid_int), ctx.cache, ctx.enrich_rdkit, ctx.enrich_chembl)
                return AlignResult(
                    input=str(ctx.raw_input),
                    status=status,
                    modality=MODALITY_SMALL_MOLECULE,
                    parsed_name=ent.canonical_name if ent else f"CID {cid_int}",
                    parsed_dose=ctx.dose,
                    parsed_unit=ctx.unit,
                    parsed_duration=ctx.duration,
                    parsed_duration_unit=ctx.duration_unit,
                    entity=ent,
                    candidates=cands,
                    sources_resolved=["pubchem:cid"],
                )
            entity = DrugEntity(
                input_string=ctx.base,
                canonical_name=f"CID {cid_int}",
                pubchem_cid=cid_int,
                confidence=1.0,
                match_path=["identifier:pubchem_cid"],
                source="raw_cid",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"identifier_cid": 1.0},
                matched_by="IdentifierResolver",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_EXACT,
                modality=MODALITY_SMALL_MOLECULE,
                parsed_name=f"CID {cid_int}",
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note="resolved via PubChem CID identifier",
            )

        # 5. ChEMBL ID
        if kind == "chembl_id":
            chembl_id_str = str(val).upper()
            hits = None
            if ctx.cache:
                cached = ctx.cache.get(f"chembl:id:{chembl_id_str.lower()}", vocab_version=None)
                if cached and cached.get("candidates"):
                    hits = cached["candidates"]
            if not hits and ctx.online:
                try:
                    from .annotate import chembl_lookup_by_id
                    hits = chembl_lookup_by_id(chembl_id_str, cache=ctx.cache)
                except Exception:
                    hits = None
            if hits:
                status, ent, cands = build_online_candidates(chembl_id_str, hits)
                enrich_entity(ent, chembl_id_str, ctx.cache, ctx.enrich_rdkit, ctx.enrich_chembl)
                return AlignResult(
                    input=str(ctx.raw_input),
                    status=status,
                    modality=MODALITY_SMALL_MOLECULE,
                    parsed_name=ent.canonical_name if ent else chembl_id_str,
                    parsed_dose=ctx.dose,
                    parsed_unit=ctx.unit,
                    parsed_duration=ctx.duration,
                    parsed_duration_unit=ctx.duration_unit,
                    entity=ent,
                    candidates=cands,
                    sources_resolved=["chembl:id"],
                )
            entity = DrugEntity(
                input_string=ctx.base,
                canonical_name=chembl_id_str,
                chembl_id=chembl_id_str,
                confidence=1.0,
                match_path=["identifier:chembl_id"],
                source="raw_chembl",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"identifier_chembl": 1.0},
                matched_by="IdentifierResolver",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_EXACT,
                modality=MODALITY_SMALL_MOLECULE,
                parsed_name=chembl_id_str,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note="resolved via ChEMBL identifier",
            )

        # 6. CAS Number
        if kind == "cas":
            cas_str = str(val)
            hits = None
            if ctx.cache:
                cached = (
                    ctx.cache.get(f"cir:{cas_str.lower()}", vocab_version=None)
                    or ctx.cache.get(f"pubchem:name:{cas_str.lower()}", vocab_version=None)
                )
                if cached and cached.get("candidates"):
                    hits = cached["candidates"]
            if not hits and ctx.online:
                try:
                    from .annotate import nci_cir_lookup, pubchem_lookup_multi
                    hits = pubchem_lookup_multi(cas_str, cache=ctx.cache) or nci_cir_lookup(cas_str, cache=ctx.cache)
                except Exception:
                    hits = None
            if hits:
                status, ent, cands = build_online_candidates(cas_str, hits)
                enrich_entity(ent, cas_str, ctx.cache, ctx.enrich_rdkit, ctx.enrich_chembl)
                return AlignResult(
                    input=str(ctx.raw_input),
                    status=status,
                    modality=MODALITY_SMALL_MOLECULE,
                    parsed_name=ent.canonical_name if ent else cas_str,
                    parsed_dose=ctx.dose,
                    parsed_unit=ctx.unit,
                    parsed_duration=ctx.duration,
                    parsed_duration_unit=ctx.duration_unit,
                    entity=ent,
                    candidates=cands,
                    sources_resolved=["cas"],
                )
            entity = DrugEntity(
                input_string=ctx.base,
                canonical_name=cas_str,
                cas_number=cas_str,
                confidence=1.0,
                match_path=["identifier:cas"],
                source="raw_cas",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"identifier_cas": 1.0},
                matched_by="IdentifierResolver",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_EXACT,
                modality=MODALITY_SMALL_MOLECULE,
                parsed_name=cas_str,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note="resolved via CAS number",
            )

        # 7. DrugBank ID
        if kind == "drugbank_id":
            db_id = str(val).upper()
            entity = DrugEntity(
                input_string=ctx.base,
                canonical_name=db_id,
                drugbank_id=db_id,
                confidence=1.0,
                match_path=["identifier:drugbank"],
                source="raw_drugbank",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"identifier_drugbank": 1.0},
                matched_by="IdentifierResolver",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_EXACT,
                modality=MODALITY_SMALL_MOLECULE,
                parsed_name=db_id,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note="resolved via DrugBank identifier",
            )

        # 8. Ensembl Gene ID
        if kind == "ensembl_id":
            ens_id = str(val).upper()
            entity = GeneticEntity(
                input_string=ctx.base,
                canonical_name=ens_id,
                gene_symbol=ens_id,
                ensembl_id=ens_id,
                action="perturbation",
                confidence=1.0,
                match_path=["identifier:ensembl"],
                source="raw_ensembl",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"identifier_ensembl": 1.0},
                matched_by="IdentifierResolver",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_EXACT,
                modality=MODALITY_GENETIC,
                parsed_name=ens_id,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                genetic_target=ens_id,
                genetic_action="perturbation",
                entity=entity,
                candidates=[cand],
                note="resolved via Ensembl Gene identifier",
            )

        # 9. UniProt Accession
        if kind == "uniprot_accession":
            uni_id = str(val).upper()
            entity = BiologicEntity(
                input_string=ctx.base,
                canonical_name=uni_id,
                category="protein",
                hgnc_or_uniprot=uni_id,
                confidence=1.0,
                match_path=["identifier:uniprot"],
                source="raw_uniprot",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"identifier_uniprot": 1.0},
                matched_by="IdentifierResolver",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_EXACT,
                modality=MODALITY_BIOLOGIC,
                parsed_name=uni_id,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note="resolved via UniProt Accession identifier",
            )

        return None


class GeneticResolver(BaseResolver):
    """Genetic and synthetic biology perturbation resolver (CRISPR KO/CRISPRi/CRISPRa/prime editing/base editing/epigenetic editing/optogenetics/chemogenetics)."""

    name = "GeneticResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        # 1. CRISPR sgRNA sequence alignment reverse lookup
        try:
            from .adapter import lookup_sgrna
            sgrna_entity = lookup_sgrna(ctx.base, ctx.config)
            if sgrna_entity is not None:
                cand = Candidate(
                    entity=sgrna_entity,
                    score=1.0,
                    score_breakdown={"sgrna_reference": 1.0},
                    matched_by="GeneticResolver",
                    matched_surface=ctx.base,
                )
                is_ctrl = sgrna_entity.action == "control"
                return AlignResult(
                    input=str(ctx.raw_input),
                    status=STATUS_CONTROL if is_ctrl else STATUS_UNMATCHED,
                    is_control=is_ctrl,
                    modality=MODALITY_CONTROL if is_ctrl else MODALITY_GENETIC,
                    parsed_name=sgrna_entity.gene_symbol,
                    parsed_dose=ctx.dose,
                    parsed_unit=ctx.unit,
                    parsed_duration=ctx.duration,
                    parsed_duration_unit=ctx.duration_unit,
                    genetic_target=sgrna_entity.gene_symbol,
                    genetic_action=sgrna_entity.action,
                    entity=sgrna_entity,
                    candidates=[cand],
                    note=f"CRISPR sgRNA sequence matching {sgrna_entity.gene_symbol}",
                )
        except Exception:
            pass

        # 2. Pattern and syntactic rule resolution
        gen_info = _pre.detect_genetic_perturbation(ctx.base, ctx.config)
        if gen_info is not None:
            target, action = gen_info
            system_type = "crispr_ko"
            effector = None
            if action.startswith("epigenetic_edit:"):
                system_type = "epigenome_edit"
                effector = action.split(":", 1)[1]
            elif action.startswith("chemogenetics:"):
                system_type = "chemogenetics"
                effector = action.split(":", 1)[1]
            elif action.startswith("optogenetics:"):
                system_type = "optogenetics"
                effector = action.split(":", 1)[1]
            elif action.startswith("editing:"):
                system_type = "crispr_base" if ("ABE" in action or "CBE" in action) else "crispr_prime"
            elif action == "dual_KO":
                system_type = "crispr_dual"
            elif action == "CRISPRi":
                system_type = "crispri"
            elif action == "CRISPRa":
                system_type = "crispra"
            elif action == "KD":
                system_type = "rnai"
            elif action == "OE":
                system_type = "overexpression"

            entity = GeneticEntity(
                input_string=ctx.base,
                canonical_name=target,
                gene_symbol=target,
                action=action,
                system_type=system_type,
                effector=effector,
                confidence=1.0,
                match_path=[f"genetic:{action.lower()}"],
                source="genetic_rules",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"genetic_rule": 1.0},
                matched_by="GeneticResolver",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_UNMATCHED,
                modality=MODALITY_GENETIC,
                parsed_name=ctx.base,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                genetic_target=target,
                genetic_action=action,
                entity=entity,
                candidates=[cand],
                note=f"genetic perturbation: {action} targeting {target}",
            )
        return None


class BiologicResolver(BaseResolver):
    """Biologics resolver (cytokines, growth factors, monoclonal antibodies, etc.)."""

    name = "BiologicResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        bio_info = _pre.detect_biologic(ctx.base, ctx.config)
        if bio_info is not None:
            c_name = bio_info.get("canonical_name", ctx.base)
            cat = bio_info.get("category", "biologic")
            xid = bio_info.get("hgnc_or_uniprot")
            entity = BiologicEntity(
                input_string=ctx.base,
                canonical_name=c_name,
                category=cat,
                hgnc_or_uniprot=xid,
                confidence=1.0,
                match_path=["vocab:biologic"],
                source="biologics",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"biologics_vocab": 1.0},
                matched_by="BiologicsMatcher",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_EXACT,
                modality=MODALITY_BIOLOGIC,
                parsed_name=c_name,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note=f"biologic entity ({cat})",
            )
        return None


class EnvironmentalResolver(BaseResolver):
    """Physical and microenvironmental stress resolver (hypoxia, starvation, heat shock, radiation, etc.)."""

    name = "EnvironmentalResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        env_info = _pre.detect_environmental(ctx.base, ctx.config)
        if env_info is not None:
            cat = env_info.get("category", "environmental")
            # Preserve classic small molecule chemical inducers (e.g., erastin, tunicamycin, thapsigargin, rotenone, rsl3):
            # If categorized as cellular_stress but user input does not explicitly contain stress modifier terms (e.g. stress, induced, ferroptosis),
            # pass through to downstream small molecule resolvers to acquire real chemical structures (SMILES, CID, InChIKey);
            # intercept and assign environmental modality only when stress modifiers are explicitly declared.
            if cat == "cellular_stress":
                stress_keywords = ("stress", "induced", "induction", "ferroptosis", "damage", "shock", "granule", "inhibition", "depletion")
                raw_lower = ctx.raw_input.strip().lower() if isinstance(ctx.raw_input, str) else ctx.base.lower()
                if not any(k in raw_lower for k in stress_keywords):
                    return None

            c_name = env_info.get("canonical_name", ctx.base)
            param = env_info.get("parameter")
            note_str = (
                f"environmental perturbation ({cat}: {param})"
                if param
                else f"environmental perturbation ({cat})"
            )
            entity = EnvironmentalEntity(
                input_string=ctx.base,
                canonical_name=c_name,
                category=cat,
                parameter=param,
                confidence=1.0,
                match_path=["vocab:environmental"],
                source="environmental",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"environmental_vocab": 1.0},
                matched_by="EnvironmentalMatcher",
                matched_surface=ctx.base,
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_UNMATCHED,
                modality=MODALITY_ENVIRONMENTAL,
                parsed_name=c_name,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note=note_str,
            )
        return None


class DeliveryVehicleResolver(BaseResolver):
    """Delivery vehicles and nanodelivery systems resolver (LNP, AAV, Exosome, Lentivirus, etc.)."""

    name = "DeliveryVehicleResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        veh_info = _pre.detect_delivery_vehicle(ctx.base, ctx.config) or _pre.detect_delivery_vehicle(ctx.cleaned, ctx.config)
        if veh_info is not None:
            c_name = veh_info.get("canonical_name", ctx.base)
            c_type = veh_info.get("carrier_type", "lnp")
            serotype = veh_info.get("serotype")
            formulation = veh_info.get("formulation", [])
            tropism = veh_info.get("tropism")
            cargo = veh_info.get("cargo", [])

            entity = DeliveryVehicleEntity(
                input_string=ctx.base,
                canonical_name=c_name,
                carrier_type=c_type,
                serotype=serotype,
                formulation=formulation,
                tropism=tropism,
                cargo=cargo,
                confidence=1.0,
                match_path=["delivery:rules"],
                source="delivery_vehicles",
            )
            cand = Candidate(
                entity=entity,
                score=1.0,
                score_breakdown={"delivery_vocab": 1.0},
                matched_by="DeliveryVehicleResolver",
                matched_surface=ctx.base,
            )
            note_parts = [f"delivery system ({c_type}"]
            if serotype:
                note_parts.append(f"serotype: {serotype}")
            if cargo:
                note_parts.append(f"cargo: {', '.join(cargo)}")
            note_str = ", ".join(note_parts) + ")"

            return AlignResult(
                input=str(ctx.raw_input),
                status=STATUS_EXACT,
                modality=MODALITY_DELIVERY,
                parsed_name=c_name,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                entity=entity,
                candidates=[cand],
                note=note_str,
            )
        return None


class UserMappingResolver(BaseResolver):
    """User-defined mapping resolver (highest local override priority)."""

    name = "UserMappingResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        if not ctx.user_mapping:
            return None
        mapped = (
            ctx.user_mapping.get(ctx.base)
            or ctx.user_mapping.get(ctx.cleaned)
            or ctx.user_mapping.get(str(ctx.raw_input))
        )
        if mapped is None:
            return None

        if isinstance(mapped, str):
            entity = DrugEntity(
                input_string=ctx.cleaned or ctx.base,
                canonical_name=mapped,
                confidence=1.0,
                match_path=["user_mapping"],
            )
        elif isinstance(mapped, PerturbationEntity):
            entity = mapped
        else:
            entity = DrugEntity(
                input_string=ctx.cleaned or ctx.base,
                canonical_name=str(mapped),
                confidence=1.0,
                match_path=["user_mapping"],
            )
        cand = Candidate(
            entity=entity,
            score=1.0,
            score_breakdown={"user_mapping": 1.0},
            matched_by="user_mapping",
            matched_surface=ctx.cleaned or ctx.base,
        )
        return AlignResult(
            input=str(ctx.raw_input),
            status=STATUS_EXACT,
            parsed_name=ctx.cleaned,
            parsed_dose=ctx.dose,
            parsed_unit=ctx.unit,
            parsed_duration=ctx.duration,
            parsed_duration_unit=ctx.duration_unit,
            modality=MODALITY_SMALL_MOLECULE,
            entity=entity,
            candidates=[cand],
            note="matched via user_mapping",
        )


class CachedCorpusResolver(BaseResolver):
    """Local SQLite / in-memory cached corpus resolver (exact, fuzzy, and ambiguous matching)."""

    name = "CachedCorpusResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        if not ctx.cleaned:
            return None
        status, entity, candidates = _norm.resolve_cached(
            ctx.cleaned,
            ctx.cache,
            fuzzy=ctx.fuzzy,
            fuzzy_threshold=ctx.fuzzy_threshold,
        )
        if status != STATUS_UNMATCHED:
            if entity is not None:
                entity.input_string = ctx.cleaned
                enrich_entity(entity, ctx.cleaned, ctx.cache, ctx.enrich_rdkit, ctx.enrich_chembl)
            note = (
                f"ambiguous: {len(candidates)} candidates, auto-resolution requires threshold"
                if status == STATUS_AMBIGUOUS
                else None
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=status,
                parsed_name=ctx.cleaned,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                modality=MODALITY_SMALL_MOLECULE,
                entity=entity,
                candidates=candidates,
                note=note,
            )
        return None


class OnlineCascadeResolver(BaseResolver):
    """Online API cascade resolver (PubChem / ChEMBL cascade query and negative cache detection)."""

    name = "OnlineCascadeResolver"

    def resolve(self, ctx: AlignmentContext) -> AlignResult | None:
        if not ctx.online or not ctx.cleaned:
            return None

        net_fail = False
        sources_resolved = []
        hits = []
        try:
            from .annotate import resolve_online_cascade
            hits, sources = resolve_online_cascade(ctx.cleaned, cache=ctx.cache)
            sources_resolved = sources
        except Exception:  # noqa: BLE001
            hits = []

        if not hits and ctx.cache is not None:
            cached = ctx.cache.get(
                f"pubchem:name:{ctx.cleaned.strip().lower()}",
                vocab_version=None,
            )
            net_fail = bool(cached and cached.get("unreachable"))

        if hits:
            status, entity, candidates = build_online_candidates(ctx.cleaned, hits)
            enrich_entity(entity, ctx.cleaned, ctx.cache, ctx.enrich_rdkit, ctx.enrich_chembl)
            note = (
                f"ambiguous: {len(candidates)} candidates, auto-resolution requires threshold"
                if status == STATUS_AMBIGUOUS
                else None
            )
            return AlignResult(
                input=str(ctx.raw_input),
                status=status,
                parsed_name=ctx.cleaned,
                parsed_dose=ctx.dose,
                parsed_unit=ctx.unit,
                parsed_duration=ctx.duration,
                parsed_duration_unit=ctx.duration_unit,
                modality=MODALITY_SMALL_MOLECULE,
                sources_resolved=sources_resolved,
                entity=entity,
                candidates=candidates,
                note=note,
            )

        ctx.options["online_net_fail"] = net_fail
        return None


class AlignmentPipeline:
    """Perturbation alignment pipeline, maintaining and executing the resolver chain."""

    def __init__(self, resolvers: Sequence[BaseResolver] | None = None) -> None:
        self.resolvers: list[BaseResolver] = (
            list(resolvers) if resolvers is not None else self.default_resolvers()
        )

    @classmethod
    def default_resolvers(cls) -> list[BaseResolver]:
        """Default built-in resolver chain."""
        return [
            EmptyInputResolver(),
            ControlResolver(),
            SequentialRegimeResolver(),
            ComboResolver(),
            UserMappingResolver(),
            IdentifierResolver(),
            GeneticResolver(),
            BiologicResolver(),
            EnvironmentalResolver(),
            DeliveryVehicleResolver(),
            CachedCorpusResolver(),
            OnlineCascadeResolver(),
        ]

    def add_resolver(
        self,
        resolver: BaseResolver,
        priority: int | None = None,
        before: str | None = None,
        after: str | None = None,
    ) -> None:
        """Dynamically register a resolver. Supports index insertion, or relative insertion before/after a named resolver."""
        if before:
            for idx, r in enumerate(self.resolvers):
                if r.name == before or r.__class__.__name__ == before:
                    self.resolvers.insert(idx, resolver)
                    return
        if after:
            for idx, r in enumerate(self.resolvers):
                if r.name == after or r.__class__.__name__ == after:
                    self.resolvers.insert(idx + 1, resolver)
                    return
        if priority is not None:
            self.resolvers.insert(priority, resolver)
        else:
            self.resolvers.append(resolver)

    def remove_resolver(self, name_or_class: str | type) -> bool:
        """Remove a resolver by name or class type."""
        for idx, r in enumerate(self.resolvers):
            if (
                isinstance(name_or_class, str)
                and (r.name == name_or_class or r.__class__.__name__ == name_or_class)
            ) or (isinstance(name_or_class, type) and isinstance(r, name_or_class)):
                del self.resolvers[idx]
                return True
        return False

    def __len__(self) -> int:
        return len(self.resolvers)

    def __iter__(self):
        return iter(self.resolvers)

    def resolve_single(
        self,
        value: Any,
        parent_ctx: AlignmentContext | None = None,
        config: _pre.PreprocessConfig | None = None,
        online: bool = False,
        cache: BaseCache | None = None,
        fuzzy: bool = False,
        fuzzy_threshold: int = 80,
        enrich_chembl: bool = False,
        enrich_rdkit: bool = False,
        user_mapping: dict[str, str | PerturbationEntity] | None = None,
        options: dict[str, Any] | None = None,
    ) -> AlignResult:
        """Resolve a single perturbation input through the resolver chain."""
        if parent_ctx is not None:
            ctx = AlignmentContext.create(
                raw_input=value,
                config=parent_ctx.config,
                cache=parent_ctx.cache,
                online=parent_ctx.online,
                fuzzy=parent_ctx.fuzzy,
                fuzzy_threshold=parent_ctx.fuzzy_threshold,
                enrich_chembl=parent_ctx.enrich_chembl,
                enrich_rdkit=parent_ctx.enrich_rdkit,
                user_mapping=parent_ctx.user_mapping,
                options=options or dict(parent_ctx.options),
                pipeline=self,
            )
        else:
            ctx = AlignmentContext.create(
                raw_input=value,
                config=config if config is not None else _pre.DEFAULT_CONFIG,
                cache=cache,
                online=online,
                fuzzy=fuzzy,
                fuzzy_threshold=fuzzy_threshold,
                enrich_chembl=enrich_chembl,
                enrich_rdkit=enrich_rdkit,
                user_mapping=user_mapping,
                options=options or {},
                pipeline=self,
            )

        for resolver in self.resolvers:
            res = resolver.resolve(ctx)
            if res is not None:
                return res

        # Fallback to UNMATCHED
        net_fail = ctx.options.get("online_net_fail", False)
        note = (
            "API unreachable (network / rate limit), negative-cached 1h"
            if net_fail
            else "not found in PubChem / ChEMBL (or offline with empty cache)"
        )
        return AlignResult(
            input=str(value),
            status=STATUS_UNMATCHED,
            parsed_name=ctx.cleaned,
            parsed_dose=ctx.dose,
            parsed_unit=ctx.unit,
            parsed_duration=ctx.duration,
            parsed_duration_unit=ctx.duration_unit,
            modality=MODALITY_SMALL_MOLECULE,
            sources_resolved=[],
            entity=None,
            candidates=[],
            note=note,
        )
