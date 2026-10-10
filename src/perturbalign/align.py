"""perturbalign.align — Core entrypoint (Pure alignment engine, data sourced from APIs).

Single-line injection for AnnData / Pandas DataFrame / generic sequences:

    mapping = perturbalign.align(adata.obs["perturbation"])
    df = perturbalign.to_dataframe(mapping)

The package focuses exclusively on **alignment logic**: name cleaning / dose parsing /
control detection / combination therapy splitting / disambiguation.
Molecular data (structures / identifiers / targets) **defaults entirely to online APIs**
(PubChem name -> structure, ChEMBL enrichment optional), with local SQLite cache as the
persistence layer (cache-as-corpus: historical queries enable offline exact hits and fuzzy matching).
No bundled molecular databases; zero assumptions about user data formats.

Tiered offline behavior:
- Logic layer (cleaning / controls / combinations / dosing) is always available.
- Structure layer is empty when offline, but backfilled whenever cache hits occur.

Returns list[AlignResult] (dataclass containing candidate list and confidence scores);
optionally writes back to obs (annotate_anndata + audit).
"""
from __future__ import annotations

import copy

from . import normalize as _norm
from . import preprocess as _pre
from .pipeline import (
    AlignmentPipeline,
    BaseResolver,
    build_online_candidates,
    enrich_entity,
)
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
    AlignResult,
    Candidate,
    DrugEntity,
    PerturbationEntity,
)


def _val_key(v):
    """Extract a hashable key suitable for deduplication and bucketing (compatible with None / NaN / pd.NA / scalars)."""
    if v is None:
        return (None, None)
    try:
        import pandas as pd  # type: ignore
        if pd.isna(v):
            return (float("nan"), "__NA__")
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        hash(v)
        return v
    except TypeError:
        return str(v)


_enrich_entity = enrich_entity
_build_online_candidates = build_online_candidates
_DEFAULT_PIPELINE = AlignmentPipeline()


def get_default_pipeline() -> AlignmentPipeline:
    """Return the global default AlignmentPipeline instance."""
    return _DEFAULT_PIPELINE


def reset_default_pipeline() -> AlignmentPipeline:
    """Reset the global default AlignmentPipeline instance to factory default resolvers."""
    global _DEFAULT_PIPELINE
    _DEFAULT_PIPELINE = AlignmentPipeline()
    return _DEFAULT_PIPELINE


def _align_single(
    value,
    config: _pre.PreprocessConfig,
    online: bool,
    cache,
    fuzzy: bool = False,
    fuzzy_threshold: int = 80,
    enrich_chembl: bool = False,
    enrich_rdkit: bool = False,
    user_mapping: dict[str, str | PerturbationEntity] | None = None,
    pipeline: AlignmentPipeline | None = None,
) -> AlignResult:
    """Internal single-item alignment entrypoint (delegated to AlignmentPipeline resolver chain)."""
    p = pipeline or _DEFAULT_PIPELINE
    return p.resolve_single(
        value=value,
        config=config,
        online=online,
        cache=cache,
        fuzzy=fuzzy,
        fuzzy_threshold=fuzzy_threshold,
        enrich_chembl=enrich_chembl,
        enrich_rdkit=enrich_rdkit,
        user_mapping=user_mapping,
    )


def align(
    values,
    online: bool = True,
    extra_control_tokens: list[str] | None = None,
    config: _pre.PreprocessConfig | None = None,
    cache=None,
    fuzzy: bool = False,
    fuzzy_threshold: int = 80,
    enrich_chembl: bool = False,
    enrich_rdkit: bool = False,
    user_mapping: dict[str, str | PerturbationEntity] | None = None,
    extra_resolvers: list[BaseResolver] | None = None,
    pipeline: AlignmentPipeline | None = None,
) -> list[AlignResult]:
    """Standardize and align a collection of perturbation labels.

    The package only performs alignment logic; molecular data (structures/identifiers)
    **defaults entirely to online APIs** (PubChem name -> structure + SQLite local cache;
    ChEMBL enrichment optional).

    values: str / list / tuple / pandas.Series / pandas.DataFrame (processed element-wise).
    online: Whether to enable online API queries (default True; False = pure offline, logic layer + cached corpus only).
    fuzzy: Whether to enable cached corpus fuzzy matching (rapidfuzz optional, or falls back to difflib).
    fuzzy_threshold: Fuzzy match score threshold (0-100, default 80).
    enrich_chembl: Query ChEMBL by name on hit to populate chembl_id / targets / moa (online + cache).
    enrich_rdkit: Compute parent InChIKey via RDKit on hit (parent_inchikey, optional dependency).
    extra_control_tokens: Append extra control tokens (without modifying resource files).
    config: Custom PreprocessConfig (allows replacing vocabulary/regex resource paths).
    cache: Custom Cache instance (defaults to SQLiteCache singleton; must implement get/set/iter_names).
    user_mapping: User custom override dictionary (e.g., {"ABC-001": "Doxorubicin"}), highest priority.
    extra_resolvers: Custom resolver list, inserted before cache and online resolvers by default.
    pipeline: Fully custom AlignmentPipeline instance (overrides default pipeline if provided).
    """
    if values is None:
        values = [None]  # Empty input (_align_single -> empty)
    elif isinstance(values, str):
        values = [values]
    # pandas.Series / single-column DataFrame compatibility (explicit error for multi-column DataFrames)
    if hasattr(values, "columns"):
        if len(values.columns) != 1:
            raise ValueError(
                "align() received a DataFrame with multiple columns; please pass adata.obs['perturbation'] or a single-column DataFrame"
            )
        values = values.iloc[:, 0]
    if hasattr(values, "tolist"):
        raw_list = list(values.tolist())
    elif not isinstance(values, (list, tuple)):
        raw_list = list(values)
    else:
        raw_list = list(values)

    if not raw_list:
        return []

    if config is None:
        if extra_control_tokens:
            config = _pre.DEFAULT_CONFIG.with_extra_controls(extra_control_tokens)
        else:
            config = _pre.DEFAULT_CONFIG
    if cache is None:
        try:
            from .cache import get_default_cache
            cache = get_default_cache()
        except Exception:  # noqa: BLE001 — Cache failure does not block the main pipeline
            cache = None

    if pipeline is None:
        if extra_resolvers:
            pipeline = AlignmentPipeline()
            for r in reversed(extra_resolvers):
                pipeline.add_resolver(r, before="CachedCorpusResolver")
        else:
            pipeline = _DEFAULT_PIPELINE

    # 1) Unique value deduplication mapping (optimization for millions of single cells: compute unique labels only)
    seen: dict[object, tuple[object, list[int]]] = {}
    for idx, v in enumerate(raw_list):
        k = _val_key(v)
        if k not in seen:
            seen[k] = (v, [])
        seen[k][1].append(idx)

    # 2) Run offline/local layer on unique labels first (controls, combos, historical cache, user dict, custom resolvers)
    aligned_unique: dict[object, AlignResult] = {}
    for k, (v, _) in seen.items():
        aligned_unique[k] = pipeline.resolve_single(
            v,
            config=config,
            online=False,
            cache=cache,
            fuzzy=fuzzy,
            fuzzy_threshold=fuzzy_threshold,
            enrich_chembl=enrich_chembl,
            enrich_rdkit=enrich_rdkit,
            user_mapping=user_mapping,
        )

    # 3) If online is enabled, collect unmatched drug names needing API queries, batch prefetch, and cascade-resolve raw identifiers
    if online:
        def _needs_online(r: AlignResult) -> bool:
            if r.modality != MODALITY_SMALL_MOLECULE:
                return False
            if r.status == STATUS_UNMATCHED and r.parsed_name:
                return True
            if r.entity and r.entity.source and r.entity.source.startswith("raw_"):
                return True
            return False

        needed_names: set[str] = set()
        for res in aligned_unique.values():
            if res.status == STATUS_COMBO:
                for p in res.parts:
                    if _needs_online(p) and not (p.entity and p.entity.source and p.entity.source.startswith("raw_")):
                        if p.parsed_name:
                            needed_names.add(p.parsed_name)
            elif _needs_online(res):
                if not (res.entity and res.entity.source and res.entity.source.startswith("raw_")):
                    if res.parsed_name:
                        needed_names.add(res.parsed_name)

        batch_results: dict[str, list[dict] | None] = {}
        if needed_names:
            names_list = sorted(needed_names)
            if len(names_list) > 1:
                try:
                    from .annotate import pubchem_lookup_batch
                    batch_results = pubchem_lookup_batch(names_list, cache=cache) or {}
                except Exception:  # noqa: BLE001
                    pass
            elif len(names_list) == 1:
                try:
                    from .annotate import pubchem_lookup_multi
                    n = names_list[0]
                    h = pubchem_lookup_multi(n, cache=cache)
                    batch_results = {n: h}
                except Exception:  # noqa: BLE001
                    pass

        # Backfill online resolved entities (including uncataloged generic drugs and raw_* authoritative identifier cascading)
        def _resolve_item_online(item: AlignResult):
            if not _needs_online(item):
                return
            is_raw = bool(item.entity and item.entity.source and item.entity.source.startswith("raw_"))
            hits = None
            sources = []
            query_name = item.input if is_raw else (item.parsed_name or item.input)

            if not is_raw and item.parsed_name and item.parsed_name in batch_results:
                hits = batch_results[item.parsed_name]
                if hits:
                    sources = ["pubchem"]

            if not hits:
                try:
                    from .annotate import resolve_online_cascade
                    hits, sources = resolve_online_cascade(query_name, cache=cache)
                except Exception:  # noqa: BLE001
                    hits, sources = None, []

            if hits:
                st, ent, cands = _build_online_candidates(query_name, hits)
                _enrich_entity(ent, query_name, cache, enrich_rdkit, enrich_chembl)
                item.status = st
                item.entity = ent
                item.candidates = cands
                item.sources_resolved = sources
                item.note = None if st == STATUS_ONLINE else f"ambiguous: {len(cands)} candidates"
            else:
                if not is_raw:
                    # Diagnostic note for confirmed unindexed or network failure
                    net_fail = False
                    if cache is not None:
                        cached = cache.get(f"pubchem:name:{query_name.strip().lower()}", vocab_version=None)
                        net_fail = bool(cached and cached.get("unreachable"))
                    item.note = ("API unreachable (network / rate limit), negative-cached 1h" if net_fail
                                else "not found in PubChem / ChEMBL (or offline with empty cache)")

        for res in aligned_unique.values():
            if res.status == STATUS_COMBO:
                for p in res.parts:
                    _resolve_item_online(p)
            else:
                _resolve_item_online(res)

    # 4) Reconstruct and map back to original sequence length (identical inputs share instances to save memory)
    results: list[AlignResult] = [None] * len(raw_list)  # type: ignore
    for k, (v, indices) in seen.items():
        res = aligned_unique[k]
        for idx in indices:
            orig_val = raw_list[idx]
            if str(orig_val) == res.input:
                results[idx] = res
            else:
                r_copy = copy.copy(res)
                r_copy.input = str(orig_val)
                results[idx] = r_copy

    return results


def to_dataframe(mapping: list[AlignResult]):
    """Convert a list of AlignResult objects to a pandas DataFrame (pandas is an optional dependency)."""
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("to_dataframe requires pandas, please run: pip install pandas") from exc

    rows = []
    for rec in mapping:
        ent = rec.entity
        row = {
            "input": rec.input,
            "status": str(rec.status),
            "modality": str(rec.modality),
            "regime_type": rec.regime_type,
            "is_control": rec.is_control,
            "drug_name": ent.canonical_name if ent else None,
            "canonical_smiles": getattr(ent, "canonical_smiles", None) if ent else None,
            "inchikey": getattr(ent, "inchikey", None) if ent else None,
            "pubchem_cid": getattr(ent, "pubchem_cid", None) if ent else None,
            "targets": "; ".join(getattr(ent, "targets", [])) if (ent and getattr(ent, "targets", None)) else None,
            "moa": getattr(ent, "moa", None) if ent else None,
            "confidence": ent.confidence if ent else None,
            "match_path": ";".join(ent.match_path) if ent else None,
            "source": ent.source if ent else None,
            "parsed_dose": rec.parsed_dose,
            "parsed_unit": rec.parsed_unit,
            "parsed_duration": rec.parsed_duration,
            "parsed_duration_unit": rec.parsed_duration_unit,
            "genetic_target": rec.genetic_target,
            "genetic_action": rec.genetic_action,
            "sources_resolved": ";".join(rec.sources_resolved) if rec.sources_resolved else None,
            "parsed_name": rec.parsed_name,
            "n_candidates": len(rec.candidates),
            "note": rec.note,
            "parts": "; ".join(
                f"{p.parsed_name}={p.entity.canonical_name if p.entity else p.status}"
                for p in rec.parts
            ) if rec.parts else None,
            "combo_names": "; ".join(rec.combo_names) if rec.parts else None,
            "combo_smiles": "; ".join(s if s else "" for s in rec.combo_smiles) if (rec.parts and any(rec.combo_smiles)) else None,
            "combo_doses": ", ".join(str(d) if d is not None else "" for d in rec.combo_doses) if (rec.parts and any(d is not None for d in rec.combo_doses)) else None,
            "combo_units": ", ".join(u if u else "" for u in rec.combo_units) if (rec.parts and any(rec.combo_units)) else None,
            "combo_targets": "; ".join(rec.combo_targets) if (rec.parts and rec.combo_targets) else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)

