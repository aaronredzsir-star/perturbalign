"""Module 2: Normalization Engine (cached corpus + confidence scoring).

All primary data originates from upstream APIs (PubChem/ChEMBL), with the local cache as the persistence layer.
Cache acts as the corpus: previously queried name -> entity mappings support exact hit and offline fuzzy matching.

Matching cascade (when fuzzy=False, only exact matching is executed):
  1) Cache exact hit (case-insensitive normalized key) -> exact
  2) Cache entry contains multiple candidates -> rank by confidence, return ambiguous + candidates if unresolved
  3) fuzzy=True and exact missed -> rapidfuzz error-tolerant matching against cached corpus -> fuzzy
     (falls back gracefully if rapidfuzz is not installed)
"""
from __future__ import annotations

from . import resources
from .schema import (
    AMBIGUOUS_GAP,
    STATUS_AMBIGUOUS,
    STATUS_EXACT,
    STATUS_FUZZY,
    STATUS_UNMATCHED,
    Candidate,
    DrugEntity,
)


def _as_candidates(cached: dict, query: str) -> list[Candidate]:
    """Normalize cached entry (new format {"candidates": [...]} or legacy single dict) to candidate list.

    Candidate confidence: higher ranking in PubChem order has higher confidence (first candidate 0.7, remainder 0.4).
    """
    items = cached.get("candidates") if isinstance(cached.get("candidates"), list) else [cached]
    out = []
    for i, hit in enumerate(items):
        entity = DrugEntity(
            input_string=query,
            canonical_name=hit.get("canonical_name") or query,
            canonical_smiles=hit.get("canonical_smiles"),
            inchikey=hit.get("inchikey"),
            pubchem_cid=hit.get("pubchem_cid"),
            source="pubchem",
            confidence=0.7 if i == 0 else 0.4,
            match_path=["exact"],
            vocabulary_version=resources.VOCABULARY_VERSION,
        )
        out.append(Candidate(
            entity=entity,
            score=entity.confidence,
            score_breakdown={"pubchem_order": entity.confidence},
            matched_by="exact cache hit",
            matched_surface=hit.get("matched_surface", query),
        ))
    return out


def resolve_cached(name: str, cache, fuzzy: bool = False,
                   fuzzy_threshold: int = 80, fuzzy_limit: int = 5,
                   ) -> tuple[str, DrugEntity | None, list[Candidate]]:
    """Resolve a single (cleaned) drug name against the cached corpus.

    Returns (status, entity, candidates).
    - Cache miss -> unmatched
    - 1 candidate -> exact
    - Multiple candidates and score gap < AMBIGUOUS_GAP -> ambiguous (lists all candidates without silent guessing)
    - Multiple candidates with sufficient gap -> pick top1
    - fuzzy=True and exact missed -> WRatio error-tolerant matching against cached corpus -> fuzzy
    """
    if cache is None:
        return STATUS_UNMATCHED, None, []

    cached = cache.get(f"pubchem:name:{name.strip().lower()}",
                       vocab_version=resources.VOCABULARY_VERSION)
    if cached is not None and not cached.get("unreachable") and not cached.get("empty"):
        candidates = _as_candidates(cached, name)
        if len(candidates) == 1:
            ent = candidates[0].entity
            ent.input_string = name
            return STATUS_EXACT, ent, candidates
        # Multiple CIDs for the same query = ambiguous name (avoid silent guessing)
        return STATUS_AMBIGUOUS, None, candidates

    if not fuzzy:
        return STATUS_UNMATCHED, None, []

    # Level-2 fuzzy matching: cached corpus (prefers rapidfuzz, falls back to standard difflib)
    try:
        from rapidfuzz import fuzz as _fuzz  # type: ignore
        def _similarity(s1: str, s2: str) -> float:
            return float(_fuzz.WRatio(s1, s2))
    except ImportError:
        from difflib import SequenceMatcher
        def _similarity(s1: str, s2: str) -> float:
            return SequenceMatcher(None, s1, s2).ratio() * 100.0

    hits = []
    for key_text, result in cache.iter_names("pubchem:name:",
                                             vocab_version=resources.VOCABULARY_VERSION):
        q = name.strip().lower()
        key = key_text[len("pubchem:name:"):]
        wratio = _similarity(q, key)
        if wratio < fuzzy_threshold:  # Threshold on 0-100 scale
            continue

        score = wratio / 100 * 0.75   # Normalized confidence, capped at 0.75 (below exact tier)
        for cand in _as_candidates(result, name):
            hits.append((score, key, cand))
    if not hits:
        return STATUS_UNMATCHED, None, []
    hits.sort(key=lambda t: t[0], reverse=True)
    dedup: dict[str, tuple[float, Candidate]] = {}
    for score, key, cand in hits:
        ik = cand.entity.inchikey or cand.entity.canonical_name
        if ik not in dedup or score > dedup[ik][0]:
            dedup[ik] = (score, cand)
    ordered = sorted(dedup.values(), key=lambda t: t[0], reverse=True)[:fuzzy_limit]
    candidates = [c for _, c in ordered]
    for cand in candidates:
        cand.matched_by = "fuzzy matcher"
    if len(candidates) == 1:
        ent = candidates[0].entity
        ent.input_string = name
        ent.confidence = ordered[0][0]
        ent.match_path = ["fuzzy"]
        return STATUS_FUZZY, ent, candidates
    # Multiple fuzzy candidates with insufficient gap -> ambiguous (avoid silent guessing)
    if ordered[0][0] - ordered[1][0] >= AMBIGUOUS_GAP:
        ent = candidates[0].entity
        ent.input_string = name
        ent.confidence = ordered[0][0]
        ent.match_path = ["fuzzy"]
        return STATUS_FUZZY, ent, candidates
    return STATUS_AMBIGUOUS, None, candidates
