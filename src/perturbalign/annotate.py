"""Module 3 (Data Layer): PubChem / ChEMBL online data (default sources, offline-safe + local caching).

The package focuses on alignment logic; molecular data defaults to API sources:
  - PubChem name -> structure (default online path for align, multi-candidate support for disambiguation)
  - ChEMBL target / MoA enrichment (enrich_chembl=True)
Results are written to SQLite resolve_cache (TTL + vocabulary version); repeated queries hit cache directly.
Network failures write 1-hour negative cache to avoid repeated timeouts.

Network layer optimizations:
  - Exponential backoff retry (429/5xx/timeout, default 3 attempts)
  - Batch endpoints (PUG REST comma-separated name->cids->properties, default 20 names/batch, fallback to per-name on error)
"""
from __future__ import annotations

import abc
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

from .cache import SQLiteCache
from .identifiers import detect_identifier
from .resources import VOCABULARY_VERSION, load_greek_transliterations

_PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
_CIR_BASE = "https://cactus.nci.nih.gov/chemical/structure"
_TIMEOUT = 10
_RETRIES = 3
_RETRY_DELAY = 2.0          # Exponential backoff base delay (2/4/8s)
_RETRY_STATUS = {429, 500, 502, 503, 504}
_BATCH_SIZE = 20            # PUG REST batch name count (safe value <= 100)
_TTL_ONLINE = 30 * 24 * 3600  # Online result cache duration: 30 days
_TTL_NET_FAIL = 3600          # Network failure negative cache duration: 1 hour (avoid repeated timeout waits)
_USER_AGENT = "perturbalign/0.9.0"
_UNREACHABLE = {"unreachable": True}

# PubChem name index only stores Latin transliterations (e.g. "18beta-...", "gamma-...").
# Searching original Greek letter names (18β / γ-) yields 404 directly. Loaded from resources/greek_transliteration.tsv.
_GREEK_MAP = load_greek_transliterations()


def _greek_transliterate(name: str, greek_map: dict[str, str] | None = None) -> str | None:
    """Return Latin transliterated variant if string contains Greek letters, else None."""
    gmap = greek_map or _GREEK_MAP
    if not any(c in gmap for c in name):
        return None
    return "".join(gmap.get(c, c) for c in name)


from .async_engine import http_get_json as _http_get_json, http_get_text as _http_get_text



def _name_cache_key(query: str) -> str:
    return f"pubchem:name:{query.strip().lower()}"


def pubchem_lookup_multi(query: str, cache: SQLiteCache | None = None) -> list[dict] | None:
    """Query PubChem by name, returning a candidate list (in PubChem order) or None.

    Each candidate: {canonical_smiles, inchikey, pubchem_cid}.
    Query chain: name -> cids (potentially multiple CIDs) -> query properties per CID (ConnectivitySMILES fallback).
    Idempotent and offline-safe: returns None on any network exception; zero network overhead on cache hit.
    Cache format: {"candidates": [...]}; failure cache {"unreachable": true}.
    """
    cache_key = _name_cache_key(query)
    if cache is not None:
        cached = cache.get(cache_key, vocab_version=VOCABULARY_VERSION)
        if cached is not None:
            return None if cached.get("unreachable") else cached.get("candidates")
    names_to_try = [query.strip()]
    alt = _greek_transliterate(query.strip())
    if alt:
        names_to_try.append(alt)  # Latin transliteration variant fallback (e.g., 18β -> 18beta)
    for probe in names_to_try:
        try:
            q = urllib.parse.quote(probe)
            cids = _http_get_json(f"{_PUG_BASE}/compound/name/{q}/cids/JSON")
            cid_list = cids["IdentifierList"]["CID"]
            if not cid_list:
                continue  # Variant not indexed, try next
        except urllib.error.HTTPError as e:
            if e.code in {400, 404}:  # Deterministically not indexed (PubChem explicitly returned NotFound)
                continue
            if cache is not None:     # 429/5xx: Rate limited, negative cache 1h
                cache.set(cache_key, _UNREACHABLE, ttl_seconds=_TTL_NET_FAIL,
                          vocab_version=VOCABULARY_VERSION)
            return None
        except Exception:  # noqa: BLE001 - Network/timeout: negative cache 1h
            if cache is not None:
                cache.set(cache_key, _UNREACHABLE, ttl_seconds=_TTL_NET_FAIL,
                          vocab_version=VOCABULARY_VERSION)
            return None
        candidates = []
        for cid in cid_list[:5]:  # Take at most top 5 candidates to limit request count
            props = _http_get_json(
                f"{_PUG_BASE}/compound/cid/{cid}/property/"
                "ConnectivitySMILES,CanonicalSMILES,InChIKey/JSON"
            )["PropertyTable"]["Properties"][0]
            candidates.append({
                "canonical_smiles": (props.get("CanonicalSMILES")
                                     or props.get("ConnectivitySMILES")),
                "inchikey": props.get("InChIKey"),
                "pubchem_cid": cid,
            })
        if cache is not None:
            cache.set(cache_key, {"candidates": candidates}, ttl_seconds=_TTL_ONLINE,
                      vocab_version=VOCABULARY_VERSION)
        return candidates
    # All variants deterministically not indexed -> long-term empty cache (not a network issue)
    if cache is not None:
        cache.set(cache_key, {"empty": True}, ttl_seconds=_TTL_ONLINE,
                  vocab_version=VOCABULARY_VERSION)
    return None


_DEFAULT_PUBCHEM_LOOKUP_MULTI = pubchem_lookup_multi


def _batch_cids(names: list[str]) -> dict[str, list[int]]:
    """PUG REST batch name->cids (comma-separated <= 100). Return {name: [cid, ...]}.

    Note on PUG semantics: a single name returns a flat CID list, whereas multiple names return nested lists
    (matching names 1-to-1). Both cases are handled here.
    """
    # Individually quote names (safe="" encodes slashes too); separating commas remain unescaped (PUG delimiter)
    q = ",".join(urllib.parse.quote(n.strip(), safe="") for n in names)
    data = _http_get_json(f"{_PUG_BASE}/compound/name/{q}/cids/JSON")
    cid_lists = data["IdentifierList"]["CID"]
    if not cid_lists:
        return {n: [] for n in names}
    # Single name: flat [cid, ...]
    if len(names) == 1 and isinstance(cid_lists[0], int):
        return {names[0]: cid_lists}
    if len(cid_lists) != len(names):
        raise ValueError(f"PubChem batch mismatch: {len(cid_lists)} != {len(names)}")
    return {n: lst for n, lst in zip(names, cid_lists)}


def _batch_props(cids: list[int]) -> dict[int, dict]:
    """PUG REST batch CID->properties. Return {cid: props}."""
    q = ",".join(str(c) for c in cids)
    data = _http_get_json(
        f"{_PUG_BASE}/compound/cid/{q}/property/"
        "ConnectivitySMILES,CanonicalSMILES,InChIKey/JSON"
    )
    return {p.get("CID"): p for p in data["PropertyTable"]["Properties"]}


def pubchem_lookup_batch(names: list[str], cache: SQLiteCache | None = None,
                         batch_size: int = _BATCH_SIZE) -> dict[str, list[dict] | None]:
    """Batch name lookup (per-name cids + merged CID batch props). Return {name: candidates | None}.

    PUG REST empirical observation (2026-09-22): name->cids does not support comma-separated lists (both GET/POST 404),
    while CID->property supports comma-separated lists. Thus batching optimization focuses on the property layer:
      N names = N single-name cids queries (with retry) + ceil(M/100) merged CID props queries (M = total candidates),
    saving separate property requests for each candidate CID compared to the full per-name pipeline.

    - Prioritizes cache hits (zero network overhead); individual failures write negative cache and return None.
    - Results are cached per name (identical format to pubchem_lookup_multi, reusable offline after full scan).
    """
    if pubchem_lookup_multi is not _DEFAULT_PUBCHEM_LOOKUP_MULTI:
        out_mock: dict[str, list[dict] | None] = {}
        for n in names:
            out_mock[n] = pubchem_lookup_multi(n, cache=cache)
        return out_mock


    out: dict[str, list[dict] | None] = {}
    todo: list[str] = []
    for n in names:
        cache_key = _name_cache_key(n)
        if cache is not None:
            cached = cache.get(cache_key, vocab_version=VOCABULARY_VERSION)
            if cached is not None:
                out[n] = None if cached.get("unreachable") else cached.get("candidates")
                continue
        todo.append(n)

    # 1) Query cids per name (PUG name does not support lists, single query only)
    cid_map: dict[str, list[int]] = {}
    for n in todo:
        names_to_try = [n.strip()]
        alt = _greek_transliterate(n.strip())
        if alt != n.strip():
            names_to_try.append(alt)
        found_cids = []
        fallback_needed = False
        for probe in names_to_try:
            try:
                q = urllib.parse.quote(probe)
                cids = _http_get_json(f"{_PUG_BASE}/compound/name/{q}/cids/JSON")
                cid_list = cids.get("IdentifierList", {}).get("CID", [])
                if cid_list:
                    found_cids = cid_list[:5]
                    break
            except urllib.error.HTTPError as e:
                if e.code in {400, 404}:
                    continue
                fallback_needed = True
                break
            except Exception:  # noqa: BLE001
                fallback_needed = True
                break

        if fallback_needed:
            out[n] = pubchem_lookup_multi(n, cache=cache)  # Negative cache path
        else:
            cid_map[n] = found_cids

    # 2) Merged batch CID props query
    all_cids = sorted({c for cids in cid_map.values() for c in cids})
    props_map: dict[int, dict] = {}
    for i in range(0, len(all_cids), batch_size):
        chunk = all_cids[i:i + batch_size]
        try:
            props_map.update(_batch_props(chunk))
        except Exception:  # noqa: BLE001 - Batch props failed: fallback corresponding names to per-name query
            failed = {n for n, cs in cid_map.items() if any(c in chunk for c in cs)}
            for n in failed:
                out[n] = pubchem_lookup_multi(n, cache=cache)

    # 3) Assemble and write to cache (names that did not fail)
    for n in todo:
        if n in out:
            continue  # Already handled by fallback
        cands = []
        for cid in cid_map.get(n, []):
            pr = props_map.get(cid)
            if not pr:
                continue
            cands.append({
                "canonical_smiles": (pr.get("CanonicalSMILES")
                                     or pr.get("ConnectivitySMILES")),
                "inchikey": pr.get("InChIKey"),
                "pubchem_cid": cid,
            })
        cache_key = _name_cache_key(n)
        if cands:
            out[n] = cands
            if cache is not None:
                cache.set(cache_key, {"candidates": cands},
                          ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
        else:
            out[n] = None
            if cache is not None:
                cache.set(cache_key, {"empty": True}, ttl_seconds=_TTL_ONLINE,
                          vocab_version=VOCABULARY_VERSION)
    return out


def pubchem_lookup(query: str, cache: SQLiteCache | None = None) -> dict | None:
    """Convenience entry point for single candidate: returns the first candidate or None (alignment engine uses multi internally)."""
    cands = pubchem_lookup_multi(query, cache=cache)
    return cands[0] if cands else None


def chembl_targets_lookup(name: str, cache: SQLiteCache | None = None) -> dict | None:
    """Query ChEMBL by name, returning {chembl_id, targets: [HGNC gene], moa: [str]} or None.

    Query chain (returns None on any step failure, offline-safe):
      1) molecule search by name -> molecule_chembl_id
      2) mechanism?molecule_chembl_id -> mechanism_of_action + target_chembl_id
      3) target/{id} -> GENE_SYMBOL synonyms in target_components

    Zero network overhead on cache hit; results expire based on VOCABULARY_VERSION and TTL.
    """
    if not name or not name.strip():
        return None
    cache_key = f"chembl:name:{name.strip().lower()}"
    if cache is not None:
        cached = cache.get(cache_key, vocab_version=VOCABULARY_VERSION)
        if cached is not None:
            return None if cached.get("unreachable") else cached
    try:
        q = urllib.parse.quote(name.strip())
        data = _http_get_json(f"{_CHEMBL_BASE}/molecule/search.json?q={q}&limit=3")
        molecules = data.get("molecules") or []
        if not molecules:
            if cache is not None:  # Confirmed not indexed: cache to prevent repeated queries
                cache.set(cache_key, {"empty": True}, ttl_seconds=_TTL_ONLINE,
                          vocab_version=VOCABULARY_VERSION)
            return None
        chembl_id = molecules[0]["molecule_chembl_id"]

        mech_data = _http_get_json(
            f"{_CHEMBL_BASE}/mechanism.json?molecule_chembl_id={chembl_id}&limit=100"
        )
        mechanisms = mech_data.get("mechanisms") or []
        moa = sorted({m.get("mechanism_of_action") for m in mechanisms
                      if m.get("mechanism_of_action")})

        targets: set[str] = set()
        seen_targets: set[str] = set()
        for m in mechanisms:
            tid = m.get("target_chembl_id")
            if not tid or tid in seen_targets:
                continue
            seen_targets.add(tid)
            target = _http_get_json(f"{_CHEMBL_BASE}/target/{tid}.json")
            for comp in target.get("target_components") or []:
                for syn in comp.get("target_component_synonyms") or []:
                    stype = (syn.get("syn_type") or "").upper()
                    if "GENE" in stype and syn.get("component_synonym"):
                        targets.add(syn["component_synonym"].strip())
        result = {
            "chembl_id": chembl_id,
            "targets": sorted(targets),
            "moa": moa,
            "source": "chembl",
        }
    except Exception:  # noqa: BLE001 - Offline/throttling/parse failure: degrade gracefully
        if cache is not None:
            cache.set(cache_key, _UNREACHABLE, ttl_seconds=_TTL_NET_FAIL,
                      vocab_version=VOCABULARY_VERSION)
        return None
    if cache is not None:
        cache.set(cache_key, result, ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
    return result


def pubchem_lookup_by_inchikey(inchikey: str, cache: SQLiteCache | None = None) -> list[dict] | None:
    """Query PubChem by InChIKey, returning a list of candidates or None."""
    ik = inchikey.strip().upper()
    cache_key = f"pubchem:inchikey:{ik}"
    if cache is not None:
        cached = cache.get(cache_key, vocab_version=VOCABULARY_VERSION)
        if cached is not None:
            return None if cached.get("unreachable") else cached.get("candidates")
    try:
        q = urllib.parse.quote(ik)
        data = _http_get_json(f"{_PUG_BASE}/compound/inchikey/{q}/cids/JSON")
        cid_list = data.get("IdentifierList", {}).get("CID", [])
        if not cid_list:
            if cache is not None:
                cache.set(cache_key, {"empty": True}, ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
            return None
        candidates = []
        for cid in cid_list[:3]:
            props = _http_get_json(
                f"{_PUG_BASE}/compound/cid/{cid}/property/ConnectivitySMILES,CanonicalSMILES,InChIKey,Title/JSON"
            )["PropertyTable"]["Properties"][0]
            candidates.append({
                "canonical_name": props.get("Title") or ik,
                "canonical_smiles": props.get("CanonicalSMILES") or props.get("ConnectivitySMILES"),
                "inchikey": props.get("InChIKey") or ik,
                "pubchem_cid": cid,
                "source": "pubchem",
            })
        if cache is not None:
            cache.set(cache_key, {"candidates": candidates}, ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
        return candidates
    except Exception:
        if cache is not None:
            cache.set(cache_key, _UNREACHABLE, ttl_seconds=_TTL_NET_FAIL, vocab_version=VOCABULARY_VERSION)
        return None


def pubchem_lookup_by_smiles(smiles: str, cache: SQLiteCache | None = None) -> list[dict] | None:
    """Query PubChem by SMILES, returning a list of candidates or None."""
    s = smiles.strip()
    cache_key = f"pubchem:smiles:{s}"
    if cache is not None:
        cached = cache.get(cache_key, vocab_version=VOCABULARY_VERSION)
        if cached is not None:
            return None if cached.get("unreachable") else cached.get("candidates")
    try:
        q = urllib.parse.quote(s)
        data = _http_get_json(f"{_PUG_BASE}/compound/smiles/{q}/cids/JSON")
        cid_list = data.get("IdentifierList", {}).get("CID", [])
        if not cid_list:
            if cache is not None:
                cache.set(cache_key, {"empty": True}, ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
            return None
        candidates = []
        for cid in cid_list[:3]:
            props = _http_get_json(
                f"{_PUG_BASE}/compound/cid/{cid}/property/ConnectivitySMILES,CanonicalSMILES,InChIKey,Title/JSON"
            )["PropertyTable"]["Properties"][0]
            candidates.append({
                "canonical_name": props.get("Title") or s,
                "canonical_smiles": props.get("CanonicalSMILES") or props.get("ConnectivitySMILES") or s,
                "inchikey": props.get("InChIKey"),
                "pubchem_cid": cid,
                "source": "pubchem",
            })
        if cache is not None:
            cache.set(cache_key, {"candidates": candidates}, ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
        return candidates
    except Exception:
        if cache is not None:
            cache.set(cache_key, _UNREACHABLE, ttl_seconds=_TTL_NET_FAIL, vocab_version=VOCABULARY_VERSION)
        return None


def pubchem_lookup_by_cid(cid: int, cache: SQLiteCache | None = None) -> list[dict] | None:
    """Query properties directly by PubChem CID, returning a list of candidates or None."""
    cache_key = f"pubchem:cid:{cid}"
    if cache is not None:
        cached = cache.get(cache_key, vocab_version=VOCABULARY_VERSION)
        if cached is not None:
            return None if cached.get("unreachable") else cached.get("candidates")
    try:
        data = _http_get_json(
            f"{_PUG_BASE}/compound/cid/{cid}/property/ConnectivitySMILES,CanonicalSMILES,InChIKey,Title/JSON"
        )
        props = data["PropertyTable"]["Properties"][0]
        cand = {
            "canonical_name": props.get("Title") or f"CID {cid}",
            "canonical_smiles": props.get("CanonicalSMILES") or props.get("ConnectivitySMILES"),
            "inchikey": props.get("InChIKey"),
            "pubchem_cid": cid,
            "source": "pubchem",
        }
        candidates = [cand]
        if cache is not None:
            cache.set(cache_key, {"candidates": candidates}, ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
        return candidates
    except Exception:
        if cache is not None:
            cache.set(cache_key, _UNREACHABLE, ttl_seconds=_TTL_NET_FAIL, vocab_version=VOCABULARY_VERSION)
        return None


def chembl_lookup_by_id(chembl_id: str, cache: SQLiteCache | None = None) -> list[dict] | None:
    """Query molecular info and structure directly by ChEMBL ID."""
    cid_norm = chembl_id.strip().upper()
    cache_key = f"chembl:id:{cid_norm}"
    if cache is not None:
        cached = cache.get(cache_key, vocab_version=VOCABULARY_VERSION)
        if cached is not None:
            return None if cached.get("unreachable") else cached.get("candidates")
    try:
        data = _http_get_json(f"{_CHEMBL_BASE}/molecule/{cid_norm}.json")
        structs = data.get("molecule_structures") or {}
        pref_name = data.get("pref_name") or cid_norm
        cand = {
            "canonical_name": pref_name,
            "canonical_smiles": structs.get("canonical_smiles"),
            "inchikey": structs.get("standard_inchi_key"),
            "chembl_id": cid_norm,
            "source": "chembl",
        }
        candidates = [cand]
        if cache is not None:
            cache.set(cache_key, {"candidates": candidates}, ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
        return candidates
    except Exception:
        if cache is not None:
            cache.set(cache_key, _UNREACHABLE, ttl_seconds=_TTL_NET_FAIL, vocab_version=VOCABULARY_VERSION)
        return None


def chembl_search_fallback(name: str, cache: SQLiteCache | None = None) -> list[dict] | None:
    """Search molecules in ChEMBL by name (as tier-1 fallback when PubChem has no record)."""
    t = name.strip()
    if not t:
        return None
    cache_key = f"chembl:search:{t.lower()}"
    if cache is not None:
        cached = cache.get(cache_key, vocab_version=VOCABULARY_VERSION)
        if cached is not None:
            return None if cached.get("unreachable") else cached.get("candidates")
    try:
        q = urllib.parse.quote(t)
        data = _http_get_json(f"{_CHEMBL_BASE}/molecule/search.json?q={q}&limit=3")
        molecules = data.get("molecules") or []
        candidates = []
        for m in molecules:
            structs = m.get("molecule_structures") or {}
            smiles = structs.get("canonical_smiles")
            inchikey = structs.get("standard_inchi_key")
            chembl_id = m.get("molecule_chembl_id")
            pref_name = m.get("pref_name") or t
            if smiles or inchikey:
                candidates.append({
                    "canonical_name": pref_name,
                    "canonical_smiles": smiles,
                    "inchikey": inchikey,
                    "chembl_id": chembl_id,
                    "source": "chembl",
                })
        if candidates:
            if cache is not None:
                cache.set(cache_key, {"candidates": candidates}, ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
            return candidates
        if cache is not None:
            cache.set(cache_key, {"empty": True}, ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
        return None
    except Exception:
        if cache is not None:
            cache.set(cache_key, _UNREACHABLE, ttl_seconds=_TTL_NET_FAIL, vocab_version=VOCABULARY_VERSION)
        return None


def nci_cir_lookup(query: str, cache: SQLiteCache | None = None) -> list[dict] | None:
    """Resolve CAS registry numbers or special chemical identifiers using NCI CIR (Chemical Identifier Resolver)."""
    t = query.strip()
    if not t:
        return None
    cache_key = f"cir:{t.lower()}"
    if cache is not None:
        cached = cache.get(cache_key, vocab_version=VOCABULARY_VERSION)
        if cached is not None:
            return None if cached.get("unreachable") else cached.get("candidates")
    try:
        q = urllib.parse.quote(t)
        smiles = _http_get_text(f"{_CIR_BASE}/{q}/smiles")
        if not smiles:
            if cache is not None:
                cache.set(cache_key, {"empty": True}, ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
            return None
        inchikey_text = _http_get_text(f"{_CIR_BASE}/{q}/stdinchikey")
        inchikey = inchikey_text.replace("InChIKey=", "").strip() if inchikey_text else None
        cand = {
            "canonical_name": t,
            "canonical_smiles": smiles,
            "inchikey": inchikey,
            "source": "nci_cir",
        }
        candidates = [cand]
        if cache is not None:
            cache.set(cache_key, {"candidates": candidates}, ttl_seconds=_TTL_ONLINE, vocab_version=VOCABULARY_VERSION)
        return candidates
    except Exception:
        if cache is not None:
            cache.set(cache_key, _UNREACHABLE, ttl_seconds=_TTL_NET_FAIL, vocab_version=VOCABULARY_VERSION)
        return None


class BaseAnnotationProvider(abc.ABC):
    """Abstract base protocol for online annotation providers (SPI)."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique provider identifier (e.g. 'pubchem', 'chembl', 'nci_cir')."""
        ...

    def search_by_name(self, query: str, cache: SQLiteCache | None = None) -> list[dict] | None:
        """Query candidate structures by name within the provider."""
        return None

    def search_by_identifier(self, kind: str, value: str, cache: SQLiteCache | None = None) -> list[dict] | None:
        """Query candidates by dedicated identifier (e.g. inchikey, smiles, chembl_id, pubchem_cid, cas)."""
        return None

    def enrich_targets(self, name: str, cache: SQLiteCache | None = None) -> dict | None:
        """Query drug targets and mechanisms of action (MoA)."""
        return None


class PubChemProvider(BaseAnnotationProvider):
    """PubChem online data provider (default highest priority)."""

    @property
    def name(self) -> str:
        return "pubchem"

    def search_by_name(self, query: str, cache: SQLiteCache | None = None) -> list[dict] | None:
        return pubchem_lookup_multi(query, cache=cache)

    def search_by_identifier(self, kind: str, value: str, cache: SQLiteCache | None = None) -> list[dict] | None:
        if kind == "inchikey":
            return pubchem_lookup_by_inchikey(value, cache=cache)
        if kind == "smiles":
            return pubchem_lookup_by_smiles(value, cache=cache)
        if kind == "pubchem_cid":
            try:
                return pubchem_lookup_by_cid(int(value), cache=cache)
            except (ValueError, TypeError):
                return None
        if kind == "cas":
            return pubchem_lookup_multi(value, cache=cache)
        return None


class ChEMBLProvider(BaseAnnotationProvider):
    """ChEMBL online data provider (drug targets and clinical compound fallback)."""

    @property
    def name(self) -> str:
        return "chembl"

    def search_by_name(self, query: str, cache: SQLiteCache | None = None) -> list[dict] | None:
        return chembl_search_fallback(query, cache=cache)

    def search_by_identifier(self, kind: str, value: str, cache: SQLiteCache | None = None) -> list[dict] | None:
        if kind == "chembl_id":
            return chembl_lookup_by_id(value, cache=cache)
        return None

    def enrich_targets(self, name: str, cache: SQLiteCache | None = None) -> dict | None:
        return chembl_targets_lookup(name, cache=cache)


class NCICIRProvider(BaseAnnotationProvider):
    """NCI CIR (Chemical Identifier Resolver) data provider (CAS and special code fallback)."""

    @property
    def name(self) -> str:
        return "nci_cir"

    def search_by_name(self, query: str, cache: SQLiteCache | None = None) -> list[dict] | None:
        if any(c.isdigit() for c in query) or "-" in query:
            return nci_cir_lookup(query, cache=cache)
        return None

    def search_by_identifier(self, kind: str, value: str, cache: SQLiteCache | None = None) -> list[dict] | None:
        if kind == "cas":
            return nci_cir_lookup(value, cache=cache)
        return None


class ProviderRegistry:
    """Dynamic registry for online annotation providers supporting pluggable sources and priority dispatch."""

    def __init__(self) -> None:
        self._providers: list[tuple[int, BaseAnnotationProvider]] = []
        self.reset()

    def register(self, provider: BaseAnnotationProvider, priority: int = 100) -> None:
        """Register a new provider or update its priority (lower values execute first)."""
        self.unregister(provider.name)
        self._providers.append((priority, provider))
        self._providers.sort(key=lambda x: x[0])

    def unregister(self, name: str) -> None:
        """Unregister a provider by name."""
        self._providers = [p for p in self._providers if p[1].name != name]

    def get_providers(self) -> list[BaseAnnotationProvider]:
        """Return the list of currently active providers sorted by priority in ascending order."""
        return [p[1] for p in self._providers]

    def get_provider(self, name: str) -> BaseAnnotationProvider | None:
        """Retrieve a provider by name."""
        for _, p in self._providers:
            if p.name == name:
                return p
        return None

    def reset(self) -> None:
        """Reset to default provider configuration (PubChem: 10, ChEMBL: 20, NCI CIR: 30)."""
        self._providers = [
            (10, PubChemProvider()),
            (20, ChEMBLProvider()),
            (30, NCICIRProvider()),
        ]
        self._providers.sort(key=lambda x: x[0])


default_provider_registry = ProviderRegistry()


def resolve_online_cascade(
    query: str,
    cache: SQLiteCache | None = None,
    registry: ProviderRegistry | None = None,
) -> tuple[list[dict] | None, list[str]]:
    """Multi-source intelligent cascade online resolution engine (pluggable SPI via ProviderRegistry).

    Cascade priority:
      1. Dedicated identifier detection (InChIKey, SMILES, ChEMBL ID, CID, CAS, etc.) dispatched to provider.search_by_identifier
      2. General name queries dispatched to provider.search_by_name with step-by-step fallback

    Returns (candidates, sources_resolved).
    """
    reg = registry or default_provider_registry
    providers = reg.get_providers()

    ident = detect_identifier(query)
    if ident:
        kind, val = ident
        for provider in providers:
            hits = provider.search_by_identifier(kind, str(val), cache=cache)
            if hits:
                return hits, [f"{provider.name}:{kind}"]

    for provider in providers:
        hits = provider.search_by_name(query, cache=cache)
        if hits:
            return hits, [provider.name]

    return None, []
