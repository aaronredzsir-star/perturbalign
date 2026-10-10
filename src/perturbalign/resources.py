"""Resource loaders: load external domain vocabularies (TSVs) into Python data structures.

Zero third-party dependencies (standard library csv + hashlib).
Provides the universal VocabularyRegistry for dynamic registration, overriding, and querying
of all domain dictionaries and code tables (built-in and custom).
"""
from __future__ import annotations

import csv
import hashlib
import os
from typing import Any, Callable

_RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources")


def _compute_vocab_version() -> str:
    """Compute a deterministic hash version string based on all .tsv files in the resources directory.

    Any modification automatically updates the version, ensuring dependent caches invalidate correctly.
    """
    hasher = hashlib.sha256()
    if os.path.isdir(_RESOURCE_DIR):
        for fname in sorted(os.listdir(_RESOURCE_DIR)):
            if fname.endswith(".tsv"):
                fpath = os.path.join(_RESOURCE_DIR, fname)
                try:
                    with open(fpath, "rb") as f:
                        hasher.update(fname.encode("utf-8"))
                        hasher.update(f.read())
                except OSError:
                    pass
    return f"0.9.0-{hasher.hexdigest()[:8]}"


VOCABULARY_VERSION = _compute_vocab_version()


def _read_tsv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        # Skip comment lines starting with #; the first non-comment line is the header
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    return [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]


def _resolve(path: str | None, name: str) -> str:
    return path or os.path.join(_RESOURCE_DIR, name)


class VocabularyRegistry:
    """Universal registry for perturbation alignment domain vocabularies, dictionaries, and code tables.

    Provides uniform registration, overriding, and querying across all domain dictionaries:
    - Built-in tables (22 TSVs): control_tokens, salt_suffixes, dose_units, duration_units,
      combo_separators, vendor_prefixes, formulation_codes, biologics_tokens,
      environmental_tokens, investigational_codes, stereo_tokens, inorganic_compounds,
      metal_cations, genetic_actions, genetic_control_targets, biologic_suffixes,
      greek_transliterations, perturbation_protocols, cocktails, delivery_vehicles,
      protac_e3_ligands, sgrna_reference.
    - Custom user vocabularies: arbitrary tables or in-memory mappings (dict, list, set).
    """

    def __init__(self) -> None:
        self._vocabularies: dict[str, Any] = {}

    def register(self, name: str, data_or_path: Any) -> None:
        """Register or override a domain vocabulary dictionary or table.

        Args:
            name: Vocabulary identifier (e.g. 'control_tokens', 'investigational_codes',
                  'vendor_prefixes', or any custom domain vocabulary name).
            data_or_path: Filepath (str) to a TSV file, or an in-memory collection (dict, set, list).
        """
        name_key = name.strip().lower()
        if isinstance(data_or_path, str) and (os.path.isfile(data_or_path) or data_or_path.endswith(".tsv")):
            loader = _DEFAULT_LOADERS.get(name_key)
            if loader is not None:
                self._vocabularies[name_key] = loader(data_or_path)
            else:
                self._vocabularies[name_key] = _read_tsv(data_or_path)
        else:
            if name_key == "investigational_codes" and isinstance(data_or_path, dict):
                # Normalize and index code variants (hyphens, spaces) for fast lookup
                indexed: dict[str, dict] = dict(data_or_path)
                for k, v in list(data_or_path.items()):
                    if isinstance(v, dict):
                        k_low = k.lower()
                        indexed[k_low] = v
                        indexed[k_low.replace("-", "")] = v
                        indexed[k_low.replace(" ", "")] = v
                        indexed[k_low.replace("-", "").replace(" ", "")] = v
                self._vocabularies[name_key] = indexed
            else:
                self._vocabularies[name_key] = data_or_path

    def get(self, name: str, default: Any = None) -> Any:
        """Retrieve the active vocabulary by name. Lazily loads built-in TSV if not yet cached."""
        name_key = name.strip().lower()
        if name_key in self._vocabularies:
            return self._vocabularies[name_key]
        loader = _DEFAULT_LOADERS.get(name_key)
        if loader is not None:
            val = loader()
            self._vocabularies[name_key] = val
            return val
        return default

    def list_vocabularies(self) -> list[str]:
        """List all available vocabulary names (built-ins and custom registered)."""
        names = set(_DEFAULT_LOADERS.keys()) | set(self._vocabularies.keys())
        return sorted(names)

    def reset(self, name: str | None = None) -> None:
        """Reset registered vocabulary back to factory default TSVs."""
        if name is not None:
            self._vocabularies.pop(name.strip().lower(), None)
        else:
            self._vocabularies.clear()


default_vocab_registry = VocabularyRegistry()


def register_vocabulary(name: str, data_or_path: Any) -> None:
    """Register or override a vocabulary dictionary or table in the default registry."""
    default_vocab_registry.register(name, data_or_path)


def get_vocabulary(name: str, default: Any = None) -> Any:
    """Retrieve an active vocabulary dictionary or table from the default registry."""
    return default_vocab_registry.get(name, default)


def list_vocabularies() -> list[str]:
    """List all available vocabulary names in the default registry."""
    return default_vocab_registry.list_vocabularies()


def reset_vocabulary_registry(name: str | None = None) -> None:
    """Reset the default vocabulary registry back to factory default TSVs."""
    default_vocab_registry.reset(name)


def load_control_tokens(path: str | None = None) -> set[str]:
    """Load control and vehicle tokens (lowercase whole words)."""
    if path is None and "control_tokens" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["control_tokens"]
    return {
        row["token"].strip().lower()
        for row in _read_tsv(_resolve(path, "control_tokens.tsv"))
        if row.get("token", "").strip()
    }


def load_salt_suffixes(path: str | None = None) -> list[str]:
    """Load chemical salt suffixes (sorted descending by length for greedy stripping)."""
    if path is None and "salt_suffixes" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["salt_suffixes"]
    suffixes = [
        row["suffix"].strip()
        for row in _read_tsv(_resolve(path, "salt_suffixes.tsv"))
        if row.get("suffix", "").strip()
    ]
    return sorted(suffixes, key=len, reverse=True)


def load_dose_units(path: str | None = None) -> list[tuple[str, str]]:
    """Load dosage units mapping: (surface, canonical), sorted descending by surface length."""
    if path is None and "dose_units" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["dose_units"]
    units = [
        (row["surface"].strip(), row["canonical"].strip())
        for row in _read_tsv(_resolve(path, "dose_units.tsv"))
        if row.get("surface", "").strip()
    ]
    return sorted(units, key=lambda u: len(u[0]), reverse=True)


def load_combo_separators(path: str | None = None) -> list[tuple[str, int]]:
    """Load perturbation combination delimiters: (separator, priority), sorted ascending by priority."""
    if path is None and "combo_separators" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["combo_separators"]
    seps = [
        (row["separator"], int(row["priority"]))
        for row in _read_tsv(_resolve(path, "combo_separators.tsv"))
        if row.get("separator", "")
    ]
    return sorted(seps, key=lambda s: s[1])


def load_vendor_prefixes(path: str | None = None) -> list[str]:
    """Load reagent vendor catalog prefixes (sorted descending by length)."""
    if path is None and "vendor_prefixes" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["vendor_prefixes"]
    vs = [
        row["vendor"].strip()
        for row in _read_tsv(_resolve(path, "vendor_prefixes.tsv"))
        if row.get("vendor", "").strip()
    ]
    return sorted(vs, key=len, reverse=True)


def load_formulation_codes(path: str | None = None) -> list[str]:
    """Load pharmaceutical dosage formulation suffixes (sorted descending by length)."""
    if path is None and "formulation_codes" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["formulation_codes"]
    fs = [
        row["suffix"].strip()
        for row in _read_tsv(_resolve(path, "formulation_codes.tsv"))
        if row.get("suffix", "").strip()
    ]
    return sorted(fs, key=len, reverse=True)


def load_duration_units(path: str | None = None) -> list[tuple[str, str]]:
    """Load duration and time units mapping: (surface, canonical), sorted descending by length."""
    if path is None and "duration_units" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["duration_units"]
    units = [
        (row["surface"].strip(), row["canonical"].strip())
        for row in _read_tsv(_resolve(path, "duration_units.tsv"))
        if row.get("surface", "").strip()
    ]
    return sorted(units, key=lambda u: len(u[0]), reverse=True)


def load_biologics_tokens(path: str | None = None) -> dict[str, dict]:
    """Load biologics, cytokines, and antibody dictionary: {token: {canonical_name, category, hgnc_or_uniprot}}."""
    if path is None and "biologics_tokens" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["biologics_tokens"]
    res = {}
    for row in _read_tsv(_resolve(path, "biologics_tokens.tsv")):
        tok = row.get("token", "").strip().lower()
        if tok:
            res[tok] = {
                "canonical_name": row.get("canonical_name", "").strip(),
                "category": row.get("category", "").strip(),
                "hgnc_or_uniprot": row.get("hgnc_or_uniprot", "").strip(),
            }
    return res


def load_environmental_tokens(path: str | None = None) -> dict[str, dict]:
    """Load physical and environmental stress tokens: {token: {canonical_name, category}}."""
    if path is None and "environmental_tokens" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["environmental_tokens"]
    res = {}
    for row in _read_tsv(_resolve(path, "environmental_tokens.tsv")):
        tok = row.get("token", "").strip().lower()
        if tok:
            res[tok] = {
                "canonical_name": row.get("canonical_name", "").strip(),
                "category": row.get("category", "").strip(),
            }
    return res


def load_investigational_codes(path: str | None = None) -> dict[str, dict]:
    """Load investigational compounds and R&D codes: {normalized_code: {code, canonical_name, targets, class_name, developer, smiles}}."""
    if path is None and "investigational_codes" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["investigational_codes"]
    res = {}
    for row in _read_tsv(_resolve(path, "investigational_codes.tsv")):
        raw_code = row.get("code", "").strip()
        if not raw_code:
            continue
        canon = row.get("canonical_name", "").strip()
        targets = [t.strip() for t in row.get("targets", "").split(";") if t.strip()]
        info = {
            "code": raw_code,
            "canonical_name": canon,
            "targets": targets,
            "class_name": row.get("class_name", "").strip(),
            "developer": row.get("developer", "").strip(),
            "smiles": row.get("smiles", "").strip(),
        }
        # Multi-key index: raw lowercase, stripped hyphens, stripped spaces
        keys = {
            raw_code.lower(),
            raw_code.lower().replace("-", ""),
            raw_code.lower().replace(" ", ""),
            raw_code.lower().replace("-", "").replace(" ", ""),
        }
        for k in keys:
            res[k] = info
    return res


def load_stereo_tokens(path: str | None = None) -> set[str]:
    """Load stereochemical and isomeric tokens (lowercase)."""
    if path is None and "stereo_tokens" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["stereo_tokens"]
    return {
        row["token"].strip().lower()
        for row in _read_tsv(_resolve(path, "stereo_tokens.tsv"))
        if row.get("token", "").strip()
    }


def load_inorganic_compounds(path: str | None = None) -> set[str]:
    """Load inorganic counterion tokens (lowercase, protects against over-stripping metal prefixes)."""
    if path is None and "inorganic_compounds" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["inorganic_compounds"]
    return {
        row["suffix"].strip().lower()
        for row in _read_tsv(_resolve(path, "inorganic_compounds.tsv"))
        if row.get("suffix", "").strip()
    }


def load_metal_cations(path: str | None = None) -> list[str]:
    """Load metal cation salt prefixes (sorted descending by length)."""
    if path is None and "metal_cations" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["metal_cations"]
    cations = [
        row["prefix"].strip().lower()
        for row in _read_tsv(_resolve(path, "metal_cations.tsv"))
        if row.get("prefix", "").strip()
    ]
    return sorted(cations, key=len, reverse=True)


def load_genetic_actions(path: str | None = None) -> dict[str, dict]:
    """Load genetic perturbation action mechanics: {tag_lower: {canonical_action, mechanism, description}}."""
    if path is None and "genetic_actions" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["genetic_actions"]
    res = {}
    for row in _read_tsv(_resolve(path, "genetic_actions.tsv")):
        tag = row.get("tag", "").strip()
        if tag:
            res[tag.lower()] = {
                "canonical_action": row.get("canonical_action", "").strip(),
                "mechanism": row.get("mechanism", "").strip(),
                "description": row.get("description", "").strip(),
            }
    return res


def load_genetic_control_targets(path: str | None = None) -> set[str]:
    """Load genetic negative/safe control targets (lowercase, e.g., nt, scramble, aavs1, luciferase)."""
    if path is None and "genetic_control_targets" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["genetic_control_targets"]
    return {
        row["target"].strip().lower()
        for row in _read_tsv(_resolve(path, "genetic_control_targets.tsv"))
        if row.get("target", "").strip()
    }


def load_biologic_suffixes(path: str | None = None) -> list[tuple[str, str]]:
    """Load biologic drug INN suffix rules: [(suffix, category), ...], sorted descending by suffix length."""
    if path is None and "biologic_suffixes" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["biologic_suffixes"]
    suffixes = [
        (row["suffix"].strip().lower(), row["category"].strip())
        for row in _read_tsv(_resolve(path, "biologic_suffixes.tsv"))
        if row.get("suffix", "").strip()
    ]
    return sorted(suffixes, key=lambda s: len(s[0]), reverse=True)


def load_greek_transliterations(path: str | None = None) -> dict[str, str]:
    """Load Greek character to Latin transliteration table: {greek_char: latin}."""
    if path is None and "greek_transliterations" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["greek_transliterations"]
    return {
        row["greek_char"].strip(): row["latin"].strip()
        for row in _read_tsv(_resolve(path, "greek_transliteration.tsv"))
        if row.get("greek_char", "").strip()
    }


def load_perturbation_protocols(path: str | None = None) -> dict[str, dict]:
    """Load public perturbation protocol metadata schemas: {protocol_key: {standard_source, canonical_modality, default_action, description}}."""
    if path is None and "perturbation_protocols" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["perturbation_protocols"]
    res = {}
    for row in _read_tsv(_resolve(path, "perturbation_protocols.tsv")):
        key = row.get("protocol_key", "").strip().lower()
        if key:
            res[key] = {
                "standard_source": row.get("standard_source", "").strip(),
                "canonical_modality": row.get("canonical_modality", "").strip(),
                "default_action": row.get("default_action", "").strip(),
                "description": row.get("description", "").strip(),
            }
    return res


def load_cocktails(path: str | None = None) -> dict[str, dict]:
    """Load predefined cytokine cocktails and polarization mixes: {cocktail_name: {canonical_name, components, category, description}}."""
    if path is None and "cocktails" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["cocktails"]
    res = {}
    for row in _read_tsv(_resolve(path, "cocktails.tsv")):
        name = row.get("cocktail_name", "").strip().lower()
        if name:
            components_str = row.get("components", "").strip()
            components = [c.strip() for c in components_str.split("+") if c.strip()]
            res[name] = {
                "canonical_name": row.get("canonical_name", "").strip(),
                "components": components,
                "components_str": components_str,
                "category": row.get("category", "").strip(),
                "description": row.get("description", "").strip(),
            }
    return res


def load_delivery_vehicles(path: str | None = None) -> dict[str, dict]:
    """Load delivery vehicles, LNP formulations, and AAV serotypes: {token: {canonical_name, carrier_type, serotype, formulation, tropism, description}}."""
    if path is None and "delivery_vehicles" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["delivery_vehicles"]
    res = {}
    for row in _read_tsv(_resolve(path, "delivery_vehicles.tsv")):
        tok = row.get("token", "").strip().lower()
        if tok:
            form_str = row.get("formulation", "").strip()
            formulation = [f.strip() for f in form_str.split(",") if f.strip()]
            res[tok] = {
                "canonical_name": row.get("canonical_name", "").strip(),
                "carrier_type": row.get("carrier_type", "").strip(),
                "serotype": row.get("serotype", "").strip() or None,
                "formulation": formulation,
                "tropism": row.get("tropism", "").strip() or None,
                "description": row.get("description", "").strip(),
            }
    return res


def load_protac_e3_ligands(path: str | None = None) -> dict[str, dict]:
    """Load PROTAC and molecular glue E3 ligase recruiting ligands: {token: {canonical_name, e3_ligase, e3_ligand, target_protein, is_molecular_glue, description}}."""
    if path is None and "protac_e3_ligands" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["protac_e3_ligands"]
    res = {}
    for row in _read_tsv(_resolve(path, "protac_e3_ligands.tsv")):
        tok = row.get("token", "").strip().lower()
        if tok:
            is_glue = row.get("is_molecular_glue", "False").strip().lower() in ("true", "1", "yes")
            res[tok] = {
                "canonical_name": row.get("canonical_name", "").strip(),
                "e3_ligase": row.get("e3_ligase", "").strip(),
                "e3_ligand": row.get("e3_ligand", "").strip(),
                "target_protein": row.get("target_protein", "").strip() or None,
                "is_molecular_glue": is_glue,
                "description": row.get("description", "").strip(),
            }
    return res


def load_sgrna_reference(path: str | None = None) -> dict[str, dict]:
    """Load CRISPR sgRNA 20bp protospacers to target gene reference: {protospacer_seq_upper: {gene_symbol, species, source, description}}."""
    if path is None and "sgrna_reference" in default_vocab_registry._vocabularies:
        return default_vocab_registry._vocabularies["sgrna_reference"]
    res = {}
    for row in _read_tsv(_resolve(path, "sgrna_reference.tsv")):
        seq = row.get("protospacer_seq", "").strip().upper()
        if seq:
            res[seq] = {
                "gene_symbol": row.get("gene_symbol", "").strip(),
                "species": row.get("species", "").strip(),
                "source": row.get("source", "").strip(),
                "description": row.get("description", "").strip(),
            }
    return res


_DEFAULT_LOADERS: dict[str, Callable[..., Any]] = {
    "control_tokens": load_control_tokens,
    "salt_suffixes": load_salt_suffixes,
    "dose_units": load_dose_units,
    "combo_separators": load_combo_separators,
    "vendor_prefixes": load_vendor_prefixes,
    "formulation_codes": load_formulation_codes,
    "duration_units": load_duration_units,
    "biologics_tokens": load_biologics_tokens,
    "environmental_tokens": load_environmental_tokens,
    "investigational_codes": load_investigational_codes,
    "stereo_tokens": load_stereo_tokens,
    "inorganic_compounds": load_inorganic_compounds,
    "metal_cations": load_metal_cations,
    "genetic_actions": load_genetic_actions,
    "genetic_control_targets": load_genetic_control_targets,
    "biologic_suffixes": load_biologic_suffixes,
    "greek_transliterations": load_greek_transliterations,
    "perturbation_protocols": load_perturbation_protocols,
    "cocktails": load_cocktails,
    "delivery_vehicles": load_delivery_vehicles,
    "protac_e3_ligands": load_protac_e3_ligands,
    "sgrna_reference": load_sgrna_reference,
}


__all__ = [
    "VOCABULARY_VERSION",
    "VocabularyRegistry",
    "default_vocab_registry",
    "register_vocabulary",
    "get_vocabulary",
    "list_vocabularies",
    "reset_vocabulary_registry",
    "load_control_tokens",
    "load_salt_suffixes",
    "load_dose_units",
    "load_duration_units",
    "load_combo_separators",
    "load_vendor_prefixes",
    "load_formulation_codes",
    "load_biologics_tokens",
    "load_environmental_tokens",
    "load_investigational_codes",
    "load_stereo_tokens",
    "load_inorganic_compounds",
    "load_metal_cations",
    "load_genetic_actions",
    "load_genetic_control_targets",
    "load_biologic_suffixes",
    "load_greek_transliterations",
    "load_perturbation_protocols",
    "load_cocktails",
    "load_delivery_vehicles",
    "load_protac_e3_ligands",
    "load_sgrna_reference",
]
