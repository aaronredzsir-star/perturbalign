"""Module 1: Preprocessor & Parser (Data-driven architecture).

Domain knowledge (control tokens, salt suffixes, dose units, combination separators, vendor prefixes,
formulation suffixes) is externalized to resources/*.tsv and loaded by PreprocessConfig;
code handles general syntax parsing and execution logic. Users can extend via
align(extra_control_tokens=[...], ...) or custom configuration files.

Optimizations:
- Salt suffix lookbehind upgraded to (?<![A-Za-z0-9-]) (triphosphate / 1-phosphate preserved)
- Metal + inorganic anions (Sodium Chloride) preserve prefix
- strip_leading_parens: stereochemistry prefix removal + full enclosing parens unnesting
- parse_dose excludes uppercase gene symbols (e.g. NOTCH2NL)
- split_underscore preserves hyphenated gene symbols (e.g. BCR-ABL_A269V)
- split_combo fragment quality gates for symbol delimiters (+ / :)
- Vendor prefix and formulation suffix stripping
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from . import resources


def normalize_input_text(raw_input: Any) -> tuple[str, bool, str]:
    """Phase 0: Global input normalizer (canonical cleaning entry point).

    - Safely handles None / NaN / empty values;
    - Applies Unicode NFKC normalization (standardizes fullwidth parens, colons, plus signs, non-breaking spaces);
    - Strips zero-width characters and invisible control characters;
    - Normalizes whitespace.

    Returns (normalized_text, is_empty, empty_note).
    """
    if raw_input is None:
        return "", True, "empty input (None)"
    try:
        import pandas as pd  # type: ignore
        if pd.isna(raw_input):
            return "", True, "empty input (NaN/NA)"
    except (ImportError, TypeError, ValueError):
        pass

    raw_str = str(raw_input)
    if not raw_str.strip():
        return "", True, "empty input"

    norm = unicodedata.normalize("NFKC", raw_str)
    norm = re.sub(r"[\u200b-\u200f\ufeff]", "", norm)
    clean = " ".join(norm.split())
    if not clean:
        return "", True, "empty input"

    return clean, False, ""

# --- General syntax rules ---
_TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")
_LEADING_STEREO_RE = re.compile(r"^\s*\([+\-±SREDZLcistrans/]+\)\s*-")
_TRAILING_JUNK_RE = re.compile(r"[\s?.,;:!]+$")
_RATIO_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)?:\d+(?:\.\d+)?\s+")
_WORD_AND_RE = re.compile(r"\s+and\s+", re.IGNORECASE)
# All-uppercase gene names (NOTCH2NL / USP6NL / EGFR etc.): excluded from dose parsing
_ALL_CAPS_GENE_RE = re.compile(r"^[A-Z]{2,}\d+[A-Z]{0,2}$")
# Gene_mutation/perturbation tags (including fusion gene hyphens such as BCR-ABL_A269V, and dTAG/PROTAC/DN suffixes)
_GENE_MUTATION_UNDERSCORE_RE = re.compile(
    r"[A-Za-z0-9-]+_(?:[A-Za-z]+\d+[A-Za-z]+|WT|KO|KD|OE|sg\w+|sh\w+|dTAG|PROTAC|AID|DN|degron|p\.\w+|ABE|CBE|PE\w*)"
)
# Symbol delimiters (without whitespace), requiring fragment quality gate
_SYMBOL_SEPS = {"+", "/", ":"}
# Endogenous genes and safe-harbor sites requiring genetic prefix/suffix protection
_ENDOGENOUS_SAFE_HARBOR_LOCI = frozenset({"ccr5", "hprt1", "rosa26", "aavs1", "chr2", "chr8", "chr12"})


@dataclass
class PreprocessConfig:
    """Preprocessor configuration: loaded from resources/*.tsv, customizable and extensible."""

    control_tokens: set[str] = field(default_factory=resources.load_control_tokens)
    salt_suffixes: list[str] = field(default_factory=resources.load_salt_suffixes)
    dose_units: list[tuple[str, str]] = field(default_factory=resources.load_dose_units)
    duration_units: list[tuple[str, str]] = field(default_factory=resources.load_duration_units)
    combo_separators: list[tuple[str, int]] = field(default_factory=resources.load_combo_separators)
    vendor_prefixes: list[str] = field(default_factory=resources.load_vendor_prefixes)
    formulation_codes: list[str] = field(default_factory=resources.load_formulation_codes)
    biologics_tokens: dict[str, dict] = field(default_factory=resources.load_biologics_tokens)
    environmental_tokens: dict[str, dict] = field(default_factory=resources.load_environmental_tokens)
    stereo_tokens: set[str] = field(default_factory=resources.load_stereo_tokens)
    inorganic_compounds: set[str] = field(default_factory=resources.load_inorganic_compounds)
    metal_cations: list[str] = field(default_factory=resources.load_metal_cations)
    genetic_actions: dict[str, dict] = field(default_factory=resources.load_genetic_actions)
    genetic_control_targets: set[str] = field(default_factory=resources.load_genetic_control_targets)
    biologic_suffixes: list[tuple[str, str]] = field(default_factory=resources.load_biologic_suffixes)
    perturbation_protocols: dict[str, dict] = field(default_factory=resources.load_perturbation_protocols)
    cocktails: dict[str, dict] = field(default_factory=resources.load_cocktails)
    delivery_vehicles: dict[str, dict] = field(default_factory=resources.load_delivery_vehicles)
    protac_e3_ligands: dict[str, dict] = field(default_factory=resources.load_protac_e3_ligands)
    sgrna_reference: dict[str, dict] = field(default_factory=resources.load_sgrna_reference)
    extra_control_tokens: set[str] = field(default_factory=set)
    max_clean_iters: int = 8  # Maximum alternating iterations in clean_name

    def __post_init__(self):
        # Salt suffixes sorted by length descending (xHCl/2HCl/trihydrate before shorter suffixes to avoid fragments).
        # Letter-starting suffixes use (?<![A-Za-z0-9-]) lookbehind: reject adjacent letters/digits/hyphens
        # (phosphate in triphosphate preceded by "i", 1-phosphate preceded by "-", neither is stripped),
        # allowing whitespace or string start ("Doxorubicin HCl"). Number-starting suffixes (2HCl) match whole tokens.
        self._salt_re = re.compile(
            r"\s*(?:" + "|".join(
                (rf"(?<![A-Za-z0-9-]){re.escape(s)}" if s[0].isalpha() else re.escape(s))
                for s in sorted(self.salt_suffixes, key=len, reverse=True)
            ) + r")\s*$",
            re.IGNORECASE,
        )
        # Prefix metal salts (Sodium Phenylbutyrate -> Phenylbutyrate);
        # Metal halides/carbonates (Sodium Chloride) protected by inorganic fallback in strip_salt.
        self._salt_prefix_re = re.compile(
            r"^(?:" + "|".join(re.escape(x) for x in sorted(self.metal_cations, key=len, reverse=True)) + r")\s+",
            re.IGNORECASE,
        )
        # Inorganic compound counter-ions / word endings (prevents Sodium Chloride -> Chloride)
        self._inorganic_re = re.compile(
            r"^(?:" + "|".join(re.escape(s) for s in sorted(self.inorganic_compounds, key=len, reverse=True)) + r")\b$",
            re.IGNORECASE,
        )
        # Stereochemistry prefix tokens (2,3-cis- / ortho- etc.)
        self._stereo_prefix_re = re.compile(
            r"^(?:\d+(?:,\d+)*-)?(?:" + "|".join(re.escape(s) for s in sorted(self.stereo_tokens, key=len, reverse=True)) + r")-",
            re.IGNORECASE,
        )
        # Genetic perturbation suffix tags (KO, KD, OE, WT, sg1, sh2, CRISPRi...)
        act_tags = list(self.genetic_actions.keys()) + [r"sg\d+", r"sh\d+"]
        self._genetic_suffix_re = re.compile(
            r"^([A-Za-z0-9-]+)[-_ ](" + "|".join(re.escape(t) if not t.endswith(r"\d+") else t for t in sorted(act_tags, key=len, reverse=True)) + r")$",
            re.IGNORECASE,
        )
        surfaces = [s for s, _ in self.dose_units]
        unit_alt = "|".join(re.escape(s) for s in surfaces)
        # Dose at end: "Trametinib 10nM" / "Geldanamycin_1uM" / "BMS-345541_10.0_uM" / "1e-5 M"
        self._dose_re = re.compile(
            r"^(?P<name>.+?)[_ ]?(?P<dose>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)[_ \s]*"
            r"(?P<unit>" + unit_alt + r")\s*$",
            re.IGNORECASE,
        )
        # Dose at start: "10nM Trametinib" (must have whitespace between unit and name)
        self._dose_leading_re = re.compile(
            r"^(?P<dose>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)[_ \s]*(?P<unit>" + unit_alt + r")\s+(?P<name>.+?)\s*$",
            re.IGNORECASE,
        )
        # Dose in parentheses: "Trametinib (10nM)"
        self._dose_paren_re = re.compile(
            r"^(?P<name>.+?)\s*\(\s*(?P<dose>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)[_ \s]*(?P<unit>" + unit_alt + r")\s*\)\s*$",
            re.IGNORECASE,
        )
        self._unit_canon = {s.lower(): c for s, c in self.dose_units}

        # Standalone dose (no name, e.g. "10nM")
        self._dose_standalone_re = re.compile(
            r"^\s*(?P<dose>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)[_ \s]*(?P<unit>" + unit_alt + r")\s*$",
            re.IGNORECASE,
        )

        # Duration unit regex
        dur_surfaces = [s for s, _ in self.duration_units]
        dur_alt = "|".join(re.escape(s) for s in dur_surfaces)
        # Duration at end: "24h" / "24 h" / "_24hr"
        self._duration_re = re.compile(
            r"^(?P<name>.+?)[_ ]?(?P<dur>\d+(?:\.\d+)?)[_ \s]*(?P<unit>" + dur_alt + r")\s*$",
            re.IGNORECASE,
        )
        # Duration at start: "24h Trametinib"
        self._duration_leading_re = re.compile(
            r"^(?P<dur>\d+(?:\.\d+)?)\s*(?P<unit>" + dur_alt + r")\s+(?P<name>.+?)\s*$",
            re.IGNORECASE,
        )
        # Duration in parentheses: "Trametinib (24h)"
        self._duration_paren_re = re.compile(
            r"^(?P<name>.+?)\s*\(\s*(?P<dur>\d+(?:\.\d+)?)\s*(?P<unit>" + dur_alt + r")\s*\)\s*$",
            re.IGNORECASE,
        )
        # Standalone duration (no name, e.g. "24h")
        self._duration_standalone_re = re.compile(
            r"^\s*(?P<dur>\d+(?:\.\d+)?)\s*(?P<unit>" + dur_alt + r")\s*$",
            re.IGNORECASE,
        )
        self._duration_canon = {s.lower(): c for s, c in self.duration_units}

        # Vendor prefix regex: vendor + whitespace + code-like token with digit
        self._vendor_re = re.compile(
            r"^(?:" + "|".join(re.escape(v) for v in self.vendor_prefixes) + r")\s+(?=\S*\d)",
            re.IGNORECASE,
        )
        # Formulation/manufacturer/regulatory suffix regex: standalone token at end
        self._formulation_re = re.compile(
            r"\s+(?:" + "|".join(
                re.escape(f) for f in sorted(self.formulation_codes, key=len, reverse=True)
            ) + r")\s*$",
            re.IGNORECASE,
        )

    @property
    def all_control_tokens(self) -> set[str]:
        gen_ctrls = {t.lower() for t in self.genetic_control_targets if t.lower() not in _ENDOGENOUS_SAFE_HARBOR_LOCI}
        return self.control_tokens | gen_ctrls | {t.lower() for t in self.extra_control_tokens}

    def with_extra_controls(self, extra_control_tokens: set[str] | list[str]) -> "PreprocessConfig":
        """Quickly derive a new configuration appending control tokens, reusing regexes with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_cfg.extra_control_tokens = set(self.extra_control_tokens) | {t.lower() for t in extra_control_tokens}
        return new_cfg

    def with_extra_biologics(self, biologics: dict[str, dict] | list[str] | None = None, **kwargs) -> "PreprocessConfig":
        """Dynamically append biologics/cytokines configuration with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_dict = dict(self.biologics_tokens)
        if isinstance(biologics, dict):
            new_dict.update({k.lower(): v for k, v in biologics.items()})
        elif isinstance(biologics, (list, set, tuple)):
            for b in biologics:
                new_dict[b.lower()] = {"canonical_name": b, "category": "biologic", "hgnc_or_uniprot": ""}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                new_dict[k.lower()] = v
            else:
                new_dict[k.lower()] = {"canonical_name": str(v), "category": "biologic", "hgnc_or_uniprot": ""}
        new_cfg.biologics_tokens = new_dict
        return new_cfg

    def with_extra_environmental(self, env_tokens: dict[str, dict] | list[str] | None = None, **kwargs) -> "PreprocessConfig":
        """Dynamically append environmental/physical/metabolic stress tokens with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_dict = dict(self.environmental_tokens)
        if isinstance(env_tokens, dict):
            new_dict.update({k.lower(): v for k, v in env_tokens.items()})
        elif isinstance(env_tokens, (list, set, tuple)):
            for e in env_tokens:
                new_dict[e.lower()] = {"canonical_name": e, "category": "stress"}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                new_dict[k.lower()] = v
            else:
                new_dict[k.lower()] = {"canonical_name": str(v), "category": "stress"}
        new_cfg.environmental_tokens = new_dict
        return new_cfg

    def with_extra_inorganics(self, *compounds: str) -> "PreprocessConfig":
        """Dynamically append inorganic compound counter-ions/suffixes with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_cfg.inorganic_compounds = set(self.inorganic_compounds) | {c.lower() for c in compounds}
        new_cfg._inorganic_re = re.compile(
            r"^(?:" + "|".join(re.escape(s) for s in sorted(new_cfg.inorganic_compounds, key=len, reverse=True)) + r")\b$",
            re.IGNORECASE,
        )
        return new_cfg

    def with_extra_metal_cations(self, *cations: str) -> "PreprocessConfig":
        """Dynamically append metal cations vocabulary with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_cfg.metal_cations = sorted(set(self.metal_cations) | set(cations), key=len, reverse=True)
        new_cfg._salt_prefix_re = re.compile(
            r"^(?:" + "|".join(re.escape(x) for x in new_cfg.metal_cations) + r")\s+",
            re.IGNORECASE,
        )
        return new_cfg

    def with_extra_salt_suffixes(self, *suffixes: str) -> "PreprocessConfig":
        """Dynamically append salt suffixes with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_cfg.salt_suffixes = sorted(set(self.salt_suffixes) | set(suffixes), key=len, reverse=True)
        new_cfg._salt_re = re.compile(
            r"\s*(?:" + "|".join(
                (rf"(?<![A-Za-z0-9-]){re.escape(s)}" if s[0].isalpha() else re.escape(s))
                for s in new_cfg.salt_suffixes
            ) + r")\s*$",
            re.IGNORECASE,
        )
        return new_cfg

    def with_extra_combo_separators(self, *separators: str | tuple[str, int]) -> "PreprocessConfig":
        """Dynamically append combination separators with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        current = list(self.combo_separators)
        for sep in separators:
            if isinstance(sep, tuple):
                current.append(sep)
            else:
                current.append((sep, 100))
        new_cfg.combo_separators = sorted(current, key=lambda x: x[1])
        return new_cfg

    def with_extra_dose_units(self, *units: tuple[str, str]) -> "PreprocessConfig":
        """Dynamically append dose units and canonical names with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_cfg.dose_units = list(self.dose_units) + list(units)
        new_cfg.__post_init__()
        return new_cfg

    def with_extra_duration_units(self, *units: tuple[str, str]) -> "PreprocessConfig":
        """Dynamically append duration units and canonical names with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_cfg.duration_units = list(self.duration_units) + list(units)
        new_cfg.__post_init__()
        return new_cfg

    def with_extra_genetic_actions(self, **actions: str | dict) -> "PreprocessConfig":
        """Dynamically append genetic action modality tag mappings with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_acts = dict(self.genetic_actions)
        for k, v in actions.items():
            if isinstance(v, dict):
                new_acts[k.lower()] = v
            else:
                new_acts[k.lower()] = {"canonical_action": str(v), "modality": "genetic", "description": ""}
        new_cfg.genetic_actions = new_acts
        act_tags = list(new_cfg.genetic_actions.keys()) + [r"sg\d+", r"sh\d+"]
        new_cfg._genetic_suffix_re = re.compile(
            r"^([A-Za-z0-9-]+)[-_ ](" + "|".join(re.escape(t) if not t.endswith(r"\d+") else t for t in sorted(act_tags, key=len, reverse=True)) + r")$",
            re.IGNORECASE,
        )
        return new_cfg

    def with_extra_genetic_control_targets(self, *targets: str) -> "PreprocessConfig":
        """Dynamically append genetic negative control targets with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_cfg.genetic_control_targets = set(self.genetic_control_targets) | {t.lower() for t in targets}
        return new_cfg

    def with_extra_perturbation_protocols(self, **protocols: dict) -> "PreprocessConfig":
        """Dynamically append perturbation experimental protocol metadata with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_protos = dict(self.perturbation_protocols)
        for k, v in protocols.items():
            new_protos[k.lower()] = v
        new_cfg.perturbation_protocols = new_protos
        return new_cfg

    def with_extra_cocktails(self, **cocktails: dict | str) -> "PreprocessConfig":
        """Dynamically append immune cocktails or differentiation recipes with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_c = dict(self.cocktails)
        for k, v in cocktails.items():
            if isinstance(v, dict):
                new_c[k.lower()] = v
            else:
                comp_str = str(v)
                comps = [c.strip() for c in comp_str.split("+") if c.strip()]
                new_c[k.lower()] = {
                    "canonical_name": k,
                    "components": comps,
                    "components_str": comp_str,
                    "category": "custom_cocktail",
                    "description": "",
                }
        new_cfg.cocktails = new_c
        return new_cfg

    def with_extra_delivery_vehicles(self, **vehicles: dict | str) -> "PreprocessConfig":
        """Dynamically append delivery vehicles or serotypes with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_v = dict(self.delivery_vehicles)
        for k, v in vehicles.items():
            if isinstance(v, dict):
                new_v[k.lower()] = v
            else:
                new_v[k.lower()] = {
                    "canonical_name": str(v),
                    "carrier_type": "custom",
                    "serotype": None,
                    "formulation": [],
                    "tropism": None,
                    "description": "",
                }
        new_cfg.delivery_vehicles = new_v
        return new_cfg

    def with_extra_protac_ligands(self, **ligands: dict) -> "PreprocessConfig":
        """Dynamically append PROTAC / molecular glue E3 ligands with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_l = dict(self.protac_e3_ligands)
        for k, v in ligands.items():
            new_l[k.lower()] = v
        new_cfg.protac_e3_ligands = new_l
        return new_cfg

    def with_extra_sgrna(self, **guides: str | dict) -> "PreprocessConfig":
        """Dynamically append 20bp sgRNA target sequence mappings with zero disk I/O."""
        import copy
        new_cfg = copy.copy(self)
        new_s = dict(self.sgrna_reference)
        for k, v in guides.items():
            seq_up = k.strip().upper()
            if isinstance(v, dict):
                new_s[seq_up] = v
            else:
                new_s[seq_up] = {"gene_symbol": str(v), "species": "unknown", "source": "user", "description": ""}
        new_cfg.sgrna_reference = new_s
        return new_cfg


DEFAULT_CONFIG = PreprocessConfig()


def _raw_parse_dose(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> tuple[str, float | None, str | None]:
    """Underlying single dose pattern match."""
    def _canon(unit_surface: str) -> str:
        return config._unit_canon.get(unit_surface.lower(), unit_surface)

    m = config._dose_standalone_re.match(text)
    if m:
        return "", float(m.group("dose")), _canon(m.group("unit"))
    m = config._dose_paren_re.match(text)
    if m:
        return m.group("name").strip(" _,;/:"), float(m.group("dose")), _canon(m.group("unit"))
    m = config._dose_re.match(text)
    if m:
        return m.group("name").strip(" _,;/:"), float(m.group("dose")), _canon(m.group("unit"))
    m = config._dose_leading_re.match(text)
    if m:
        return m.group("name").strip(" _,;/:"), float(m.group("dose")), _canon(m.group("unit"))
    return text, None, None


def _raw_parse_duration(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> tuple[str, float | None, str | None]:
    """Underlying single duration pattern match."""
    def _canon(unit_surface: str) -> str:
        return config._duration_canon.get(unit_surface.lower(), unit_surface)

    m = config._duration_standalone_re.match(text)
    if m:
        return "", float(m.group("dur")), _canon(m.group("unit"))
    m = config._duration_paren_re.match(text)
    if m:
        return m.group("name").strip(" _,;/:"), float(m.group("dur")), _canon(m.group("unit"))
    m = config._duration_re.match(text)
    if m:
        return m.group("name").strip(" _,;/:"), float(m.group("dur")), _canon(m.group("unit"))
    m = config._duration_leading_re.match(text)
    if m:
        return m.group("name").strip(" _,;/:"), float(m.group("dur")), _canon(m.group("unit"))
    return text, None, None


_COMBO_SPATIO_PAREN_RE = re.compile(
    r"^(?P<name>.+?)\s*\(\s*(?P<part1>[^,()]+)(?:,|\s+)(?P<part2>[^,()]+)\s*\)\s*$"
)


def parse_duration(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> tuple[str, float | None, str | None]:
    """Extract duration and unit, returning (name_without_duration, duration, canonical_unit)."""
    text = text.strip()
    if _ALL_CAPS_GENE_RE.match(text) or _GENE_MUTATION_UNDERSCORE_RE.fullmatch(text):
        return text, None, None
    return _raw_parse_duration(text, config)


def parse_spatiotemporal(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> tuple[str, float | None, str | None, float | None, str | None]:
    """Simultaneously extract dose and duration. Returns (name_without_dose_duration, dose, canonical_dose_unit, duration, canonical_duration_unit).

    Supports complex spatiotemporal notations:
    - Trailing series: "Trametinib 10nM 24h" / "Trametinib_10nM_24h" / "Trametinib 24h 10nM"
    - Parenthesized combinations: "Trametinib (10nM, 24h)" / "EGF (10ng/ml 30min)"
    - Duration-only: "DMSO 6h" / "Hypoxia 24h"
    - Protected gene tags: "NOTCH2NL" / "EGFR_T790M" not parsed as doses
    """
    text = text.strip()
    if _ALL_CAPS_GENE_RE.match(text) or _GENE_MUTATION_UNDERSCORE_RE.fullmatch(text):
        return text, None, None, None, None

    # 1) Check parenthesized combinations: Name (10nM, 24h)
    m_paren = _COMBO_SPATIO_PAREN_RE.match(text)
    if m_paren:
        base_name = m_paren.group("name").strip()
        p1 = m_paren.group("part1").strip()
        p2 = m_paren.group("part2").strip()
        _, d1, u1 = _raw_parse_dose(p1, config)
        _, t1, tu1 = _raw_parse_duration(p1, config)
        _, d2, u2 = _raw_parse_dose(p2, config)
        _, t2, tu2 = _raw_parse_duration(p2, config)
        if d1 is not None and t2 is not None:
            return base_name, d1, u1, t2, tu2
        if t1 is not None and d2 is not None:
            return base_name, d2, u2, t1, tu1

    curr_name = text
    dose: float | None = None
    dose_unit: str | None = None
    dur: float | None = None
    dur_unit: str | None = None

    # 2) Multi-round alternating extraction (up to 2 rounds)
    for _ in range(2):
        curr_name = re.sub(r"[\s,;/:_]+$", "", curr_name)
        if dur is None:
            n, t, tu = _raw_parse_duration(curr_name, config)
            if t is not None:
                dur, dur_unit = t, tu
                curr_name = re.sub(r"[\s,;/:_]+$", "", n)
                continue
        if dose is None:
            n, d, du = _raw_parse_dose(curr_name, config)
            if d is not None:
                dose, dose_unit = d, du
                curr_name = re.sub(r"[\s,;/:_]+$", "", n)
    # 3) Check trailing alias parentheses: "Trametinib 10uM (GSK1120212)"
    if dose is None and dur is None:
        m_alias = _TRAILING_PAREN_RE.search(curr_name)
        if m_alias:
            inner = curr_name[m_alias.start() : m_alias.end()].strip(" ()")
            _, in_d, _, in_t, _ = parse_spatiotemporal(inner, config)
            if in_d is None and in_t is None:
                prefix_candidate = curr_name[: m_alias.start()].strip()
                p_base, p_d, p_u, p_t, p_tu = parse_spatiotemporal(prefix_candidate, config)
                if p_d is not None or p_t is not None:
                    alias_part = curr_name[m_alias.start() :].strip()
                    combined_name = f"{p_base} {alias_part}".strip()
                    return combined_name, p_d, p_u, p_t, p_tu

    return re.sub(r"[\s,;/:_]+$", "", curr_name).strip(), dose, dose_unit, dur, dur_unit


def parse_dose(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> tuple[str, float | None, str | None]:
    """Extract dose and unit, returning (name_without_dose, dose, canonical_unit).

    Compatible with duration-bearing inputs (e.g. "Trametinib 10nM 24h"), peeling duration to protect drug names.
    All-uppercase gene names (NOTCH2NL / USP6NL) skip dose parsing completely.
    """
    clean_name, dose, unit, _dur, _dunit = parse_spatiotemporal(text, config)
    return clean_name, dose, unit


# --- Multimodal Entity Detectors (Genetic, Biologic, Environmental) ---
_GENETIC_PREFIX_RE = re.compile(
    r"^(?:(?P<pfx>sg(?:RNA)?|sh(?:RNA)?|CRISPR[ia]?)(?P<sep>[-_]?))(?P<target>[A-Za-z0-9-]+?)(?:[-_](?P<subidx>\d+|[a-z]\d*))?$",
    re.IGNORECASE,
)


def _is_valid_genetic_target(target: str, pfx: str, sep: str, raw_text: str) -> bool:
    """Validate whether target captured by regex is a legitimate genetic target to avoid false positives on investigational codes."""
    if len(target) < 2 or target.isdigit():
        return False
    if not sep and pfx.lower() in ("sg", "sh"):
        if not (raw_text.startswith("sg") or raw_text.startswith("sh")):
            return False
        if target.islower() and len(target) > 3:
            return False
    return True
_GENETIC_MUTANT_RE = re.compile(
    r"^([A-Za-z0-9-]+)[-_]([A-Za-z]+\d+[A-Za-z]+)$",
)
_DUAL_CRISPR_RE = re.compile(
    r"^sg(?:RNA)?[-_]?([A-Za-z0-9-]+?)[+/&_]sg(?:RNA)?[-_]?([A-Za-z0-9-]+?)$",
    re.IGNORECASE,
)
_BASE_PRIME_RE = re.compile(
    r"^(?:(ABE|CBE|PE[23]?)[-_]([A-Za-z0-9-]+)|([A-Za-z0-9-]+)[-_](ABE|CBE|PE[23]?))$",
    re.IGNORECASE,
)
_DEGRON_RE = re.compile(
    r"^(?:([A-Za-z0-9-]+)[-_](dTAG|PROTAC|AID|degron)|(dTAG|PROTAC|AID|degron)[-_]([A-Za-z0-9-]+))$",
    re.IGNORECASE,
)
_DOMINANT_NEGATIVE_RE = re.compile(
    r"^(?:DN[-_]([A-Za-z0-9-]+)|([A-Za-z0-9-]+)[-_]DN)$",
    re.IGNORECASE,
)
_HGVS_MUTANT_RE = re.compile(
    r"^([A-Za-z0-9-]+)[-_](p\.[A-Za-z]+(?:\d+)[A-Za-z]+|\([A-Za-z]+\d+[A-Za-z]+\))$",
    re.IGNORECASE,
)
_EPIGENOME_RE = re.compile(
    r"^(?:(dCas9[-_]?(?:KRAB(?:[-_]MeCP2)?|TET1|DNMT3[AL]|p300|VPR|VP64|SAM|SunTag|LSD1|HDAC3)|CRISPR(?:off|on))[-_]([A-Za-z0-9-]+)|([A-Za-z0-9-]+)[-_](dCas9[-_]?(?:KRAB(?:[-_]MeCP2)?|TET1|DNMT3[AL]|p300|VPR|VP64|SAM|SunTag|LSD1|HDAC3)|CRISPR(?:off|on)))$",
    re.IGNORECASE,
)
_CHEMOGENETICS_RE = re.compile(
    r"^(?:(hM3Dq|hM4Di|hM3D\(Gq\)|hM4D\(Gi\)|KORD|PSAM[-_]GlyR|PSAM[-_]5HT3)(?:[-_]([A-Za-z0-9-]+))?|([A-Za-z0-9-]+)[-_](hM3Dq|hM4Di|KORD))$",
    re.IGNORECASE,
)
_OPTOGENETICS_RE = re.compile(
    r"^(?:(ChR2(?:\([A-Za-z0-9]+\))?|NpHR(?:\d(?:\.\d)?)?|eNpHR3\.0|Arch[T]?|Chrimson[R]?|GGaMP\d*|miniSOG)(?:[-_]([A-Za-z0-9-]+))?|([A-Za-z0-9-]+)[-_](ChR2|NpHR|Arch|Chrimson))$",
    re.IGNORECASE,
)
_DELIVERY_PREFIX_RE = re.compile(
    r"^(?P<carrier>LNP|AAV[-_]?(?:PHP\.[a-z0-9]+|DJ|Retro|\d+)?|Exosome|EV|Lentivirus|Lenti|Adenovirus|Lipofectamine(?:\s*3000|\s*2000)?|PEI)"
    r"(?:[-_/:](?P<cargo>.+))?$",
    re.IGNORECASE,
)
_PROTAC_RE = re.compile(
    r"^(?P<target>[A-Za-z0-9-]+)[-_](?:PROTAC|Degrader)(?:\((?P<e3>[A-Za-z0-9]+)\))?$",
    re.IGNORECASE,
)
_SEQUENTIAL_ARROW_RE = re.compile(
    r"\s*(?<!<)(?:->|-->|=>)(?:\s*(?:followed\s+by|then))?\s*|\s*(?:;\s*then\b|/\s*followed\s+by\b|\bfollowed\s+by\b|\bthen\b)\s*",
    re.IGNORECASE,
)
_PULSE_CHASE_RE = re.compile(
    r"^\s*pulse[:\s]+(?P<pulse>.+?)\s*(?:->|-->|=>|;\s*then|;\s*chase|\bfollowed\s+by\b|\bthen\b|,?\s*\bchase[:\s]+)\s*(?P<chase>.+?)\s*$",
    re.IGNORECASE,
)


def detect_genetic_perturbation(name: str, config: PreprocessConfig = DEFAULT_CONFIG) -> tuple[str, str] | None:
    """Identify genetic perturbation tags (CRISPR KO / dual CRISPR / base editing / prime editing / epigenetic / chemogenetic / optogenetic / degron / dominant negative / shRNA / overexpression / mutation).

    Returns (target_gene, action) or None.
    If target is a negative control (e.g. sgNT -> target='NT'), returns None to let detect_control handle it.
    """
    t = name.strip()

    # 1) Dual-targeting CRISPR (Dual sgRNA, e.g. sgTP53+sgBRCA1, sgTP53/sgBRCA1, sgTP53_sgBRCA1)
    m_dual = _DUAL_CRISPR_RE.match(t)
    if m_dual:
        g1, g2 = m_dual.group(1).strip(), m_dual.group(2).strip()
        if g1.lower() in config.genetic_control_targets and g2.lower() in config.genetic_control_targets:
            return None
        return f"{g1}+{g2}", "dual_KO"

    # 2) Base editing and prime editing (ABE_TP53, CBE_BRCA1, PE3_HEK3)
    m_bp = _BASE_PRIME_RE.match(t)
    if m_bp:
        tag = (m_bp.group(1) or m_bp.group(4)).upper()
        target = (m_bp.group(2) or m_bp.group(3)).strip()
        if target.lower() in config.genetic_control_targets:
            return None
        return target, f"editing:{tag}"

    # 3) Epigenetic editing systems (dCas9-KRAB, dCas9-KRAB-MeCP2, dCas9-TET1, CRISPRoff, CRISPRon)
    m_epi = _EPIGENOME_RE.match(t)
    if m_epi:
        eff = (m_epi.group(1) or m_epi.group(4)).strip()
        target = (m_epi.group(2) or m_epi.group(3)).strip()
        if target.lower() in config.genetic_control_targets:
            return None
        return target, f"epigenetic_edit:{eff}"

    # 4) Chemogenetic receptors / DREADD (hM3Dq, hM4Di, KORD, PSAM)
    m_chemo = _CHEMOGENETICS_RE.match(t)
    if m_chemo:
        rec = (m_chemo.group(1) or m_chemo.group(4)).strip()
        target = (m_chemo.group(2) or m_chemo.group(3)) or rec
        return target.strip(), f"chemogenetics:{rec}"

    # 5) Optogenetic channels and probes (ChR2, NpHR, Arch, Chrimson)
    m_opto = _OPTOGENETICS_RE.match(t)
    if m_opto:
        opto = (m_opto.group(1) or m_opto.group(4)).strip()
        target = (m_opto.group(2) or m_opto.group(3)) or opto
        return target.strip(), f"optogenetics:{opto}"

    # 6) Targeted protein degradation / Degron (BRD4_dTAG, SMARCA4_PROTAC, MYC_AID)
    m_deg = _DEGRON_RE.match(t)
    if m_deg:
        target = (m_deg.group(1) or m_deg.group(4)).strip()
        tag = (m_deg.group(2) or m_deg.group(3)).upper()
        if target.lower() in config.genetic_control_targets:
            return None
        return target, f"degradation:{tag}"

    # 7) Dominant negative mutations (RAS_DN, DN_TP53)
    m_dn = _DOMINANT_NEGATIVE_RE.match(t)
    if m_dn:
        target = (m_dn.group(1) or m_dn.group(2)).strip()
        if target.lower() in config.genetic_control_targets:
            return None
        return target, "dominant_negative"

    # 8) HGVS standard mutations (EGFR_p.T790M, BRAF_p.V600E)
    m_hgvs = _HGVS_MUTANT_RE.match(t)
    if m_hgvs:
        target = m_hgvs.group(1).strip()
        mut = m_hgvs.group(2).strip()
        if target.lower() in config.genetic_control_targets:
            return None
        return target, f"mutant:{mut}"

    # 9) Prefix format: sgTP53, CRISPRi_MYC, shEGFR
    m = _GENETIC_PREFIX_RE.match(t)
    if m:
        pfx = m.group("pfx")
        sep = m.group("sep")
        target = m.group("target").strip()
        if not _is_valid_genetic_target(target, pfx, sep, t):
            return None
        if target.lower() in config.genetic_control_targets:
            return None
        lower_t = t.lower()
        if lower_t.startswith("crispri"):
            action = "CRISPRi"
        elif lower_t.startswith("crispra"):
            action = "CRISPRa"
        elif lower_t.startswith("sh"):
            action = "KD"
        else:
            action = "KO"
        return target, action

    # 10) Suffix format: TP53_KO, MYC_OE, BRCA1_sh2 (matched dynamically via config._genetic_suffix_re)
    m = config._genetic_suffix_re.match(t)
    if m:
        target = m.group(1).strip()
        if target.lower() in config.genetic_control_targets:
            return None
        tag = m.group(2).strip()
        upper_tag = tag.upper()
        if upper_tag.startswith("SG"):
            action = "KO"
        elif upper_tag.startswith("SH"):
            action = "KD"
        else:
            act_meta = config.genetic_actions.get(tag.lower())
            action = act_meta["canonical_action"] if act_meta else upper_tag
        return target, action

    # 11) Mutation format: EGFR_T790M, BCR-ABL_A269V
    m = _GENETIC_MUTANT_RE.match(t)
    if m:
        target = m.group(1).strip()
        mutation = m.group(2).strip()
        return target, f"mutant:{mutation}"

    return None


def detect_delivery_vehicle(name: str, config: PreprocessConfig = DEFAULT_CONFIG) -> dict | None:
    """Identify nanoparticle delivery systems, LNP formulations, AAV serotypes, and exosome carriers.

    Supports standalone carrier recognition (e.g. "LNP", "MC3", "AAV9", "Lipofectamine 3000")
    and carrier+cargo complex identification (e.g. "LNP-mRNA", "AAV-PHP.eB-Cre", "Exosome-miR-21").
    """
    t = name.strip()
    tok = t.lower()

    # 1. Exact vocabulary hit
    if tok in config.delivery_vehicles:
        res = dict(config.delivery_vehicles[tok])
        res["cargo"] = []
        return res

    # 2. Complex carrier-cargo regex recognition
    m = _DELIVERY_PREFIX_RE.match(t)
    if m:
        c_str = m.group("carrier").strip().lower()
        cargo_str = m.group("cargo").strip() if m.group("cargo") else None

        info = config.delivery_vehicles.get(c_str)
        if not info:
            c_clean = c_str.replace("_", "-")
            info = config.delivery_vehicles.get(c_clean)

        if info:
            res = dict(info)
            res["cargo"] = [cargo_str] if cargo_str else []
            return res

        if c_str.startswith("aav"):
            return {
                "canonical_name": c_str.upper(),
                "carrier_type": "aav",
                "serotype": c_str.upper(),
                "formulation": [],
                "tropism": None,
                "cargo": [cargo_str] if cargo_str else [],
                "description": f"AAV vector ({c_str.upper()})",
            }
        if c_str.startswith("lnp"):
            return {
                "canonical_name": "Lipid Nanoparticle",
                "carrier_type": "lnp",
                "serotype": None,
                "formulation": [],
                "tropism": "liver",
                "cargo": [cargo_str] if cargo_str else [],
                "description": "Lipid nanoparticle delivery system",
            }
    return None


def detect_protac(name: str, config: PreprocessConfig = DEFAULT_CONFIG) -> dict | None:
    """Identify targeted protein degradation molecules (PROTACs) and molecular glues."""
    tok = name.strip().lower()
    if tok in config.protac_e3_ligands:
        return dict(config.protac_e3_ligands[tok])
    m = _PROTAC_RE.match(name.strip())
    if m:
        target = m.group("target").strip()
        e3 = m.group("e3").strip() if m.group("e3") else "CRBN/VHL"
        return {
            "canonical_name": name.strip(),
            "e3_ligase": e3,
            "e3_ligand": "",
            "target_protein": target,
            "is_molecular_glue": False,
            "description": f"PROTAC targeting {target} via {e3}",
        }
    return None


def split_sequential_regime(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> list[dict] | None:
    """Detect and split sequential regimens, pulse-chase, and washout experiments.

    Returns stage list:
    [
        {
            "stage_index": int,
            "action_type": str,   # pretreat, treatment, pulse, chase, washout, recovery
        "text": str,          # Pharmacological description after action prefix removal
            "dose": float | None,
            "unit": str | None,
            "duration": float | None,
            "duration_unit": str | None,
            "raw_stage": str,
        },
        ...
    ]
    Returns None if not a temporal protocol.
    """
    t = text.strip()
    if t.lower().startswith("inchi=") or t.lower().startswith("inchi/"):
        return None
    # Protect reversible biochemical reaction equations (containing <=> or <->) from being split by =>
    if "<=>" in t or "<->" in t:
        return None
    from .identifiers import detect_identifier
    if detect_identifier(t) is not None:
        return None

    # 1. Pulse-chase specific match (Pulse ... chase ...)
    m_pc = _PULSE_CHASE_RE.match(t)
    if m_pc:
        p_str = m_pc.group("pulse").strip()
        c_str = m_pc.group("chase").strip()
        c_str = re.sub(r"^(?:chase(?:\s+in|\s+with|:)?)\s*", "", c_str, flags=re.I).strip()

        stages = []
        c_name, d, u, dur, dur_u = parse_spatiotemporal(p_str, config)
        p_text = re.sub(r"^(?:pulse(?:\s+with|:)?)\s*", "", c_name or p_str, flags=re.I).strip()
        stages.append({
            "stage_index": 0,
            "action_type": "pulse",
            "text": p_text,
            "dose": d,
            "unit": u,
            "duration": dur,
            "duration_unit": dur_u,
            "raw_stage": p_str,
        })
        c_name2, d2, u2, dur2, dur_u2 = parse_spatiotemporal(c_str, config)
        c_text = re.sub(r"^(?:chase(?:\s+in|\s+with|:)?)\s*", "", c_name2 or c_str, flags=re.I).strip()
        stages.append({
            "stage_index": 1,
            "action_type": "chase",
            "text": c_text,
            "dose": d2,
            "unit": u2,
            "duration": dur2,
            "duration_unit": dur_u2,
            "raw_stage": c_str,
        })
        return stages

    # 2. Arrow and temporal connector matching (->, followed by, then)
    if _SEQUENTIAL_ARROW_RE.search(t):
        raw_parts = [p.strip() for p in _SEQUENTIAL_ARROW_RE.split(t) if p.strip()]
        if len(raw_parts) >= 2:
            stages = []
            for i, rp in enumerate(raw_parts):
                rp_clean = re.sub(r"^stage\s*\d+:\s*", "", rp, flags=re.I).strip()
                action = "treatment"
                low = rp_clean.lower()
                if low.startswith("pre-treat") or low.startswith("pretreat") or "pre-treatment with" in low or "pretreatment with" in low:
                    action = "pretreat"
                    rp_clean = re.sub(r"^(?:pre[-_]?treatment(?:\s+with)?|pretreat(?:\s+with)?)\s*", "", rp_clean, flags=re.I).strip()
                elif low.startswith("pulse"):
                    action = "pulse"
                    rp_clean = re.sub(r"^pulse(?:\s+with|:)?\s*", "", rp_clean, flags=re.I).strip()
                elif low.startswith("chase"):
                    action = "chase"
                    rp_clean = re.sub(r"^chase(?:\s+in|\s+with|:)?\s*", "", rp_clean, flags=re.I).strip()
                elif low.startswith("washout") or low == "wash":
                    action = "washout"
                    rp_clean = re.sub(r"^wash(?:out)?(?:\s+with|:)?\s*", "", rp_clean, flags=re.I).strip()
                    if not rp_clean:
                        rp_clean = "washout"
                elif low.startswith("recovery") or low.startswith("rest"):
                    action = "recovery"
                    rp_clean = re.sub(r"^(?:recovery|rest)(?:\s+in|\s+with|:)?\s*", "", rp_clean, flags=re.I).strip()

                c_name, d, u, dur, dur_u = parse_spatiotemporal(rp_clean, config)
                stages.append({
                    "stage_index": i,
                    "action_type": action,
                    "text": c_name or rp_clean,
                    "dose": d,
                    "unit": u,
                    "duration": dur,
                    "duration_unit": dur_u,
                    "raw_stage": rp,
                })
            return stages

    return None


def detect_biologic(name: str, config: PreprocessConfig = DEFAULT_CONFIG) -> dict | None:
    """Identify biologics, cytokines, recombinant proteins, and monoclonal antibodies (based on biologics_tokens and biologic_suffixes)."""
    tok = name.strip().lower()
    if tok in config.biologics_tokens:
        return config.biologics_tokens[tok]
    # Rule-based identification of biologic suffixes (-mab, -cept, -kin, -stim, -ase etc.)
    if tok.replace("-", "").isalpha() and len(tok) >= 5:
        for suffix, category in config.biologic_suffixes:
            if len(tok) >= len(suffix) + 3 and tok.endswith(suffix):
                return {
                    "canonical_name": name.strip().capitalize(),
                    "category": category,
                    "hgnc_or_uniprot": "",
                }
    return None


_TEMP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*°?[CF]\b", re.IGNORECASE)
_O2_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*O2\b", re.IGNORECASE)
_RAD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:Gy|rad|cGy)\b", re.IGNORECASE)
_PH_RE = re.compile(r"\bpH\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?", re.IGNORECASE)
_OSM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mOsm|Osm)\b", re.IGNORECASE)
_SHEAR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:dyn/cm2|dyn/cm²|Pa|mPa)\b", re.IGNORECASE)
_PRESSURE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:atm|bar|kPa|mmHg|torr)\b", re.IGNORECASE)


def _extract_env_param(name: str, res: dict) -> None:
    for regex in (_TEMP_RE, _O2_RE, _RAD_RE, _PH_RE, _OSM_RE, _SHEAR_RE, _PRESSURE_RE):
        m = regex.search(name)
        if m:
            res["parameter"] = m.group(0)
            break


def detect_environmental(name: str, config: PreprocessConfig = DEFAULT_CONFIG) -> dict | None:
    """Identify physical and environmental stress perturbations."""
    tok = name.strip().lower()
    if tok in config.environmental_tokens:
        meta = dict(config.environmental_tokens[tok])
        _extract_env_param(name, meta)
        return meta

    m_ph = _PH_RE.search(name)
    if m_ph:
        val = float(m_ph.group(1))
        ph_cat = "acidic" if val < 7.0 else ("alkaline" if val > 7.4 else "neutral")
        return {
            "canonical_name": f"pH {val} stress",
            "category": "ph",
            "parameter": f"pH {val} ({ph_cat})",
        }

    # Traverse in descending token length using strict non-alphanumeric boundaries to avoid partial matches (e.g. uva in sodium pyruvate / fluvastatin)
    for env_token, meta in sorted(config.environmental_tokens.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = rf"(?<![a-zA-Z0-9]){re.escape(env_token)}(?![a-zA-Z0-9])"
        if re.search(pattern, tok):
            res = dict(meta)
            _extract_env_param(name, res)
            return res
    return None


def detect_control(name: str, config: PreprocessConfig = DEFAULT_CONFIG) -> bool:
    """Control and vehicle detection: lowercase whole-word match, 'negative ' prefix,
    genetic control target (with sg/sh prefix or _ctrl suffix), or all tokens present in vocabulary.
    """
    token = " ".join(name.strip().lower().split())
    if not token:
        return False
    # Protect synthetic biology effectors (optogenetics/chemogenetics/epigenetics) from colliding with controls (e.g. ChR2 vs chr2)
    stripped = name.strip()
    if _EPIGENOME_RE.match(stripped) or _CHEMOGENETICS_RE.match(stripped):
        return False
    if _OPTOGENETICS_RE.match(stripped):
        if stripped.lower() not in ("chr2", "chr8", "chr12") or not stripped.islower():
            return False
    if token in config.all_control_tokens:
        return True
    # Underscore normalized whole-word matching (EMPTY_VECTOR -> "empty vector", common in LINCS)
    clean_tok = token.replace("_", " ")
    if clean_tok in config.all_control_tokens:
        return True
    if token.startswith("negative ") or clean_tok.startswith("negative "):
        return True

    # Strip trailing numeric suffix (e.g. "neg_control_1" -> "neg control", "control_01" -> "control")
    tok_no_num = re.sub(r"[\s_-]+\d+$", "", token).replace("_", " ").strip()
    if tok_no_num in config.all_control_tokens:
        return True
    if tok_no_num.startswith("negative "):
        return True

    # Recognize common abbreviated prefixes: "neg control", "neg ctrl", "non-targeting", "nontargeting"
    if clean_tok.startswith("neg ") or tok_no_num.startswith("neg "):
        rem = (tok_no_num if tok_no_num.startswith("neg ") else clean_tok)[4:].strip()
        if rem in ("control", "ctrl") or rem in config.all_control_tokens:
            return True
    if clean_tok in ("non-targeting", "nontargeting", "non targeting") or tok_no_num in ("non-targeting", "nontargeting", "non targeting"):
        return True

    # Recognize cell line or batch prefix/suffix controls (e.g. "RPE1_control", "K562_ctrl", "A549_DMSO")
    if "_" in token:
        subparts = token.split("_")
        last_p = subparts[-1].strip()
        first_p = subparts[0].strip()
        ctrl_words = ("control", "ctrl", "untrt", "untreated", "vehicle", "dmso", "mock", "blank")
        known_hosts = ("rpe1", "k562", "mcf7", "a549", "hela", "hepg2", "pc9", "hct116", "jurkat")
        if first_p in known_hosts and (last_p in ctrl_words or last_p in config.all_control_tokens):
            return True
        if last_p in known_hosts and (first_p in ctrl_words or first_p in config.all_control_tokens):
            return True

    # Genetic control prefix and suffix recognition (e.g. sgCCR5, sgHPRT1, sgRosa26, shControl, AAVS1_sg)
    m_pfx = _GENETIC_PREFIX_RE.match(stripped)
    if m_pfx:
        tgt = m_pfx.group("target").strip().lower()
        if tgt in config.genetic_control_targets:
            return True
    m_sfx = config._genetic_suffix_re.match(stripped)
    if m_sfx:
        tgt = m_sfx.group(1).strip().lower()
        if tgt in config.genetic_control_targets:
            return True

    words = token.replace("_", " ").split()
    return len(words) >= 1 and all(w in config.all_control_tokens for w in words)


def remove_trailing_parens(name: str) -> str:
    """Repeatedly strip trailing parenthesis aliases."""
    prev = None
    while prev != name:
        prev = name
        name = _TRAILING_PAREN_RE.sub("", name).strip()
    return name


def strip_leading_parens(name: str) -> str:
    """Strip leading stereochemistry/configuration parenthesis prefixes (e.g. "(+)-JQ1" -> "JQ1").

    - Stereochemistry markers ((+)- / (S)- / (R)- / (E)- / (Z)- / cis-/trans- etc.) stripped;
    - Fully enclosed strings ("(SALINE)" / "(4-carbomethoxy fentanyl)") unnested retaining contents;
    - Other parentheses (containing aliases etc.) preserved.
    """
    prev = None
    while prev != name:
        prev = name
        name = _LEADING_STEREO_RE.sub("", name).strip()
    m = re.match(r"^\(\s*([^()]*?)\s*\)", name)
    if m:
        inner, rest = m.group(1).strip(), name[m.end():].strip()
        if not rest:
            name = inner  # Fully enclosed: unnest and retain (SALINE) -> SALINE
        else:
            name = rest.lstrip("-").strip()  # Prefix modifier: (Gal)Glc -> Glc; fallback trailing hyphen
    return name


_COVALENT_PHOSPHATE_PREFIX = re.compile(
    r"^(?:adenosine|guanosine|cytidine|uridine|thymidine|inosine|"
    r"pyridoxal|creatine|carbamoyl|thiamine|isopentenyl|glucose|"
    r"fructose|ribose|glycerol|glyceraldehyde|dihydroxyacetone|"
    r"sphingosine|inositol|trehalose|glucosamine|mannose|galactose|"
    r"sedoheptulose|phosphoenolpyruvate)\b",
    re.IGNORECASE,
)


def strip_salt(name: str, config: PreprocessConfig = DEFAULT_CONFIG) -> str:
    """Salt stripping (string layer): strip salt suffixes + prefix metal salts, retaining parent molecule name.

    Doxorubicin HCl -> Doxorubicin; Sodium Phenylbutyrate -> Phenylbutyrate.
    Metal halides/carbonates (Sodium Chloride / Lithium Carbonate) are inorganic compound names;
    if stripping leaves an inorganic suffix, fall back and preserve intact.
    If salt stripping produces an empty string, fallback to original word (Lactate/Acetate/Citrate can be parents).
    Inorganic salt drugs (Magnesium Sulfate / Zinc Sulfate) leaving metal cations fall back and preserve intact.
    Covalent high-energy phosphate metabolites (Adenosine diphosphate / Pyridoxal phosphate) preserve covalent structure.
    """
    if _COVALENT_PHOSPHATE_PREFIX.match(name) and re.search(
        r"\b(?:mono|di|tri|pyro|poly|bis)?phosph(?:ate|ite)\b", name, re.IGNORECASE
    ):
        return name

    stripped = config._salt_re.sub("", name).strip()
    if stripped:
        name = stripped
    stripped = config._salt_prefix_re.sub("", name).strip()
    if stripped and config._inorganic_re.fullmatch(stripped):
            pass  # Sodium Chloride -> "Chloride": inorganic compound, fallback and preserve
    elif stripped:
        name = stripped
    return name


def strip_vendor_prefix(name: str, config: PreprocessConfig = DEFAULT_CONFIG) -> str:
    """Strip vendor prefixes ("Selleck S1001" -> "S1001").

    Requires vendor + space + code-like token (containing digits) to avoid drug name false positives.
    Pharma investigational codes (BMS-182874 / GSK-1059615) are not vendor prefixes and are preserved.
    """
    return config._vendor_re.sub("", name).strip()


def strip_formulation(name: str, config: PreprocessConfig = DEFAULT_CONFIG) -> str:
    """Strip formulation/manufacturer/regulatory suffixes ("Diamox sr" -> "Diamox").

    Requires standalone token (preceded by whitespace/digits) situated at trailing position.
    Combination abbreviations like hct are not in suffix table (remain honestly unmatched).
    """
    return config._formulation_re.sub("", name).strip()


def clean_name(name: str, config: PreprocessConfig = DEFAULT_CONFIG) -> str:
    """Full cleaning pipeline: vendor prefix -> alternating parens/salt/formulation stripping -> special character cleanup."""
    name = name.strip()
    name = strip_vendor_prefix(name, config)
    prev = None
    iters = 0
    while prev != name and iters < config.max_clean_iters:
        prev = name
        name = remove_trailing_parens(name)
        name = strip_leading_parens(name)
        name = strip_salt(name, config)
        name = strip_formulation(name, config)
        iters += 1
    name = _TRAILING_JUNK_RE.sub("", name).strip()
    return name


def norm_key(name: str) -> str:
    """Normalization key for fuzzy matching: lowercase + alphanumeric characters only."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


_COMBO_RATIO_RE = re.compile(
    r"\s*\(?\b(\d+(?:\.\d+)?:\d+(?:\.\d+)?)\s*(?:molar|mass|vol|weight)?\s*ratio\)?",
    re.IGNORECASE,
)
_REACTION_ARROW_RE = re.compile(r"(?:<=>|<->|=>|->)", re.IGNORECASE)
_EC_ENZYME_SUFFIX_RE = re.compile(
    r"(?:oxidoreductase|transferase|hydrolase|lyase|isomerase|ligase|translocase|synthase|synthetase|kinase|dehydrogenase)\b",
    re.IGNORECASE,
)
_UNIT_DENOMINATOR_RE = re.compile(
    r"^(?:ml|l|ul|nl|pl|kg|g|mg|ug|µg|ng|pg|day|h|hr|min|s|sec|well|cell|plate)$",
    re.IGNORECASE,
)


def split_combo(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> list[str] | None:
    """Detect and split combination drug regimens (cocktails).

    Delimiters from combo_separators config (evaluated in priority order; first match returns).
    Ratio prefixes ("1:5 ") and ratio modifiers ("(1:2 molar ratio)") stripped first to avoid pseudo-components.
    InChI strings and prefixed identifiers (SMILES:, InChIKey: etc.) strictly protected against splitting.
    Symbol delimiters (+ / :) pass through fragment quality gates: balanced parentheses, length >= 2, non-numeric,
    slash context excludes stereochemical words (cis/exo); " + " (with spaces) treated as explicit combination.
    Note: Single underscore "_" is not split here; handled by split_underscore.
    """
    t = text.strip()
    if t.lower().startswith("inchi=") or t.lower().startswith("inchi/"):
        return None
    if _RATIO_PREFIX_RE.match(t):
        t = _RATIO_PREFIX_RE.sub("", t).strip()
    if _COMBO_RATIO_RE.search(t):
        t = _COMBO_RATIO_RE.sub("", t).strip()
    for sep, _prio in config.combo_separators:
        if sep == " and ":
            if _WORD_AND_RE.search(t):
                parts = [p.strip() for p in _WORD_AND_RE.split(t) if p.strip()]
                if len(parts) >= 2:
                    return parts
            continue
        if sep in t:
            strong = sep == "+" and re.search(r"\s\+\s", t) is not None
            parts = [p.strip() for p in re.split(rf"\s*{re.escape(sep)}\s*", t) if p.strip()]
            if len(parts) >= 2 and (strong or _symbol_quality_ok(parts, sep, config)):
                return parts
    return None


def _symbol_quality_ok(parts: list[str], sep: str, config: PreprocessConfig = DEFAULT_CONFIG) -> bool:
    """Fragment quality gate for symbol delimiters (+ / :). Returns False if string should not be split."""
    if sep == ":" and parts[0].lower() in {"smiles", "cas", "inchikey", "cid", "chembl", "drugbank", "ensembl", "uniprot"}:
        return False
    # Protect EC enzyme systematic names (e.g. methanol:NAD+ oxidoreductase) from being split by colon or plus
    if any(_EC_ENZYME_SUFFIX_RE.search(p) for p in parts):
        return False
    # Protect slash-containing composite concentration units (e.g. mg/mL, ug/ml, mg/day) from being split by slash
    if sep == "/" and any(_UNIT_DENOMINATOR_RE.match(p.strip()) for p in parts):
        return False
    for p in parts:
        if p.count("(") != p.count(")"):
            return False            # Delimiter inside parentheses ((NAD+) / (1+) / 2,3-(+)-)
        if p.isdigit():
            return False            # Numeric pattern (18:1, CI 45350:1, 1 in n+1)
        if len(p) < 2:
            if not (p.isalpha() and p.isupper()):
                return False        # Lowercase single letter = specification code (p/g); uppercase single letter = abbreviation (A/B) allowed
        if sep == "/" and (p.lower() in config.stereo_tokens
                           or config._stereo_prefix_re.match(p.lower())):
            return False            # Stereochemistry words like cis/exo (2,3-cis/exo-)
    return True


def split_underscore(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> list[str] | None:
    """Attempt to split on single underscore when no dose information is present (e.g. vemurafenib_cobimetinib).

    Protects parentheses: if any fragment has unbalanced parentheses (e.g. "Trametinib (DMSO_TF solvate)"
    splitting into "(DMSO"), underscore is inside parentheses, abort split.
    Gene mutation/perturbation tags (EGFR_T790M, BCR-ABL_A269V, PTEN_WT, TP53_sg1, dCas9-KRAB_MYC):
    single perturbation rather than combination, reject splitting directly.
    Nanocarrier/serotype tags (AAV_PHP.eB, LNP_mRNA): reject splitting directly.
    """
    if "_" in text:
        if _GENE_MUTATION_UNDERSCORE_RE.fullmatch(text):
            return None
        if detect_genetic_perturbation(text, config) is not None:
            return None
        if detect_delivery_vehicle(text, config) is not None:
            return None
        parts = [p.strip() for p in text.split("_") if p.strip()]
        if len(parts) >= 2:
            for p in parts:
                if p.count("(") != p.count(")"):
                    return None  # Underscore inside parentheses, not a combination delimiter
            return parts
    return None
