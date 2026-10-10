"""Structure-level normalization (Phase 2, optional rdkit dependency).

Desalting and parent primary key:
  - String-level desalting (preprocess.strip_salt) only processes name suffixes;
  - Structure-level desalting uses RDKit MolStandardize to normalize SMILES (e.g. "Doxorubicin HCl")
    to the parent neutral molecule, outputting parent_inchikey as a cross-dataset clustering key.

When rdkit is not installed, all functions return None / empty mapping (silent degradation).
"""
from __future__ import annotations

_INCHIKEY_PREFIX_LEN = 14  # connectivity layer: same parent / salt form share prefix


def _rdkit():
    """Lazy import rdkit; returns None if not installed."""
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import inchi  # noqa: F401
        from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: F401
        return (Chem, inchi, rdMolStandardize)
    except ImportError:
        return None


def parent_inchikey(smiles: str | None) -> str | None:
    """Calculate desalted parent InChIKey from SMILES (rdkit optional dependency)."""
    if not smiles:
        return None
    mods = _rdkit()
    if mods is None:
        return None
    Chem, inchi, rdMolStandardize = mods
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        parent = rdMolStandardize.GetParentMol(mol)
        if parent is None:
            return None
        return inchi.MolToInchiKey(parent)
    except Exception:  # noqa: BLE001 - return None on parse/calc failure
        return None


def cluster_by_parent(mapping, key: str = "canonical_smiles") -> dict[str, list[str]]:
    """Cluster inputs by parent InChIKey prefix (first 14 chars, aggregating salts/stereoisomers).

    Parameters:
      mapping: list[AlignResult] (or any object with parsed_name / canonical_smiles).
      key: attribute name to extract SMILES (default canonical_smiles).

    Returns:
      dict: {parent_inchikey[:14]: [input1, input2, ...]}; grouped into "" when SMILES is missing
      or rdkit is unavailable (preserves data for auditing).
    """
    groups: dict[str, list[str]] = {}
    for rec in mapping:
        entity = getattr(rec, "entity", None)
        if entity is None:
            continue
        smiles = getattr(entity, key, None)
        pik = parent_inchikey(smiles) if smiles else None
        group_key = pik[:14] if pik else ""
        groups.setdefault(group_key, []).append(rec.input)
    return groups
