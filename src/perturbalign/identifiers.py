r"""Chemical structure and authoritative registry accession detector (zero-dependency, strict validation).

Supported direct pass-through identifier types:
- InChIKey: Universal compound structure hash (14-10-1 format, supports InChIKey= prefix)
- InChI: International Chemical Identifier (InChI=1S/...)
- CAS Number: Chemical Abstracts Service registry number (with strict modulo-10 checksum validation, supports CAS: prefix)
- ChEMBL ID: EMBL-EBI compound/biologic accession (CHEMBL\d+, supports CHEMBL: prefix)
- PubChem CID: NCBI Compound numeric ID (CID: 12345 / CID12345)
- SMILES: Simplified Molecular Input Line Entry System (strict syntax and chemical feature heuristics, supports SMILES: prefix)
- DrugBank ID: DrugBank accession (DB\d{5}, supports DrugBank: prefix)
- Ensembl ID: Human gene Ensembl accession (ENSG\d{11}, supports Ensembl: prefix)
- UniProt Accession: Protein knowledgebase accession (supports UniProt: prefix)
"""
from __future__ import annotations

import re

_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z0-9]$")
_INCHIKEY_PREFIX_RE = re.compile(r"^(?:inchikey[:=\s]+)?([A-Z]{14}-[A-Z]{10}-[A-Z0-9])$", re.IGNORECASE)
_INCHI_RE = re.compile(r"^InChI=1S?/", re.IGNORECASE)
_CAS_RE = re.compile(r"^(\d{2,7})-(\d{2})-(\d)$")
_CAS_PREFIX_RE = re.compile(r"^(?:cas[:=\s]+)?(\d{2,7}-\d{2}-\d)$", re.IGNORECASE)
_CHEMBL_RE = re.compile(r"^(?:chembl[:=\s]+)?(CHEMBL\d+)$", re.IGNORECASE)
_CID_PREFIX_RE = re.compile(r"^(?:pubchem[:=\s]+|CID[:=\s]+|CID)(\d+)$", re.IGNORECASE)
_DRUGBANK_RE = re.compile(r"^(?:drugbank[:=\s]+)?(DB\d{5})$", re.IGNORECASE)
_ENSEMBL_RE = re.compile(r"^(?:ensembl[:=\s]+)?(ENSG\d{11})$", re.IGNORECASE)
_UNIPROT_RE = re.compile(
    r"^(?:uniprot[:=\s]+)?([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$",
    re.IGNORECASE,
)
_SMILES_PREFIX_RE = re.compile(r"^smiles[:=\s]+(\S+)$", re.IGNORECASE)

# Valid SMILES character set
_SMILES_VALID_CHARS = set("BCNOFPSIKAlBrClNaeHgLiZnSeTe#=-+()[]/@\\0123456789%bcdefghiklmnoprstuvwxyz")
_SMILES_STRUCT_INDICATORS = set("=#@[]\\/0123456789")


def is_inchikey(text: str) -> bool:
    """Determine whether input is a standard InChIKey hash (supports InChIKey= / InChIKey: prefix)."""
    t = text.strip()
    return bool(_INCHIKEY_PREFIX_RE.match(t))


def is_inchi(text: str) -> bool:
    """Determine whether input is an InChI string."""
    t = text.strip()
    return bool(_INCHI_RE.match(t))


def is_cas_number(text: str) -> bool:
    """Determine whether input is a valid CAS Registry Number with modulo-10 checksum (supports CAS: prefix).

    Algorithm rules:
    CAS format is R-Q-C, where R is 2-7 digits, Q is 2 digits, and C is a 1-digit checksum.
    Reversing R and Q digits, each i-th digit (1-indexed) is multiplied by i; sum modulo 10 must equal C.
    E.g. Aspirin CAS: 50-78-2 -> (8*1 + 7*2 + 0*3 + 5*4) = 42 -> 42 % 10 = 2.
    """
    t = text.strip()
    m_p = _CAS_PREFIX_RE.match(t)
    if not m_p:
        return False
    cas_core = m_p.group(1)
    m = _CAS_RE.match(cas_core)
    if not m:
        return False
    r, q, c = m.groups()
    digits = r + q
    check = sum(int(digit) * weight for weight, digit in enumerate(reversed(digits), start=1)) % 10
    return check == int(c)


def is_chembl_id(text: str) -> bool:
    """Determine whether input is a ChEMBL ID (e.g. CHEMBL25, supports CHEMBL: prefix)."""
    t = text.strip()
    return bool(_CHEMBL_RE.match(t))


def is_drugbank_id(text: str) -> bool:
    """Determine whether input is a DrugBank ID (e.g. DB00316, supports DrugBank: prefix)."""
    t = text.strip()
    return bool(_DRUGBANK_RE.match(t))


def is_ensembl_id(text: str) -> bool:
    """Determine whether input is an Ensembl Gene ID (e.g. ENSG00000141510, supports Ensembl: prefix)."""
    t = text.strip()
    return bool(_ENSEMBL_RE.match(t))


def is_uniprot_accession(text: str) -> bool:
    """Determine whether input is a UniProt Accession (e.g. P04637, P05231, supports UniProt: prefix)."""
    t = text.strip()
    return bool(_UNIPROT_RE.match(t))


def parse_pubchem_cid(text: str) -> int | None:
    """Extract PubChem CID from input matching 'CID: 2244', 'CID2244', 'PubChem: 2244'."""
    t = text.strip()
    m = _CID_PREFIX_RE.match(t)
    if m:
        return int(m.group(1))
    return None


_SMILES_TOKEN_RE = re.compile(
    r"(\[[^\]]+\]"                         # bracketed atom [C@H], [Na+], etc.
    r"|Cl|Br"                              # two-letter organic subset
    r"|[BCNOPSFIbcnops]"                   # single-letter organic subset
    r"|%[0-9]{2}"                          # two-digit ring closure %10
    r"|[1-9]"                              # single-digit ring closure 1-9
    r"|[-=#$:/\\.]"                        # bonds and disconnected dot
    r"|[()]"                               # branch open/close
    r"|@@?"                                # tetrahedral chirality
    r")"
)


def is_smiles(text: str) -> bool:
    """Determine whether input strictly conforms to OpenSMILES syntax (supports SMILES: prefix).

    Strict chemical structure requirements:
    1. No whitespace, no empty branches `()`, and balanced parentheses and square brackets;
    2. Strict SMILES tokenization (organic subset, bracketed atoms, bonds, branches, ring numbers);
    3. Excludes English words and stereoisomer prefixes (e.g. `(S)-(+)`, `rolipram`, `Ibuprofen`);
    4. Ring numbers must appear in pairs (even count);
    5. Must contain at least one valid atom.
    """
    t = text.strip()
    m_p = _SMILES_PREFIX_RE.match(t)
    explicit_prefix = bool(m_p)
    if m_p:
        t = m_p.group(1).strip()
    if len(t) < 3 or any(c.isspace() for c in t) or "()" in t:
        return False
    if t.count("(") != t.count(")") or t.count("[") != t.count("]"):
        return False

    pos = 0
    tokens: list[str] = []
    while pos < len(t):
        m = _SMILES_TOKEN_RE.match(t, pos)
        if not m:
            return False
        tokens.append(m.group(0))
        pos = m.end()

    atoms = [tok for tok in tokens if tok.startswith("[") or tok in {"Cl", "Br"} or (len(tok) == 1 and tok in "BCNOPSFIbcnops")]
    if not atoms:
        return False

    rings = [tok for tok in tokens if tok.isdigit() or tok.startswith("%")]
    for r in set(rings):
        if rings.count(r) % 2 != 0:
            return False

    if explicit_prefix:
        return True

    lower = t.lower()
    if lower.startswith("sg") or lower.startswith("crispr") or lower.startswith("sh"):
        return False

    # Without explicit prefix, must contain chemical structure indicators to avoid gene symbol collisions (e.g. BOP, SON)
    has_struct = bool(set(t) & _SMILES_STRUCT_INDICATORS or any(c in "()." for c in t) or any(c.islower() for c in t))
    if not has_struct:
        if t.upper() in {"BOP", "SON", "PIN", "BIN", "COP", "PAN", "SKIP", "CLIP", "CHIP", "SNAP", "SPIN", "SCAN", "PIG", "GAP", "CAP", "SOS"}:
            return False
        if t.count("C") < 2 and "CC" not in t:
            return False

    return True


def detect_identifier(text: str) -> tuple[str, str | int] | None:
    r"""Detect whether input is a known chemical structure or authoritative registry accession.

    Checks standard formats in deterministic priority order:
    1. InChIKey: Universal compound structure hash (14-10-1 format, supports InChIKey= prefix)
    2. InChI: International Chemical Identifier (InChI=1S/...)
    3. ChEMBL ID: EMBL-EBI accession (CHEMBL\d+, supports CHEMBL: prefix)
    4. PubChem CID: NCBI Compound ID (CID: 12345 / CID12345)
    5. CAS Number: Chemical Abstracts Service registry number with modulo-10 checksum
    6. SMILES: OpenSMILES chemical structure representation
    7. DrugBank ID: DrugBank accession (DB\d{5}, supports DrugBank: prefix)
    8. Ensembl ID: Human gene Ensembl accession (ENSG\d{11}, supports Ensembl: prefix)
    9. UniProt Accession: Protein knowledgebase accession (supports UniProt: prefix)
    """
    t = text.strip()
    if not t:
        return None

    # 1. InChIKey
    m = _INCHIKEY_PREFIX_RE.match(t)
    if m:
        return ("inchikey", m.group(1).upper())

    # 2. InChI
    if is_inchi(t):
        return ("inchi", t)

    # 3. ChEMBL ID
    m = _CHEMBL_RE.match(t)
    if m:
        return ("chembl_id", m.group(1).upper())

    # 4. PubChem CID
    cid = parse_pubchem_cid(t)
    if cid is not None:
        return ("pubchem_cid", cid)

    # 5. CAS Number (with checksum)
    m = _CAS_PREFIX_RE.match(t)
    if m and is_cas_number(t):
        return ("cas", m.group(1))

    # 6. SMILES
    m = _SMILES_PREFIX_RE.match(t)
    if m and is_smiles(m.group(1)):
        return ("smiles", m.group(1))
    if is_smiles(t):
        return ("smiles", t)

    # 7. DrugBank ID
    m = _DRUGBANK_RE.match(t)
    if m:
        return ("drugbank_id", m.group(1).upper())

    # 8. Ensembl Gene ID
    m = _ENSEMBL_RE.match(t)
    if m:
        return ("ensembl_id", m.group(1).upper())

    # 9. UniProt Accession
    m = _UNIPROT_RE.match(t)
    if m:
        return ("uniprot_accession", m.group(1).upper())

    return None


__all__ = [
    "is_inchikey",
    "is_inchi",
    "is_cas_number",
    "is_chembl_id",
    "is_drugbank_id",
    "is_ensembl_id",
    "is_uniprot_accession",
    "is_smiles",
    "parse_pubchem_cid",
    "detect_identifier",
]

