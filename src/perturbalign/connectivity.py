"""CMap connectivity and signature reversion prediction engine.

Provides Tau score computation and drug repurposing/signature reversal candidate predictions
for disease/resistance gene sets based on LINCS L1000 and Connectivity Map principles.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional


# Curated L1000 reference perturbagen signatures
_CURATED_REFERENCE_SIGNATURES: dict[str, dict[str, Any]] = {
    "Trametinib": {
        "drug": "Trametinib",
        "target": "MEK1/2 (MAP2K1/2)",
        "moa": "MEK Inhibitor",
        "up_regulated": ["CDKN1A", "SPRY2", "KLF4", "DUSP4", "GADD45A"],
        "down_regulated": ["DUSP6", "EGR1", "ETV4", "FOS", "MYC", "CCND1"],
    },
    "Doxorubicin": {
        "drug": "Doxorubicin",
        "target": "TOP2A / DNA",
        "moa": "Topoisomerase II Inhibitor",
        "up_regulated": ["CDKN1A", "BAX", "MDM2", "GADD45A", "FAS", "BBC3"],
        "down_regulated": ["MCM2", "PCNA", "TOP2A", "CCNB1", "CDK1"],
    },
    "Vorinostat": {
        "drug": "Vorinostat (SAHA)",
        "target": "HDAC1/2/3/6",
        "moa": "HDAC Inhibitor",
        "up_regulated": ["CDKN1A", "GADD45B", "HMOX1", "ATF3", "TXNIP"],
        "down_regulated": ["MYC", "CCND1", "BCL2", "MKI67"],
    },
    "Palbociclib": {
        "drug": "Palbociclib",
        "target": "CDK4/6",
        "moa": "CDK4/6 Inhibitor",
        "up_regulated": ["CDKN1A", "CDKN2D", "RB1"],
        "down_regulated": ["E2F1", "CCNE1", "CDC6", "MCM7", "CCNA2"],
    },
    "Osimertinib": {
        "drug": "Osimertinib",
        "target": "EGFR (T790M/L858R)",
        "moa": "3rd Gen EGFR TKI",
        "up_regulated": ["BCL2L11", "DUSP1", "CDKN1B"],
        "down_regulated": ["EGFR", "MAPK1", "AKT1", "MYC", "HIF1A"],
    },
    "Venetoclax": {
        "drug": "Venetoclax",
        "target": "BCL-2",
        "moa": "BCL-2 Inhibitor (BH3 Mimetic)",
        "up_regulated": ["PMAIP1", "BBC3", "BAX", "CASP3"],
        "down_regulated": ["BCL2", "MCL1", "BCL2L1"],
    },
    "Dexamethasone": {
        "drug": "Dexamethasone",
        "target": "NR3C1",
        "moa": "Glucocorticoid Receptor Agonist",
        "up_regulated": ["TSC22D3", "FKBP5", "DUSP1", "PER1"],
        "down_regulated": ["IL6", "TNF", "IL1B", "NFKB1", "CXCL8"],
    },
}


def compute_connectivity_score(
    query_up: Iterable[str],
    query_down: Iterable[str],
    reference_up: Iterable[str],
    reference_down: Iterable[str],
) -> float:
    """Calculate connectivity score between query signature and reference perturbagen signature (-100 to +100).

    Scoring mechanism:
    - Positive values (+100 max): Perturbation induces concordant transcriptional phenotype (Mimic).
    - Negative values (-100 min): Perturbation induces discordant/reversal phenotype (Reversal/Therapeutic).
    """
    q_up = {g.strip().upper() for g in query_up if g}
    q_down = {g.strip().upper() for g in query_down if g}
    r_up = {g.strip().upper() for g in reference_up if g}
    r_down = {g.strip().upper() for g in reference_down if g}

    if not (q_up or q_down) or not (r_up or r_down):
        return 0.0

    # 1. Concordance: q_up in r_up, q_down in r_down
    concordant = len(q_up & r_up) + len(q_down & r_down)

    # 2. Discordance / Reversal: q_up in r_down, q_down in r_up
    discordant = len(q_up & r_down) + len(q_down & r_up)

    total_possible = max(len(q_up) + len(q_down), 1)

    raw_score = (concordant - discordant) / total_possible
    # Scale to standard CMap Tau dimension [-100, +100]
    tau = max(min(raw_score * 100.0, 100.0), -100.0)
    return round(tau, 2)


def register_reference_signature(drug_name: str, signature: dict[str, Any]) -> None:
    """Register or customize a reference perturbagen signature for CMap connectivity scoring."""
    _CURATED_REFERENCE_SIGNATURES[drug_name] = signature


def predict_reversal_drugs(
    up_genes: list[str],
    down_genes: list[str],
    top_k: int = 5,
    reference_signatures: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Predict candidate small molecules with highest signature reversal potential for given dysregulated genes.

    Parameters
    ----------
    up_genes : list[str]
        List of abnormally upregulated genes in disease state (e.g. ['MYC', 'CCND1', 'IL6']).
    down_genes : list[str]
        List of abnormally repressed genes in disease state (e.g. ['CDKN1A', 'BAX']).
    top_k : int
        Number of top reversal candidates to return.
    reference_signatures : dict, optional
        Custom reference signature dictionary. Defaults to internal reference database.

    Returns
    -------
    list[dict]
        Ranked list of recommended candidate drugs sorted from strongest reversal to weakest.
    """
    results = []
    sigs = reference_signatures if reference_signatures is not None else _CURATED_REFERENCE_SIGNATURES
    for drug_name, sig in sigs.items():
        tau = compute_connectivity_score(
            query_up=up_genes,
            query_down=down_genes,
            reference_up=sig["up_regulated"],
            reference_down=sig["down_regulated"],
        )
        if tau < 0:  # Only retain perturbations with negative (reversal) correlation
            results.append({
                "drug": drug_name,
                "target": sig["target"],
                "moa": sig["moa"],
                "tau_score": tau,
                "reversal_confidence": round(abs(tau), 2),
                "reversed_up_genes": list({g.upper() for g in up_genes} & {g.upper() for g in sig["down_regulated"]}),
                "restored_down_genes": list({g.upper() for g in down_genes} & {g.upper() for g in sig["up_regulated"]}),
            })

    results.sort(key=lambda x: x["tau_score"])  # Most negative tau first
    return results[:top_k]


_DEFAULT_REFERENCE_KEYS = set(_CURATED_REFERENCE_SIGNATURES.keys())


def reset_reference_signatures() -> None:
    """Reset reference signatures to built-in defaults."""
    for k in list(_CURATED_REFERENCE_SIGNATURES.keys()):
        if k not in _DEFAULT_REFERENCE_KEYS:
            del _CURATED_REFERENCE_SIGNATURES[k]


__all__ = [
    "compute_connectivity_score",
    "predict_reversal_drugs",
    "register_reference_signature",
    "reset_reference_signatures",
]

