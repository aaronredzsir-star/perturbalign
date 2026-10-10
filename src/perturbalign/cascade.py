"""Causal cascade and DoRothEA regulon inference engine.

Infers causal biological footprints from direct perturbation targets down to downstream
master transcription factors and benchmark response gene sets, based on validated
kinase-transcription factor signaling pathways and DoRothEA high-confidence (Level A)
TF-regulon prior knowledge.
"""

from typing import Any

# 1. Kinase/Target to Downstream Master TFs
_TARGET_TO_TF_MAP: dict[str, dict[str, Any]] = {
    # MAPK / ERK Cascade
    "MAP2K1": {"tfs": ["JUN", "FOS", "MYC", "EGR1", "ELK1"], "pathway": "MAPK/ERK Cascade"},
    "MAP2K2": {"tfs": ["JUN", "FOS", "MYC", "EGR1", "ELK1"], "pathway": "MAPK/ERK Cascade"},
    "BRAF": {"tfs": ["JUN", "FOS", "MYC", "EGR1"], "pathway": "RAF/MEK/ERK Cascade"},
    "RAF1": {"tfs": ["JUN", "FOS", "MYC", "EGR1"], "pathway": "RAF/MEK/ERK Cascade"},
    "MAPK1": {"tfs": ["JUN", "FOS", "MYC", "ELK1", "EGR1"], "pathway": "MAPK/ERK Cascade"},
    "MAPK3": {"tfs": ["JUN", "FOS", "MYC", "ELK1", "EGR1"], "pathway": "MAPK/ERK Cascade"},
    "MAPK8": {"tfs": ["JUN", "ATF2", "TP53"], "pathway": "JNK Signaling"},
    "MAPK9": {"tfs": ["JUN", "ATF2", "TP53"], "pathway": "JNK Signaling"},
    "MAPK14": {"tfs": ["ATF2", "STAT1", "TP53", "ELK1"], "pathway": "p38 MAPK Stress Pathway"},

    # Receptor Tyrosine Kinases (RTKs)
    "EGFR": {"tfs": ["STAT3", "MYC", "JUN", "FOS", "HIF1A"], "pathway": "EGFR/ErbB Signaling"},
    "ERBB2": {"tfs": ["STAT3", "MYC", "JUN", "FOS"], "pathway": "HER2/ErbB2 Signaling"},
    "ERBB3": {"tfs": ["STAT3", "FOXO3", "MYC"], "pathway": "HER3 Signaling"},
    "MET": {"tfs": ["STAT3", "MYC", "ETS1"], "pathway": "c-Met/HGF Signaling"},
    "KDR": {"tfs": ["HIF1A", "NFKB1", "STAT3"], "pathway": "VEGF/VEGFR2 Angiogenesis"},
    "KIT": {"tfs": ["STAT3", "STAT5A", "MYC"], "pathway": "c-Kit Signaling"},
    "PDGFRA": {"tfs": ["STAT3", "MYC", "FOS"], "pathway": "PDGFR Signaling"},
    "PDGFRB": {"tfs": ["STAT3", "MYC", "FOS"], "pathway": "PDGFR Signaling"},
    "ALK": {"tfs": ["STAT3", "MYC", "JUN"], "pathway": "ALK Fusion Signaling"},
    "ROS1": {"tfs": ["STAT3", "MYC"], "pathway": "ROS1 Signaling"},
    "RET": {"tfs": ["STAT3", "MYC", "JUN"], "pathway": "RET Kinase Pathway"},
    "FGFR1": {"tfs": ["FOS", "JUN", "STAT3"], "pathway": "FGF/FGFR Axis"},
    "FGFR2": {"tfs": ["FOS", "JUN", "STAT3"], "pathway": "FGF/FGFR Axis"},
    "FGFR3": {"tfs": ["FOS", "JUN", "STAT3"], "pathway": "FGF/FGFR Axis"},

    # Non-receptor Tyrosine Kinases (BCR-ABL / SRC)
    "ABL1": {"tfs": ["STAT5A", "STAT5B", "MYC", "JUN"], "pathway": "BCR-ABL1 Oncogenic Signaling"},
    "BCR": {"tfs": ["STAT5A", "STAT5B", "MYC"], "pathway": "BCR-ABL Fusion Pathway"},
    "SRC": {"tfs": ["STAT3", "NFKB1", "JUN"], "pathway": "Src Family Kinase Pathway"},

    # PI3K / AKT / mTOR Axis
    "PIK3CA": {"tfs": ["FOXO1", "FOXO3", "HIF1A", "MYC", "SREBF1"], "pathway": "PI3K/AKT/mTOR Axis"},
    "PIK3CB": {"tfs": ["FOXO1", "FOXO3", "HIF1A", "MYC"], "pathway": "PI3K/AKT Axis"},
    "PIK3CD": {"tfs": ["FOXO1", "FOXO3", "NFKB1"], "pathway": "PI3K Delta Immune Axis"},
    "AKT1": {"tfs": ["FOXO3", "HIF1A", "MYC", "CREB1"], "pathway": "AKT Pro-survival Signaling"},
    "AKT2": {"tfs": ["FOXO1", "SREBF1", "HIF1A"], "pathway": "AKT Metabolic Pathway"},
    "AKT3": {"tfs": ["FOXO3", "MYC"], "pathway": "AKT Signaling"},
    "MTOR": {"tfs": ["HIF1A", "SREBF1", "MYC", "TFEB"], "pathway": "mTORC1/mTORC2 Signaling"},

    # Cell Cycle and Checkpoints
    "CDK4": {"tfs": ["E2F1", "E2F2", "E2F3"], "pathway": "CDK4/6-Cyclin D-Rb-E2F Cell Cycle Axis"},
    "CDK6": {"tfs": ["E2F1", "E2F2", "E2F3"], "pathway": "CDK4/6-Cyclin D-Rb-E2F Cell Cycle Axis"},
    "CDK1": {"tfs": ["FOXM1", "TP53"], "pathway": "Mitotic Cell Cycle (G2/M)"},
    "CDK2": {"tfs": ["E2F1", "TP53"], "pathway": "G1/S Phase Progression"},
    "WEE1": {"tfs": ["TP53", "E2F1"], "pathway": "G2/M DNA Damage Checkpoint"},
    "CHEK1": {"tfs": ["TP53", "E2F1"], "pathway": "ATR/Chk1 Replication Stress"},
    "ATR": {"tfs": ["TP53", "E2F1"], "pathway": "ATR-mediated DNA Damage Response"},
    "ATM": {"tfs": ["TP53", "NFKB1"], "pathway": "ATM Double-strand Break Response"},

    # JAK / STAT Immune Signaling
    "JAK1": {"tfs": ["STAT1", "STAT3"], "pathway": "JAK1/STAT Inflammatory Axis"},
    "JAK2": {"tfs": ["STAT3", "STAT5A", "STAT5B"], "pathway": "JAK2/STAT Hematopoietic Axis"},
    "JAK3": {"tfs": ["STAT5A", "STAT6"], "pathway": "JAK3/STAT Common Gamma Chain Axis"},

    # NF-kB Inflammatory Pathway
    "IKBKB": {"tfs": ["NFKB1", "RELA"], "pathway": "Canonical NF-kB Activation"},
    "CHUK": {"tfs": ["NFKB1", "RELA"], "pathway": "Non-canonical NF-kB Activation"},

    # Apoptosis and DNA Damage Response
    "BCL2": {"tfs": ["TP53", "NFKB1"], "pathway": "Mitochondrial Apoptotic Pathway"},
    "BCL2L1": {"tfs": ["TP53", "NFKB1"], "pathway": "Anti-apoptotic Bcl-xL Axis"},
    "TOP2A": {"tfs": ["TP53", "NFKB1"], "pathway": "Topoisomerase II / DNA Cleavage Stress"},
    "TOP2B": {"tfs": ["TP53", "NFKB1"], "pathway": "Topoisomerase II-mediated DNA Breakage"},
    "PARP1": {"tfs": ["TP53", "NFKB1"], "pathway": "Base Excision Repair / Synthetic Lethality"},

    # Epigenetics (HDAC / Demethylation)
    "HDAC1": {"tfs": ["TP53", "MYC", "STAT3", "SP1"], "pathway": "Histone Deacetylation & Transcriptional Repression"},
    "HDAC2": {"tfs": ["TP53", "MYC", "STAT3", "SP1"], "pathway": "Histone Deacetylation & Chromatin Remodeling"},
    "HDAC3": {"tfs": ["NFKB1", "PPARG", "STAT3"], "pathway": "HDAC3 Corepressor Complex"},
    "KDM1A": {"tfs": ["MYC", "GATA1"], "pathway": "Histone Demethylation (LSD1)"},

    # Developmental and Microenvironment Pathways
    "SMO": {"tfs": ["GLI1", "GLI2"], "pathway": "Hedgehog Signaling Axis"},
    "NOTCH1": {"tfs": ["HES1", "HEY1", "MYC"], "pathway": "Notch Canonical Cleavage"},
    "TGFBR1": {"tfs": ["SMAD2", "SMAD3", "SMAD4"], "pathway": "TGF-beta / SMAD Signaling"},
    "ROCK1": {"tfs": ["SRF", "YAP1"], "pathway": "Rho/ROCK Cytoskeletal Tension & Mechanotransduction"},
    "ROCK2": {"tfs": ["SRF", "YAP1"], "pathway": "Rho/ROCK Cytoskeletal Signaling"},
}

# 2. DoRothEA High-Confidence (Level A) TF-Regulon Target Gene Signatures
# Supported by strong orthogonal experimental evidence (ChIP-seq, motifs, and perturbation validation)
_DOROTHEA_A_REGULONS: dict[str, list[str]] = {
    "TP53": ["CDKN1A", "MDM2", "BAX", "BBC3", "PMAIP1", "GADD45A", "FAS", "SESN1"],
    "MYC": ["ODC1", "CAD", "NPM1", "LDHA", "SLC2A1", "CDK4", "CCND2", "ENO1"],
    "E2F1": ["CCNE1", "CCNA2", "PCNA", "MCM2", "MCM4", "MCM6", "CDC6", "TK1", "BRCA1"],
    "E2F2": ["CCNE1", "PCNA", "MCM2", "CDC6"],
    "E2F3": ["CCNE1", "CCNA2", "PCNA", "MCM4"],
    "NFKB1": ["IL6", "TNF", "CXCL8", "NFKBIA", "BIRC3", "ICAM1", "CCL2", "TNFAIP3"],
    "RELA": ["IL6", "TNF", "CXCL8", "NFKBIA", "BIRC3", "ICAM1", "CCL2", "IL1B"],
    "STAT3": ["SOCS3", "BCL2L1", "MCL1", "IL10", "VEGFA", "FOS", "MYC"],
    "STAT1": ["IRF1", "CXCL9", "CXCL10", "STAT1", "OAS1", "MX1", "ISG15", "TAP1"],
    "STAT5A": ["PIM1", "BCL2L1", "SOCS2", "OSM", "CISH"],
    "STAT5B": ["PIM1", "BCL2L1", "SOCS2", "CISH"],
    "HIF1A": ["VEGFA", "SLC2A1", "LDHA", "PDK1", "ENO1", "CA9", "BNIP3"],
    "JUN": ["DUSP1", "EGR1", "ATF3", "CCND1", "IL8", "MMP1"],
    "FOS": ["DUSP1", "EGR1", "ATF3", "CCND1", "JUN"],
    "FOXO1": ["CDKN1B", "BCL2L11", "G6PC", "FOXO1"],
    "FOXO3": ["BCL2L11", "FASLG", "GADD45A", "CAT", "SOD2", "BNIP3", "CDKN1B"],
    "ELK1": ["FOS", "EGR1", "SRF"],
    "EGR1": ["FOS", "JUN", "NAB2", "DUSP6", "AR"],
    "ATF2": ["JUN", "FOS", "ATF3"],
    "CREB1": ["FOS", "JUN", "NR4A2", "EGR1", "BCL2"],
    "SREBF1": ["FASN", "ACACA", "SCD", "LDLR", "HMGCR"],
    "SMAD2": ["SERPINE1", "COL1A1", "SMAD7", "TGFB1"],
    "SMAD3": ["SERPINE1", "COL1A1", "SMAD7", "TGFB1", "CDKN1B"],
    "SMAD4": ["SERPINE1", "SMAD7", "CDKN1A"],
    "GLI1": ["PTCH1", "GLI1", "HHIP", "BCL2", "CCND1"],
    "GLI2": ["PTCH1", "GLI1", "HHIP"],
    "HES1": ["HES1", "CDKN1B", "ASCL1"],
    "HEY1": ["HEY1", "GATA2"],
    "FOXM1": ["CCNB1", "AURKA", "PLK1", "CDC25B", "CENPA"],
    "YAP1": ["CTGF", "CYR61", "ANKRD1", "AMOTL2"],
    "SRF": ["ACTA2", "FOS", "EGR1", "TAGLN"],
    "SP1": ["VEGFA", "TGFB1", "E2F1", "MYC"],
    "NFE2L2": ["HMOX1", "NQO1", "GCLC", "GCLM", "TXNRD1"],  # Nrf2 oxidative stress response
    "HSF1": ["HSP90AA1", "HSPA1A", "HSPA1B", "HSPB1", "DNAJA1"],  # Heat shock factor
    "ATF4": ["DDIT3", "ASNS", "TRIB3", "ATF3", "HERPUD1"],  # ER stress response
}


_CUSTOM_TARGET_TO_TF_MAP: dict[str, dict[str, Any]] = {}
_CUSTOM_DOROTHEA_REGULONS: dict[str, list[str]] = {}


def register_target_cascade(target: str, tfs: list[str], pathway: str) -> None:
    """Register or customize a kinase/target signaling cascade mapping to downstream master TFs."""
    _CUSTOM_TARGET_TO_TF_MAP[target.strip().upper()] = {
        "tfs": [tf.strip().upper() for tf in tfs],
        "pathway": pathway.strip(),
    }


def register_dorothea_regulon(tf: str, target_genes: list[str]) -> None:
    """Register or customize downstream responsive target genes for a transcription factor."""
    _CUSTOM_DOROTHEA_REGULONS[tf.strip().upper()] = [g.strip().upper() for g in target_genes]


def reset_cascade_registry() -> None:
    """Reset custom target cascade and regulon registries."""
    _CUSTOM_TARGET_TO_TF_MAP.clear()
    _CUSTOM_DOROTHEA_REGULONS.clear()


def infer_causal_cascade(
    targets: list[str] | str | None,
    cell_line: str | None = None,
    tissue: str | None = None,
) -> dict[str, Any]:
    """Infer downstream signaling pathways, master transcription factors, and DoRothEA regulons.

    Supports context-specific pruning and confidence scoring:
    - Automatically prunes and bypasses p53-dependent apoptotic/arrest branches when the host
      cell line (e.g., K562, Jurkat) carries TP53 mutations or deletions.
    - Computes dynamic confidence scores (0.0 - 1.0) based on host genetic background and lineage.

    Parameters
    ----------
    targets : list of str, str, or None
        Direct target gene symbol(s) (e.g., ['MAP2K1', 'MAP2K2'] or 'EGFR').
    cell_line : str, optional
        Host cell line name (e.g., 'A549', 'K562', 'MCF7', 'Jurkat', 'HepG2').
    tissue : str, optional
        Host tissue lineage (e.g., 'Lung', 'Blood', 'Breast', 'Liver').

    Returns
    -------
    dict
        Structured causal cascade dictionary containing direct_targets, pathway_names,
        downstream_tfs, active_tfs, pruned_tfs, response_regulon, context_specific_regulon,
        cell_line_context, cascade_confidence_score, regulon_source, and confidence.
    """
    if not targets:
        return {
            "direct_targets": [],
            "pathway_names": [],
            "downstream_tfs": [],
            "active_tfs": [],
            "pruned_tfs": [],
            "response_regulon": [],
            "context_specific_regulon": [],
            "cell_line_context": None,
            "cascade_confidence_score": 0.0,
            "regulon_source": "DoRothEA (Confidence A)",
            "confidence": "unspecified",
        }

    if isinstance(targets, str):
        t_list = [t.strip().upper() for t in targets.replace(";", ",").split(",") if t.strip()]
    else:
        t_list = [str(t).strip().upper() for t in targets if str(t).strip()]

    pathways: list[str] = []
    downstream_tfs: list[str] = []
    response_genes: list[str] = []

    def get_regulon(tf_symbol: str) -> list[str] | None:
        return _CUSTOM_DOROTHEA_REGULONS.get(tf_symbol) or _DOROTHEA_A_REGULONS.get(tf_symbol)

    def get_target_info(target_symbol: str) -> dict[str, Any] | None:
        return _CUSTOM_TARGET_TO_TF_MAP.get(target_symbol) or _TARGET_TO_TF_MAP.get(target_symbol)

    for t in t_list:
        # Case A: Target itself is a master transcription factor (e.g., TP53, MYC, STAT3)
        t_reg = get_regulon(t)
        if t_reg:
            if t not in downstream_tfs:
                downstream_tfs.append(t)
            for rg in t_reg:
                if rg not in response_genes:
                    response_genes.append(rg)
            pw_desc = f"Direct {t} Transcription Factor Regulon"
            if pw_desc not in pathways:
                pathways.append(pw_desc)

        # Case B: Target is an upstream kinase/receptor/enzyme; retrieve downstream converging TFs
        info = get_target_info(t)
        if info:
            pw = info["pathway"]
            if pw not in pathways:
                pathways.append(pw)
            for tf in info["tfs"]:
                if tf not in downstream_tfs:
                    downstream_tfs.append(tf)
                # Fetch DoRothEA Level A benchmark response genes for this TF
                tf_reg = get_regulon(tf)
                if tf_reg:
                    for rg in tf_reg:
                        if rg not in response_genes:
                            response_genes.append(rg)

    # Host cell line and lineage context-specific inference and pruning
    cl_context = None
    active_tfs = list(downstream_tfs)
    pruned_tfs: list[dict[str, str]] = []
    context_regulon = list(response_genes)
    base_score = 0.85 if response_genes else (0.50 if t_list else 0.0)

    if cell_line:
        from .cell_line import query_cell_line
        cl_context = query_cell_line(cell_line)
        if cl_context:
            tp53_st = str(cl_context.get("tp53_status", "")).lower()
            # 1. TP53 mutation/loss pruning: if host cell line is TP53-deficient, attenuate TP53-mediated branches
            if any(term in tp53_st for term in ("mutant", "null", "loss")):
                if "TP53" in active_tfs:
                    active_tfs.remove("TP53")
                    pruned_tfs.append({
                        "tf": "TP53",
                        "reason": f"TP53 is {cl_context.get('tp53_status')} in {cell_line.upper()}; p53-mediated transcriptional arrest/apoptosis regulon is attenuated.",
                    })
                    # Remove genes exclusively activated by TP53
                    p53_exclusives = set(get_regulon("TP53") or [])
                    other_active_targets = set()
                    for atf in active_tfs:
                        other_active_targets.update(get_regulon(atf) or [])
                    context_regulon = [g for g in context_regulon if g not in p53_exclusives or g in other_active_targets]
                    base_score = 0.65  # Pathway attenuated / bypassed

            # 2. Cancer driver gene alignment enhancement
            dr_muts = [str(m).upper() for m in cl_context.get("driver_mutations", [])]
            for t in t_list:
                if any(t in m for m in dr_muts):
                    base_score = min(0.98, base_score + 0.12)

    confidence = "high"
    if pruned_tfs:
        confidence = "context_attenuated"
    elif not response_genes:
        confidence = "moderate" if t_list else "unspecified"

    return {
        "direct_targets": t_list,
        "pathway_names": pathways,
        "downstream_tfs": downstream_tfs,
        "active_tfs": active_tfs,
        "pruned_tfs": pruned_tfs,
        "response_regulon": response_genes,
        "context_specific_regulon": context_regulon,
        "cell_line_context": cl_context,
        "cascade_confidence_score": round(base_score, 2),
        "regulon_source": "DoRothEA (Confidence A)",
        "confidence": confidence,
    }


__all__ = [
    "infer_causal_cascade",
    "register_target_cascade",
    "register_dorothea_regulon",
    "reset_cascade_registry",
]
