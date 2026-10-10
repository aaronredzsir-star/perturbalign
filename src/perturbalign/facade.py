"""PerturbAlign Unified Top-Level Facade.

Provides a unified one-line entry point for single-cell perturbation data cleaning, structure resolution, embeddings, and full knowledge enrichment.
`perturbalign.process(data)`
"""
from __future__ import annotations

from typing import Any, Optional, Union
import pandas as pd

from .schema import (
    EnrichedPerturbation,
    MODALITY_CONTROL,
    MODALITY_GENETIC,
    MODALITY_SMALL_MOLECULE,
    MODALITY_BIOLOGIC,
    MODALITY_ENVIRONMENTAL,
    MODALITY_COMBO,
)


def process(
    data: Any,
    pert_col: str = "perturbation",
    cell_line: Optional[str] = None,
    online: bool = False,
    fuzzy: bool = False,
    add_embeddings: bool = True,
    add_profiles: bool = True,
    build_ppi: bool = False,
    embed_dim: int = 512,
    **kwargs: Any,
) -> Any:
    """Fully automated one-line dataset alignment and knowledge enrichment for single-cell perturbation datasets.

    Delivers out-of-the-box one-liner usability:
    1. AnnData input -> automatically cleans perturbations, populates obs, injects obsm embeddings (ESM2/ChemBERTa), and writes uns audit dictionary.
    2. DataFrame input -> returns enriched DataFrame with standardized names, normalized doses/durations, control flags, and multimodal attributes.
    3. list / Series / str input -> automatically encapsulates into UPDS PerturbationDataset or PerturbationContext.

    Parameters
    ----------
    data : AnnData, DataFrame, Series, list, or str
        data: Input single-cell perturbation data object (AnnData, DataFrame, Series, list, or single string).
    pert_col : str
        perturbation_col: Perturbation label column name (default 'perturbation').
    cell_line : str, optional
        cell_line: Host cell line (e.g. 'A549', 'MCF7', 'K562').
    online : bool
        online: Whether to allow online API queries (default True, graceful offline fallback).
    fuzzy : bool
        fuzzy: Whether to enable fuzzy spelling correction.
    add_embeddings : bool
        embed: Whether to automatically generate and register 512d multimodal embeddings into obsm.
    add_profiles : bool
        profile: Whether to annotate L1000/PRISM cellular phenotype profiles.
    build_ppi : bool
        kg: Whether to automatically align STRING PPI and interaction subgraphs.
    embed_dim : int
        embed_dim: Unified embedding space dimensionality (default 512).

    Returns
    -------
    Any
        Enriched data object (preserving input type compatibility).
    """
    # 1. AnnData input processing
    if hasattr(data, "obs") and hasattr(data, "obsm"):
        from .hub import featurize_anndata
        from .cell_line import query_cell_line

        adata = featurize_anndata(
            data,
            drug_col=pert_col,
            embed_dim=embed_dim,
            add_embeddings=add_embeddings,
            add_profiles=add_profiles,
            build_ppi_network=build_ppi,
            online=online,
            fuzzy=fuzzy,
        )

        # Inject cell line context
        if cell_line:
            c_info = query_cell_line(cell_line)
            if c_info:
                adata.uns["perturbalign_cell_line"] = c_info
                adata.obs["perturbalign_host_cell_line"] = c_info["name"]
                adata.obs["perturbalign_host_primary_disease"] = c_info["primary_disease"]
                adata.obs["perturbalign_host_tp53"] = c_info["tp53_status"]
                adata.obs["perturbalign_host_kras"] = c_info["kras_status"]

        return adata

    # 2. DataFrame input processing
    if hasattr(data, "columns") and isinstance(data, pd.DataFrame):
        from .adapter import annotate_dataframe
        out_df, audit = annotate_dataframe(
            data,
            pert_col=pert_col,
            online=online,
            fuzzy=fuzzy,
            return_audit=True,
            **kwargs,
        )
        if cell_line:
            from .cell_line import query_cell_line
            c_info = query_cell_line(cell_line)
            if c_info:
                out_df["perturbalign_host_cell_line"] = c_info["name"]
                out_df["perturbalign_host_primary_disease"] = c_info["primary_disease"]
        return out_df

    # 3. Series, list, or string -> encapsulate as UPDS standardized object
    from .specification import parse_input, build_context
    if cell_line:
        return build_context(data, cell_line=cell_line, **kwargs)
    return parse_input(data)


# --------------------------------------------------------------------------
# Unified Multi-Source Knowledge Enrichment Facade
# --------------------------------------------------------------------------
def enrich(
    query: str,
    organism: str = "auto",
    cell_line: str | None = None,
    tissue: str | None = None,
    online: bool = True,
    cache: Any = None,
) -> Any:
    """Standardize and clean a single raw perturbation query with one-stop aggregated multi-modal/cross-species knowledge profiling.

    Args:
      query: Any unstandardized perturbation input (e.g. 'Gleevec 100nM', 'sgTrp53', 'TNF-alpha', 'Hypoxia').
      organism: Target organism ('auto' for automatic inference, 'human', 'mouse', 'rat', 'zebrafish').
      cell_line: Host cell line (e.g. 'A549', 'K562', 'MCF7', 'Jurkat') for tissue-specific contexts.
      tissue: Tissue origin (e.g. 'Lung', 'Blood', 'Breast').
      online: Whether to allow online queries (default True; gracefully degrades to curated knowledge bases and cache on failure).
      cache: SQLiteCache instance, or None for auto-created cache.

    Returns:
      EnrichedPerturbation object containing standardized identifiers and multi-dimensional knowledge profile.
    """
    from .align import align
    from .schema import (
        EnrichedPerturbation,
        MODALITY_CONTROL,
        MODALITY_GENETIC,
        MODALITY_SMALL_MOLECULE,
        MODALITY_BIOLOGIC,
        MODALITY_ENVIRONMENTAL,
        MODALITY_COMBO,
    )

    clean_res = align([query], online=False)[0]
    raw_str = str(query).strip()
    canon_name = clean_res.parsed_name or raw_str
    modality = str(clean_res.modality)
    is_ctrl = clean_res.is_control

    enriched = EnrichedPerturbation(
        raw_input=raw_str,
        canonical_name=canon_name,
        modality=modality,
        is_control=is_ctrl,
        dose=clean_res.parsed_dose,
        dose_unit=clean_res.parsed_unit,
        duration=clean_res.parsed_duration,
        duration_unit=clean_res.parsed_duration_unit,
    )

    # 1. Control group identification
    if is_ctrl:
        enriched.identifiers = {"control_type": "negative_vehicle"}
        enriched.properties = {"description": "Negative or solvent vehicle baseline control"}
        return enriched

    # 2. Small molecule chemical perturbations (PubChem structure + ChEMBL / DrugBank targets)
    if modality == MODALITY_SMALL_MOLECULE:
        from .compound import (
            get_pubchem_compound,
            get_chembl_clinical_info,
            get_drugbank_annotations,
            get_chembl_indications,
        )
        pub_info = get_pubchem_compound(canon_name, online=online, cache=cache) or {}
        chembl_clin = get_chembl_clinical_info(canon_name, online=online, cache=cache) or {}
        db_info = get_drugbank_annotations(canon_name, online=online, cache=cache) or {}
        indications_list = get_chembl_indications(canon_name, online=online, cache=cache) or []

        if pub_info.get("canonical_name"):
            enriched.canonical_name = pub_info["canonical_name"]

        # Dedicated unique identifier recognition
        enriched.identifiers = {
            "pubchem_cid": pub_info.get("pubchem_cid"),
            "smiles": pub_info.get("canonical_smiles"),
            "inchikey": pub_info.get("inchikey"),
            "chembl_id": chembl_clin.get("chembl_id"),
            "drugbank_id": db_info.get("drugbank_id"),
            "iupac_name": pub_info.get("iupac_name"),
        }

        # Cascaded knowledge profile (targets, MoA, indications, and physicochemical parameters)
        targets = db_info.get("targets") or pub_info.get("targets", [])
        moa_list = list(db_info.get("mechanisms_of_action") or ([] if not db_info.get("moa") else [db_info.get("moa")]))
        if not moa_list and pub_info.get("class_name"):
            moa_list = [pub_info["class_name"]]

        from .cascade import infer_causal_cascade
        causal_cascade = infer_causal_cascade(targets, cell_line=cell_line, tissue=tissue)

        enriched.properties = {
            "molecular_weight": pub_info.get("molecular_weight"),
            "xlogp": pub_info.get("xlogp"),
            "tpsa": pub_info.get("tpsa"),
            "h_bond_donors": pub_info.get("h_bond_donors"),
            "h_bond_acceptors": pub_info.get("h_bond_acceptors"),
            "targets": targets,
            "mechanisms_of_action": moa_list,
            "pharmacological_class": db_info.get("pharmacological_class"),
            "max_clinical_phase": chembl_clin.get("max_phase", 4 if chembl_clin.get("is_approved") else 0),
            "is_approved": chembl_clin.get("is_approved", True if chembl_clin.get("max_phase") == 4 else False),
            "indications": indications_list,
            "synonyms": pub_info.get("synonyms", []),
            "causal_cascade": causal_cascade,
        }
        return enriched

    # 3. Genetic perturbations / CRISPR KO (cross-species ortholog mapping + NCBI annotations + Reactome pathways + STRING PPI)
    if modality == MODALITY_GENETIC:
        from .ortholog import get_gene_orthologs, detect_organism
        from .gene import lookup_ncbi_gene, lookup_reactome_pathways, get_string_network

        target_sym = clean_res.genetic_target or canon_name
        detected_org = detect_organism(target_sym) if organism == "auto" else organism.lower()

        # Cross-species ortholog mapping
        orthologs = get_gene_orthologs(target_sym, organism=detected_org, online=online, cache=cache)
        human_sym = orthologs.get("human", {}).get("symbol", target_sym.upper())

        # NCBI annotation (prioritize target organism)
        ncbi_data = lookup_ncbi_gene(target_sym, organism=detected_org, online=online, cache=cache)
        if not ncbi_data and detected_org != "human":
            ncbi_data = lookup_ncbi_gene(human_sym, organism="human", online=online, cache=cache)
        ncbi_data = ncbi_data or {}

        # Signaling pathways and PPI network
        reactome = lookup_reactome_pathways(human_sym, online=online, cache=cache) or []
        string_net = get_string_network(human_sym, online=online, cache=cache) or {}

        enriched.identifiers = {
            "entrez_id": ncbi_data.get("entrez_id"),
            "symbol": ncbi_data.get("symbol", target_sym),
            "ensembl_id": ncbi_data.get("ensembl_id"),
            "uniprot": orthologs.get(detected_org, {}).get("uniprot") or orthologs.get("human", {}).get("uniprot"),
            "organism": detected_org,
        }

        from .cascade import infer_causal_cascade
        causal_cascade = infer_causal_cascade([human_sym], cell_line=cell_line, tissue=tissue)

        enriched.properties = {
            "genetic_action": clean_res.genetic_action or "knockout",
            "description": ncbi_data.get("description", ""),
            "summary": ncbi_data.get("summary", ""),
            "chromosome": ncbi_data.get("chromosome", ""),
            "aliases": ncbi_data.get("aliases", []),
            "orthologs": orthologs,  # Cross-species ortholog mapping
            "pathways": [str(p.get("pathway_name") or p.get("name") or "") for p in reactome[:5] if p and (p.get("pathway_name") or p.get("name"))] if reactome else [],
            "ppi_partners": string_net.get("partners") or string_net.get("top_partners", []),
            "causal_cascade": causal_cascade,
        }
        return enriched

    # 4. Cytokines and Biologics (cognate receptors and signaling cascades)
    if modality == MODALITY_BIOLOGIC:
        from .gene import lookup_ncbi_gene, lookup_reactome_pathways, get_string_network

        _CYTOKINE_KB: dict[str, dict[str, Any]] = {
            "TNF": {"uniprot": "P01375", "receptors": ["TNFRSF1A", "TNFRSF1B"], "pathways": ["NF-kappa B signaling", "Apoptosis", "MAPK signaling"]},
            "IL6": {"uniprot": "P05231", "receptors": ["IL6R", "IL6ST"], "pathways": ["JAK-STAT3 signaling", "Immune response"]},
            "IFNG": {"uniprot": "P01579", "receptors": ["IFNGR1", "IFNGR2"], "pathways": ["Interferon-gamma signaling", "JAK-STAT1 pathway"]},
            "IFNA1": {"uniprot": "P01562", "receptors": ["IFNAR1", "IFNAR2"], "pathways": ["Type I interferon signaling", "Antiviral response"]},
            "IFNB1": {"uniprot": "P01574", "receptors": ["IFNAR1", "IFNAR2"], "pathways": ["Type I interferon signaling", "Innate immunity"]},
            "IL1B": {"uniprot": "P01584", "receptors": ["IL1R1", "IL1RAP"], "pathways": ["IL-1 signaling", "NF-kappa B cascade", "Inflammatory response"]},
            "IL1A": {"uniprot": "P01583", "receptors": ["IL1R1", "IL1RAP"], "pathways": ["IL-1 signaling pathway"]},
            "IL2": {"uniprot": "P60568", "receptors": ["IL2RA", "IL2RB", "IL2RG"], "pathways": ["T cell proliferation", "IL-2/STAT5 pathway"]},
            "IL4": {"uniprot": "P05112", "receptors": ["IL4R", "IL2RG"], "pathways": ["STAT6 activation", "Th2 differentiation"]},
            "IL10": {"uniprot": "P22301", "receptors": ["IL10RA", "IL10RB"], "pathways": ["Anti-inflammatory signaling", "STAT3 pathway"]},
            "IL12A": {"uniprot": "P29459", "receptors": ["IL12RB1", "IL12RB2"], "pathways": ["Th1 immune response", "STAT4 activation"]},
            "IL17A": {"uniprot": "Q16552", "receptors": ["IL17RA", "IL17RC"], "pathways": ["IL-17 signaling pathway", "Neutrophil recruitment"]},
            "TGFB1": {"uniprot": "P01137", "receptors": ["TGFBR1", "TGFBR2"], "pathways": ["SMAD2/3 signaling", "Epithelial to mesenchymal transition"]},
            "VEGFA": {"uniprot": "P15692", "receptors": ["FLT1", "KDR"], "pathways": ["Angiogenesis", "VEGF receptor signaling"]},
            "CXCL8": {"uniprot": "P10145", "receptors": ["CXCR1", "CXCR2"], "pathways": ["Chemokine signaling", "Neutrophil chemotaxis"]},
            "CCL2": {"uniprot": "P13500", "receptors": ["CCR2"], "pathways": ["Monocyte chemotaxis", "Chemokine receptor binding"]},
            "EGF": {"uniprot": "P01133", "receptors": ["EGFR"], "pathways": ["EGFR signaling pathway", "Cell proliferation"]},
            "FGF2": {"uniprot": "P09038", "receptors": ["FGFR1", "FGFR2", "FGFR3"], "pathways": ["FGF signaling", "MAPK activation"]},
            "CSF1": {"uniprot": "P09603", "receptors": ["CSF1R"], "pathways": ["Macrophage differentiation", "CSF1R signaling"]},
            "CSF2": {"uniprot": "P04141", "receptors": ["CSF2RA", "CSF2RB"], "pathways": ["Granulocyte and macrophage differentiation"]},
        }

        # Normalize and extract target symbol (e.g. IL-1beta -> IL1B, TGF-beta1 -> TGFB1, TNF-alpha -> TNF)
        up_raw = canon_name.upper().replace(" ", "")
        clean_key = (
            up_raw.replace("-BETA", "B")
            .replace("BETA", "B")
            .replace("-ALPHA", "A")
            .replace("ALPHA", "A")
            .replace("-GAMMA", "G")
            .replace("GAMMA", "G")
            .replace("-", "")
        )
        if clean_key == "TNFA":
            clean_key = "TNF"
        elif clean_key == "IL1":
            clean_key = "IL1B"

        info = _CYTOKINE_KB.get(clean_key) or _CYTOKINE_KB.get(up_raw.replace("-", ""))

        # Dynamic online fallback (query NCBI Gene + STRING for unknown cytokines)
        if not info and online:
            gene_info = lookup_ncbi_gene(clean_key, online=True, cache=cache)
            if gene_info:
                net = get_string_network(clean_key, online=True, cache=cache) or {}
                partners = net.get("partners") or []
                # Heuristic receptor matching: look for high-confidence interacting receptors with R / CD / REC
                detected_receptors = [
                    p for p in partners
                    if any(term in p for term in ["R", "REC", "CD"]) or p.endswith(("R", "R1", "R2", "RA", "RB"))
                ]
                pws = lookup_reactome_pathways(clean_key, online=True, cache=cache) or []
                info = {
                    "uniprot": None,
                    "receptors": detected_receptors[:4],
                    "pathways": [p.get("name") for p in pws[:4] if p.get("name")],
                }

        receptors = info.get("receptors", []) if info else []
        pathways = info.get("pathways", []) if info else []
        uniprot_id = info.get("uniprot") if info else None

        from .cascade import infer_causal_cascade
        causal_cascade = infer_causal_cascade(receptors, cell_line=cell_line, tissue=tissue)

        enriched.identifiers = {"category": "cytokine_or_biologic", "uniprot": uniprot_id}
        enriched.properties = {
            "receptors": receptors,
            "pathways": pathways,
            "description": f"Cytokine or biologic perturbation entity {canon_name}",
            "causal_cascade": causal_cascade,
        }
        return enriched

    # 5. Environmental Stress (aligned with MSigDB Hallmark stress signatures)
    if modality == MODALITY_ENVIRONMENTAL:
        from .gene import lookup_reactome_pathways

        _ENVIRONMENTAL_STRESS_KB: dict[str, dict[str, Any]] = {
            "HYPOXIA": {
                "sensor_tfs": ["HIF1A", "EPAS1"],
                "marker_genes": ["VEGFA", "SLC2A1", "LDHA", "BNIP3", "CA9"],
                "pathways": ["HIF-1 signaling pathway", "Glycolytic switch", "Response to hypoxia"],
                "description": "Low oxygen tension microenvironment driving glycolytic switch and angiogenesis.",
            },
            "HEAT_SHOCK": {
                "sensor_tfs": ["HSF1"],
                "marker_genes": ["HSPA1A", "HSPA1B", "HSP90AA1", "DNAJB1"],
                "pathways": ["Cellular response to heat stress", "Chaperone-mediated protein folding"],
                "description": "Thermal stress inducing proteotoxic heat shock response.",
            },
            "OXIDATIVE_STRESS": {
                "sensor_tfs": ["NFE2L2", "KEAP1"],
                "marker_genes": ["HMOX1", "NQO1", "SOD1", "CAT", "GCLC"],
                "pathways": ["Response to reactive oxygen species", "Glutathione metabolism"],
                "description": "Redox perturbation causing macromolecular oxidative damage.",
            },
            "DNA_DAMAGE": {
                "sensor_tfs": ["TP53", "ATM", "ATR"],
                "marker_genes": ["CDKN1A", "MDM2", "GADD45A", "RAD51", "BAX"],
                "pathways": ["DNA damage response", "p53 signaling pathway", "G1/S checkpoint"],
                "description": "Genotoxic insult or radiation inducing DNA break repair cascades.",
            },
            "GLUCOSE_STARVATION": {
                "sensor_tfs": ["ATF4", "PRKAA1"],
                "marker_genes": ["DDIT3", "TRIB3", "ASNS", "TXNIP"],
                "pathways": ["Response to starvation", "AMPK signaling", "Amino acid deprivation"],
                "description": "Nutrient depletion driving metabolic reprogramming and stress autophagy.",
            },
            "ACIDOSIS": {
                "sensor_tfs": ["HIF1A", "RELA"],
                "marker_genes": ["SLC9A1", "CA9", "CA12", "LAMP2"],
                "pathways": ["Cellular response to acidic pH", "Proton extrusion and pH regulation"],
                "description": "Extracellular acidosis stress triggering acid-sensing ion channels and survival signals.",
            },
            "OSMOTIC_STRESS": {
                "sensor_tfs": ["NFAT5", "MAPK14"],
                "marker_genes": ["SLC5A3", "AKR1B1", "SGK1"],
                "pathways": ["Response to osmotic stress", "p38 MAPK pathway"],
                "description": "Hyperosmolar pressure driving cellular dehydration and osmoprotectant synthesis.",
            },
            "ER_STRESS": {
                "sensor_tfs": ["XBP1", "ATF6", "ATF4"],
                "marker_genes": ["HSPA5", "DDIT3", "HERPUD1", "ERN1"],
                "pathways": ["Unfolded protein response (UPR)", "Endoplasmic reticulum stress response"],
                "description": "Accumulation of misfolded proteins in endoplasmic reticulum.",
            },
        }

        up_env = canon_name.upper().replace(" ", "_").replace("-", "_")
        up_raw_input = raw_str.upper().replace(" ", "_").replace("-", "_")
        combined_text = f"{up_env}_{up_raw_input}"

        _ENV_ALIAS = {
            "ACIDOSIS": ["ACIDOSIS", "ACID", "LOW_PH", "PH_6", "PH_5"],
            "HYPOXIA": ["HYPOXIA", "LOW_O2", "1%_O2", "ANOXIA"],
            "HEAT_SHOCK": ["HEAT_SHOCK", "HYPERTHERMIA", "HEAT_STRESS", "42C"],
            "GLUCOSE_STARVATION": ["GLUCOSE_STARVATION", "GLUCOSE_DEPRIVATION", "STARVATION", "NUTRIENT_DEPRIVATION"],
            "OXIDATIVE_STRESS": ["OXIDATIVE", "ROS", "H2O2", "REDOX"],
            "DNA_DAMAGE": ["DNA_DAMAGE", "GENOTOXIC", "IRRADIATION", "RADIATION"],
            "OSMOTIC_STRESS": ["OSMOTIC", "HYPEROSMOLAR", "NACL", "HYPERTONIC"],
            "ER_STRESS": ["ER_STRESS", "UPR", "TUNICA", "THAPSIGARGIN"],
        }
        matched_kb = None
        for kb_key, aliases in _ENV_ALIAS.items():
            if any(alias in combined_text for alias in aliases):
                matched_kb = _ENVIRONMENTAL_STRESS_KB[kb_key]
                break

        # Cross-species dynamic pathway matching
        online_pathways = []
        if not matched_kb and online:
            pw_res = lookup_reactome_pathways(canon_name, online=True, cache=cache)
            online_pathways = [p.get("name") for p in pw_res[:4] if p.get("name")]

        if matched_kb:
            sensor_tfs = list(matched_kb["sensor_tfs"])
            marker_genes = list(matched_kb["marker_genes"])
            pathways = list(matched_kb["pathways"])
            desc = matched_kb["description"]
        else:
            # Heuristic construction of transcriptional response set for unmapped stress
            sensor_tfs = []
            marker_genes = []
            pathways = online_pathways
            desc = f"Environmental condition {canon_name}"

        from .cascade import infer_causal_cascade
        causal_cascade = infer_causal_cascade(sensor_tfs, cell_line=cell_line, tissue=tissue)

        enriched.identifiers = {"category": "environmental_stress"}
        enriched.properties = {
            "sensor_transcription_factors": sensor_tfs,
            "marker_genes": marker_genes,
            "pathways": pathways,
            "description": desc,
            "causal_cascade": causal_cascade,
        }
        return enriched

    # 6. General fallback
    enriched.identifiers = {"raw_name": canon_name}
    enriched.properties = {"note": "Unclassified perturbation"}
    return enriched


def enrich_batch(
    queries: list[str],
    organism: str = "auto",
    online: bool = True,
    cache: Any = None,
) -> list[Any]:
    """Parse and enrich perturbation label."""
    return [enrich(q, organism=organism, online=online, cache=cache) for q in queries]


# UPDS (Unified Perturbation Data Schema v1.0) Controlled Vocabulary Specification
UPDS_SCHEMA_VERSION = "1.0.0"
UPDS_ALLOWED_MODALITIES = {
    "small_molecule",
    "crispr_ko",
    "crispr_a",
    "crispr_i",
    "rnai",
    "overexpression",
    "biologic",
    "environmental",
    "combination",
    "control",
}

UPDS_REQUIRED_OBS_COLUMNS = [
    "pert_canonical_name",
    "pert_modality",
    "pert_is_control",
    "pert_dose_val",
    "pert_dose_unit",
    "pert_dose_norm_um",
    "pert_duration_h",
    "pert_human_ortholog",
    "pert_primary_targets",
    "pert_smiles",
]


def _normalize_dose_um(val: Any, unit: Any) -> float:
    """Normalize dose to micromolar concentration (uM). Returns NaN if unparseable."""
    try:
        v = float(val)
        if pd.isna(v) or v <= 0:
            return float("nan")
    except (ValueError, TypeError):
        return float("nan")

    from .dose import standardize_dose
    u = str(unit or "").strip()
    try:
        res, _ = standardize_dose(v, u, target_unit="µM")
        return round(res, 6) if res is not None else float("nan")
    except Exception:
        return float("nan")



def _normalize_duration_h(val: Any, unit: Any) -> float:
    """Normalize duration to hours (h). Returns NaN if unparseable."""
    try:
        v = float(val)
        if pd.isna(v) or v < 0:
            return float("nan")
    except (ValueError, TypeError):
        return float("nan")

    u = str(unit or "").strip().lower()
    if u in ("h", "hr", "hrs", "hour", "hours"):
        return round(v * 1.0, 4)
    if u in ("d", "day", "days"):
        return round(v * 24.0, 4)
    if u in ("m", "min", "mins", "minute", "minutes"):
        return round(v / 60.0, 4)
    if u in ("s", "sec", "secs", "second", "seconds"):
        return round(v / 3600.0, 4)
    return float("nan")


def _map_to_upds_modality(raw_modality: str, is_ctrl: bool, action: str | None = None) -> str:
    """Map modality string to UPDS strictly controlled vocabulary enum."""
    if is_ctrl:
        return "control"
    m = (raw_modality or "").lower()
    act = (action or "").lower()

    if "crispr" in m or "genetic" in m:
        if any(term in act for term in ("a", "act", "crispra", "activation")):
            return "crispr_a"
        if any(term in act for term in ("i", "inh", "crispri", "interference", "inhibition")):
            return "crispr_i"
        if any(term in act for term in ("rnai", "sirna", "shrna")):
            return "rnai"
        if any(term in act for term in ("oe", "overexpression", "orf")):
            return "overexpression"
        return "crispr_ko"

    if "small" in m or "compound" in m or "chemical" in m or "drug" in m:
        return "small_molecule"
    if "biologic" in m or "cytokine" in m or "antibody" in m or "protein" in m:
        return "biologic"
    if "env" in m or "stress" in m or "physical" in m:
        return "environmental"
    if "comb" in m or "cocktail" in m:
        return "combination"
    return "small_molecule"


def standardize_anndata(
    adata: Any,
    obs_col: str = "perturbation",
    dose_col: str | None = None,
    unit_col: str | None = None,
    time_col: str | None = None,
    time_unit_col: str | None = None,
    cell_line_col: str | None = "cell_line",
    cell_line: str | None = None,
    tissue: str | None = None,
    organism: str = "auto",
    online: bool = False,
    copy: bool = False,
) -> Any:
    """Strictly standardize single-cell AnnData perturbation metadata to UPDS (Unified Perturbation Data Schema v1.0).

    Columns written to adata.obs:
      - pert_canonical_name: Official standardized Symbol
      - pert_modality: Strictly controlled vocabulary enum (small_molecule, crispr_ko, biologic, environmental, control...)
      - pert_is_control: Control indicator (True / False)
      - pert_dose_val: Raw numeric dose value (float)
      - pert_dose_unit: Raw dose unit (str)
      - pert_dose_norm_um: Unified micromolar concentration (float, uM)
      - pert_duration_h: Unified duration in hours (float, h)
      - pert_human_ortholog: Normalized human HGNC Symbol (str)
      - pert_primary_targets: Semicolon-separated list of standard targets (str)
      - pert_smiles: Canonical SMILES (str)

    Written to adata.uns['perturbalign']:
      - schema_version: "1.0.0"
      - standard: "UPDS-v1.0"
      - entity_knowledge: Structured multi-dimensional knowledge profile and pathway map
      - audit_summary: Audit summary dictionary
    """
    target_adata = adata.copy() if copy else adata
    if obs_col not in target_adata.obs.columns:
        raise KeyError(f"Column '{obs_col}' does not exist in adata.obs. Available columns: {list(target_adata.obs.columns)}")

    # Extract cell line and tissue context
    resolved_cl = cell_line
    if not resolved_cl and cell_line_col and cell_line_col in target_adata.obs.columns:
        cl_series = target_adata.obs[cell_line_col].dropna()
        if len(cl_series) > 0:
            resolved_cl = str(cl_series.iloc[0]).strip()

    unique_vals = target_adata.obs[obs_col].unique()
    knowledge_map = {
        val: enrich(val, organism=organism, cell_line=resolved_cl, tissue=tissue, online=online)
        for val in unique_vals
    }

    obs = target_adata.obs
    n_rows = len(obs)

    # Preprocess and standardize queries
    canonical_names = []
    modalities = []
    is_controls = []
    dose_vals = []
    dose_units = []
    dose_norm_ums = []
    duration_hs = []
    human_orthologs = []
    primary_targets = []
    smiles_list = []

    for idx in range(n_rows):
        val = obs[obs_col].iloc[idx] if hasattr(obs[obs_col], "iloc") else obs[obs_col][idx]
        k = knowledge_map.get(val)

        # Dose extraction
        d_val = obs[dose_col].iloc[idx] if dose_col and dose_col in obs.columns else (k.dose if k else None)
        d_unit = obs[unit_col].iloc[idx] if unit_col and unit_col in obs.columns else (k.dose_unit if k else "")
        d_norm_um = _normalize_dose_um(d_val, d_unit)

        # Duration extraction
        dur_val = obs[time_col].iloc[idx] if time_col and time_col in obs.columns else (k.duration if k else None)
        dur_unit = obs[time_unit_col].iloc[idx] if time_unit_col and time_unit_col in obs.columns else (k.duration_unit if k else "")
        dur_norm_h = _normalize_duration_h(dur_val, dur_unit)

        # Modality controlled vocabulary mapping
        is_ctrl = bool(k.is_control) if k else False
        act = k.properties.get("genetic_action") if k else None
        modality = _map_to_upds_modality(k.modality if k else "small_molecule", is_ctrl, act)

        # Human ortholog extraction
        h_ortho = ""
        if k and k.modality == MODALITY_GENETIC:
            orthos = k.properties.get("orthologs", {})
            h_ortho = orthos.get("human", {}).get("symbol") or k.identifiers.get("symbol", "")

        # Primary target extraction
        tgts = []
        if k:
            if "targets" in k.properties:
                tgts = k.properties.get("targets", [])
            elif "receptors" in k.properties:
                tgts = k.properties.get("receptors", [])
            elif "sensor_transcription_factors" in k.properties:
                tgts = k.properties.get("sensor_transcription_factors", [])
        tgt_str = ";".join(str(t) for t in tgts if str(t).strip())

        # SMILES
        smi = k.identifiers.get("smiles", "") if k else ""

        canonical_names.append(k.canonical_name if k else str(val))
        modalities.append(modality)
        is_controls.append(is_ctrl)
        dose_vals.append(float(d_val) if d_val is not None and not pd.isna(d_val) else float("nan"))
        dose_units.append(str(d_unit) if d_unit and not pd.isna(d_unit) else "")
        dose_norm_ums.append(d_norm_um)
        duration_hs.append(dur_norm_h)
        human_orthologs.append(str(h_ortho))
        primary_targets.append(tgt_str)
        smiles_list.append(str(smi or ""))

    # Write UPDS standard fields to adata.obs
    obs["pert_canonical_name"] = canonical_names
    obs["pert_modality"] = modalities
    obs["pert_is_control"] = is_controls
    obs["pert_dose_val"] = dose_vals
    obs["pert_dose_unit"] = dose_units
    obs["pert_dose_norm_um"] = dose_norm_ums
    obs["pert_duration_h"] = duration_hs
    obs["pert_human_ortholog"] = human_orthologs
    obs["pert_primary_targets"] = primary_targets
    obs["pert_smiles"] = smiles_list

    # Write full-dimensional knowledge graph metadata to adata.uns["perturbalign"]
    target_adata.uns["perturbalign"] = {
        "schema_version": UPDS_SCHEMA_VERSION,
        "standard": "UPDS-v1.0",
        "pipeline_version": "0.3.0",
        "entity_knowledge": {v: k.as_dict() for v, k in knowledge_map.items()},
        "audit_summary": {
            "n_cells": n_rows,
            "n_unique_perturbations": len(unique_vals),
            "modalities": sorted(list(set(modalities))),
            "n_controls": int(sum(is_controls)),
        },
    }
    # Record history log
    target_adata.uns["perturbation_knowledge"] = target_adata.uns["perturbalign"]["entity_knowledge"]

    return target_adata


def validate_upds(adata: Any) -> dict[str, Any]:
    """Validate whether AnnData strictly complies with UPDS (Unified Perturbation Data Schema v1.0) specification.

    Returns:
      {
        "is_valid": bool,
        "errors": list[str],
        "warnings": list[str],
        "schema_version": str,
        "summary": dict[str, Any],
      }
    """
    errors = []
    warnings = []

    if not hasattr(adata, "obs"):
        return {"is_valid": False, "errors": ["Target object lacks .obs attribute"], "warnings": [], "schema_version": None, "summary": {}}

    obs_cols = set(adata.obs.columns)
    for req_col in UPDS_REQUIRED_OBS_COLUMNS:
        if req_col not in obs_cols:
            errors.append(f"Missing required standard column: adata.obs['{req_col}']")

    if "pert_modality" in obs_cols:
        invalid_modalities = set(adata.obs["pert_modality"]) - UPDS_ALLOWED_MODALITIES
        if invalid_modalities:
            errors.append(f"Column 'pert_modality' contains invalid controlled vocabulary values: {sorted(list(invalid_modalities))}")

    if not hasattr(adata, "uns") or "perturbalign" not in adata.uns:
        warnings.append("adata.uns lacks 'perturbalign' metadata namespace")

    is_valid = len(errors) == 0
    summary = {
        "n_cells": len(adata.obs),
        "columns_present": [c for c in UPDS_REQUIRED_OBS_COLUMNS if c in obs_cols],
        "modalities": list(set(adata.obs["pert_modality"])) if "pert_modality" in obs_cols else [],
    }

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "schema_version": UPDS_SCHEMA_VERSION if is_valid else None,
        "summary": summary,
    }


def enrich_anndata(
    adata: Any,
    obs_col: str = "perturbation",
    organism: str = "auto",
    online: bool = True,
    copy: bool = False,
) -> Any:
    """High-level convenience wrapper: executes full knowledge enrichment and automatically converts AnnData to UPDS specification."""
    return standardize_anndata(
        adata=adata,
        obs_col=obs_col,
        organism=organism,
        online=online,
        copy=copy,
    )

