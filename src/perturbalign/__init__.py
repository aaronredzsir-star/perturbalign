"""perturbalign: single-cell chemical perturbation metadata alignment & annotation.

The package does ONLY alignment logic: messy perturbation labels
("Doxorubicin HCl 10nM", "DMSO", "GDC-0032") -> standardized mapping
(name / dose / control flag / cocktail split / ambiguity audit).
Molecular data (SMILES / InChIKey / targets / MoA) comes from ONLINE APIs
(PubChem / ChEMBL) by default, persisted in a local SQLite cache that also
serves as offline fuzzy-matching corpus; no built-in molecule database,
no dependency on any user data shape.

Pipeline: Preprocessor & Parser -> Normalization Engine (cache corpus)
-> Online API (default) -> Single-cell adapter.
"""

__version__ = "0.9.2"

from .align import align, to_dataframe, get_default_pipeline, reset_default_pipeline  # noqa: F401
from .schema import (  # noqa: F401
    AlignResult,
    AlignStatus,
    Candidate,
    DrugEntity,
    SmallMoleculeEntity,
    PerturbationEntity,
    GeneticEntity,
    BiologicEntity,
    EnvironmentalEntity,
    DeliveryVehicleEntity,
    TreatmentStage,
    EnrichedPerturbation,
    Modality,
    MODALITY_SMALL_MOLECULE,
    MODALITY_BIOLOGIC,
    MODALITY_GENETIC,
    MODALITY_ENVIRONMENTAL,
    MODALITY_CONTROL,
    MODALITY_COMBO,
    MODALITY_EMPTY,
    MODALITY_DELIVERY,
)
from .cache import (  # noqa: F401
    BaseCache,
    MemoryCache,
    NullCache,
    SQLiteCache,
    get_default_cache,
)
from .resources import (  # noqa: F401
    VocabularyRegistry,
    default_vocab_registry,
    register_vocabulary,
    get_vocabulary,
    list_vocabularies,
    reset_vocabulary_registry,
)
from .pipeline import (  # noqa: F401
    BaseResolver,
    AlignmentContext,
    AlignmentPipeline,
    IdentifierResolver,
    SequentialRegimeResolver,
    DeliveryVehicleResolver,
)
from .preprocess import (  # noqa: F401
    PreprocessConfig,
    parse_dose,
    parse_duration,
    parse_spatiotemporal,
    detect_control,
    detect_genetic_perturbation,
    detect_biologic,
    detect_environmental,
    detect_delivery_vehicle,
    detect_protac,
    split_sequential_regime,
)
from .identifiers import (  # noqa: F401
    detect_identifier,
    is_cas_number,
    is_inchikey,
    is_smiles,
    is_chembl_id,
    is_drugbank_id,
    is_ensembl_id,
    is_uniprot_accession,
)
from .adapter import (  # noqa: F401
    annotate_anndata,
    annotate_dataframe,
    annotate_mudata,
    explode_combos,
    from_lincs,
    from_scperturb,
    from_prism,
    from_jump_cp,
    lookup_sgrna,
    to_frictionless,
    to_jsonld,
    classify_protocol,
    adapt_protocol,
    BaseProtocolAdapter,
    LINCSAdapter,
    ScPerturbAdapter,
    PRISMAdapter,
    JUMPCPAdapter,
    ProtocolRegistry,
    default_protocol_registry,
)
from .annotate import (  # noqa: F401
    BaseAnnotationProvider,
    PubChemProvider,
    ChEMBLProvider,
    NCICIRProvider,
    ProviderRegistry,
    default_provider_registry,
    resolve_online_cascade,
)
from .dose import (  # noqa: F401
    classify_dose_dimension,
    standardize_dose,
    to_pconc,
    register_dose_dimension,
    DIMENSION_MOLAR,
    DIMENSION_MASS_CONCENTRATION,
    DIMENSION_BODY_WEIGHT,
    DIMENSION_ACTIVITY,
    DIMENSION_PERCENTAGE,
    DIMENSION_INFECTION_RATIO,
    DIMENSION_AMOUNT,
    DIMENSION_VOLUME,
    DIMENSION_RADIATION,
    DIMENSION_SHEAR_STRESS,
    DIMENSION_OSMOLARITY,
    DIMENSION_PRESSURE,
    DIMENSION_TEMPERATURE,
    DIMENSION_MULTIPLIER,
)
from .structure import cluster_by_parent, parent_inchikey  # noqa: F401
from .gene import (  # noqa: F401
    lookup_ncbi_gene,
    get_esm2_embedding,
    get_string_network,
    get_depmap_dependency,
    lookup_reactome_pathways,
    lookup_gene_ontology,
    get_opentargets_info,
    get_jaspar_motifs,
    get_hpa_expression,
    get_clinvar_variants,
    get_encode_ccres,
    query_gene,
)
from .compound import (  # noqa: F401
    get_pubchem_compound,
    get_chemberta_embedding,
    get_l1000_profile,
    get_prism_profile,
    get_jump_profile,
    get_chembl_clinical_info,
    get_gdsc_sensitivity,
    get_stitch_interactions,
    get_bindingdb_affinity,
    get_drugbank_annotations,
    get_fda_adverse_events,
    query_compound,
)
from .embedding import (  # noqa: F401
    embed_perturbation,
    embed_batch,
    BaseEmbeddingProvider,
    ESM2EmbeddingProvider,
    ChemBERTaEmbeddingProvider,
    EmbeddingRegistry,
    default_embedding_registry,
    register_embedding_provider,
)
from .async_engine import (  # noqa: F401
    TokenBucketLimiter,
    global_rate_limiters,
    run_concurrent_tasks,
    http_get_json,
    http_post_json,
    http_get_text,
)
from .hub import (  # noqa: F401
    get_ppi_network,
    get_profile,
    featurize_anndata,
    BaseProfileProvider,
    L1000ProfileProvider,
    PRISMProfileProvider,
    JUMPProfileProvider,
    DepMapProfileProvider,
    ProfileRegistry,
    default_profile_registry,
    register_profile_provider,
)

from .specification import (  # noqa: F401
    PerturbationAtom,
    PerturbationUnit,
    PerturbationDataset,
    parse_input,
)
from .hetero_kg import (  # noqa: F401
    HeteroNode,
    HeteroEdge,
    HeteroPath,
    HeteroSubgraph,
    HeterogeneousKnowledgeGraph,
    global_hetero_kg,
    search_hetero_pathways,
    get_hetero_subgraph,
    meta_path_search,
    predict_pathway_impact,
)
from .graph_store import SQLiteGraphStore  # noqa: F401
from .kg_loaders import (  # noqa: F401
    load_all_curated_kg,
    load_primekg,
    load_hetionet,
    load_biogrid,
    load_dorothea,
    load_reactome_kegg,
    load_stitch,
    load_synlethdb,
    load_progeny_pathways,
)
from .kg_registry import (  # noqa: F401
    BaseKGProvider,
    KGRegistry,
    default_kg_registry,
    register_kg_provider,
)
from . import kg_providers  # noqa: F401
from .epigenetics import (  # noqa: F401
    CisRegulatoryElement,
    resolve_ccres,
    predict_epigenetic_impact,
    query_epigenetics,
    register_ccre,
    reset_epigenetics_registry,
)
from .cell_line import (  # noqa: F401
    CellLineEntity,
    query_cell_line,
    register_cell_line,
    reset_cell_line_registry,
)
from .connectivity import (  # noqa: F401
    compute_connectivity_score,
    predict_reversal_drugs,
    register_reference_signature,
    reset_reference_signatures,
)
from .safety import (  # noqa: F401
    evaluate_target_safety,
    evaluate_safety,
    register_safety_target,
    register_safety_compound,
    reset_safety_registry,
)
from .exceptions import (  # noqa: F401
    PerturbAlignError,
    NormalizationError,
    AmbiguousEntityError,
    UnmatchedEntityError,
    SpatiotemporalParseError,
    DimensionalMismatchError,
    RegistryError,
    GraphInferenceError,
)
from .specification import PerturbationContext, build_context  # noqa: F401
from .kg_yaml import register_declarative_kg  # noqa: F401
from .facade import process, enrich, enrich_batch, enrich_anndata, standardize_anndata, validate_upds  # noqa: F401
from .ortholog import get_gene_orthologs, detect_organism  # noqa: F401
from .compound import resolve_investigational_code  # noqa: F401
from .cascade import (  # noqa: F401
    infer_causal_cascade,
    register_target_cascade,
    register_dorothea_regulon,
    reset_cascade_registry,
)
from .mcp_server import PerturbAlignMCPServer, run_stdio_server  # noqa: F401

__all__ = [
    "process",
    "enrich",
    "enrich_batch",
    "enrich_anndata",
    "standardize_anndata",
    "validate_upds",
    "resolve_investigational_code",
    "infer_causal_cascade",
    "get_gene_orthologs",
    "detect_organism",
    "EnrichedPerturbation",
    "align", "to_dataframe", "get_default_pipeline", "reset_default_pipeline",
    "AlignResult", "AlignStatus", "Candidate", "DrugEntity", "SmallMoleculeEntity",
    "PerturbationEntity", "GeneticEntity", "BiologicEntity", "EnvironmentalEntity",
    "DeliveryVehicleEntity", "TreatmentStage", "Modality",
    "MODALITY_SMALL_MOLECULE", "MODALITY_BIOLOGIC", "MODALITY_GENETIC", "MODALITY_ENVIRONMENTAL",
    "MODALITY_CONTROL", "MODALITY_COMBO", "MODALITY_EMPTY", "MODALITY_DELIVERY",
    "BaseCache", "MemoryCache", "NullCache", "SQLiteCache",
    "get_default_cache",
    "VocabularyRegistry", "default_vocab_registry", "register_vocabulary", "get_vocabulary", "list_vocabularies", "reset_vocabulary_registry",
    "BaseResolver", "AlignmentContext", "AlignmentPipeline", "IdentifierResolver",
    "SequentialRegimeResolver", "DeliveryVehicleResolver",
    "PreprocessConfig", "annotate_anndata", "annotate_dataframe", "annotate_mudata", "explode_combos",
    "parent_inchikey", "cluster_by_parent",
    "parse_dose", "parse_duration", "parse_spatiotemporal",
    "detect_control", "detect_genetic_perturbation",
    "detect_biologic", "detect_environmental", "detect_delivery_vehicle", "detect_protac", "split_sequential_regime",
    "detect_identifier", "is_cas_number", "is_inchikey", "is_smiles", "is_chembl_id",
    "is_drugbank_id", "is_ensembl_id", "is_uniprot_accession",
    "from_lincs", "from_scperturb", "from_prism", "from_jump_cp", "classify_protocol", "adapt_protocol",
    "lookup_sgrna", "to_frictionless", "to_jsonld",
    "BaseProtocolAdapter", "LINCSAdapter", "ScPerturbAdapter", "PRISMAdapter", "JUMPCPAdapter", "ProtocolRegistry",
    "BaseAnnotationProvider", "PubChemProvider", "ChEMBLProvider", "NCICIRProvider",
    "ProviderRegistry", "resolve_online_cascade",
    "classify_dose_dimension", "standardize_dose", "to_pconc", "register_dose_dimension",
    "DIMENSION_MOLAR", "DIMENSION_MASS_CONCENTRATION", "DIMENSION_BODY_WEIGHT",
    "DIMENSION_ACTIVITY", "DIMENSION_PERCENTAGE", "DIMENSION_INFECTION_RATIO",
    "DIMENSION_AMOUNT", "DIMENSION_VOLUME", "DIMENSION_RADIATION",
    "DIMENSION_SHEAR_STRESS", "DIMENSION_OSMOLARITY", "DIMENSION_PRESSURE",
    "DIMENSION_TEMPERATURE", "DIMENSION_MULTIPLIER",
    "lookup_ncbi_gene", "get_esm2_embedding", "get_string_network", "get_depmap_dependency",
    "lookup_reactome_pathways", "lookup_gene_ontology",
    "get_opentargets_info", "get_jaspar_motifs", "get_hpa_expression", "get_clinvar_variants", "get_encode_ccres",
    "query_gene",
    "get_pubchem_compound", "get_chemberta_embedding", "get_l1000_profile", "get_prism_profile", "get_jump_profile",
    "get_chembl_clinical_info",
    "get_gdsc_sensitivity", "get_stitch_interactions", "get_bindingdb_affinity", "get_drugbank_annotations", "get_fda_adverse_events",
    "query_compound",
    "embed_perturbation", "embed_batch",
    "BaseEmbeddingProvider", "ESM2EmbeddingProvider", "ChemBERTaEmbeddingProvider",
    "EmbeddingRegistry", "default_embedding_registry", "register_embedding_provider",
    "TokenBucketLimiter", "global_rate_limiters", "run_concurrent_tasks",
    "http_get_json", "http_post_json", "http_get_text",
    "get_ppi_network", "get_profile", "featurize_anndata",
    "BaseProfileProvider", "L1000ProfileProvider", "PRISMProfileProvider", "JUMPProfileProvider", "DepMapProfileProvider",
    "ProfileRegistry", "default_profile_registry", "register_profile_provider",
    "PerturbationAtom", "PerturbationUnit", "PerturbationDataset", "parse_input",
    "PerturbationContext", "build_context",
    "HeteroNode", "HeteroEdge", "HeteroPath", "HeteroSubgraph", "HeterogeneousKnowledgeGraph", "global_hetero_kg",
    "SQLiteGraphStore",
    "search_hetero_pathways", "get_hetero_subgraph", "meta_path_search", "predict_pathway_impact",
    "load_all_curated_kg", "load_primekg", "load_hetionet", "load_biogrid", "load_dorothea",
    "load_reactome_kegg", "load_stitch", "load_synlethdb", "load_progeny_pathways",
    "BaseKGProvider", "KGRegistry", "default_kg_registry", "register_kg_provider",
    "register_declarative_kg",
    "CisRegulatoryElement", "resolve_ccres", "predict_epigenetic_impact", "query_epigenetics",
    "register_ccre", "reset_epigenetics_registry",
    "CellLineEntity", "query_cell_line", "register_cell_line", "reset_cell_line_registry",
    "compute_connectivity_score", "predict_reversal_drugs",
    "register_reference_signature", "reset_reference_signatures",
    "evaluate_target_safety", "evaluate_safety",
    "register_safety_target", "register_safety_compound", "reset_safety_registry",
    "register_target_cascade", "register_dorothea_regulon", "reset_cascade_registry",
    "PerturbAlignError", "NormalizationError", "AmbiguousEntityError", "UnmatchedEntityError",
    "SpatiotemporalParseError", "DimensionalMismatchError", "RegistryError", "GraphInferenceError",
    "PerturbAlignMCPServer", "run_stdio_server",
    "__version__",
]
