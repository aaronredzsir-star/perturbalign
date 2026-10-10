"""Unified output schema (strongly typed dataclass and StrEnum architecture, zero required third-party dependencies).

Multi-modal unified abstraction:
- Enumeration types: Modality, AlignStatus (inheriting from str, Enum for type safety and string interoperability)
- Polymorphic entity hierarchy: PerturbationEntity base class with specialized entities (DrugEntity, GeneticEntity, BiologicEntity, EnvironmentalEntity, DeliveryVehicleEntity)
- Standardized identifier alignment for chemical and life sciences (PubChem CID, ChEMBL ID, HGNC, UniProt, InChIKey)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Union, Any


# 1. Modality Enumeration (inherits str, Enum for type safety, autocomplete, and string comparison)
class Modality(str, Enum):
    """Perturbation modality enumeration."""
    SMALL_MOLECULE = "small_molecule"
    BIOLOGIC = "biologic"
    GENETIC = "genetic"
    ENVIRONMENTAL = "environmental"
    CONTROL = "control"
    COMBO = "combo"
    EMPTY = "empty"
    DELIVERY = "delivery"

    def __str__(self) -> str:
        return self.value


# Backwards compatibility constants
MODALITY_SMALL_MOLECULE = Modality.SMALL_MOLECULE.value
MODALITY_BIOLOGIC = Modality.BIOLOGIC.value
MODALITY_GENETIC = Modality.GENETIC.value
MODALITY_ENVIRONMENTAL = Modality.ENVIRONMENTAL.value
MODALITY_CONTROL = Modality.CONTROL.value
MODALITY_COMBO = Modality.COMBO.value
MODALITY_EMPTY = Modality.EMPTY.value
MODALITY_DELIVERY = Modality.DELIVERY.value


# 2. Alignment Status Enumeration
class AlignStatus(str, Enum):
    """Perturbation matching and alignment status."""
    EXACT = "exact"
    ALIAS = "alias"
    FUZZY = "fuzzy"
    AMBIGUOUS = "ambiguous"
    CONTROL = "control"
    COMBO = "combo"
    UNMATCHED = "unmatched"
    EMPTY = "empty"
    ONLINE = "online"

    def __str__(self) -> str:
        return self.value


# Backwards compatibility constants
STATUS_EXACT = AlignStatus.EXACT.value
STATUS_ALIAS = AlignStatus.ALIAS.value
STATUS_FUZZY = AlignStatus.FUZZY.value
STATUS_AMBIGUOUS = AlignStatus.AMBIGUOUS.value
STATUS_CONTROL = AlignStatus.CONTROL.value
STATUS_COMBO = AlignStatus.COMBO.value
STATUS_UNMATCHED = AlignStatus.UNMATCHED.value
STATUS_EMPTY = AlignStatus.EMPTY.value
STATUS_ONLINE = AlignStatus.ONLINE.value

# Candidate match scores (configurable default values)
SCORE_EXACT = 1.0
SCORE_ALIAS = 0.9
SCORE_NORM_KEY = 0.8
SCORE_FUZZY_MAX = 0.75  # Fuzzy matching ceiling: lower than norm_key to ensure exact matches take precedence
AMBIGUOUS_GAP = 0.05    # Gap between top1 and top2 score below which match is deemed ambiguous


# 3. Polymorphic Entity Hierarchy (PerturbationEntity architecture)
@dataclass
class PerturbationEntity:
    """Generic abstract base class for all perturbation modalities."""

    input_string: str
    canonical_name: Optional[str] = None
    modality: str = MODALITY_SMALL_MOLECULE
    confidence: float = 0.0
    match_path: list[str] = field(default_factory=list)
    source: Optional[str] = None
    vocabulary_version: Optional[str] = None
    extra_metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DrugEntity(PerturbationEntity):
    """Resolved small molecule chemical entity (dual primary key: normalized_name + parent_inchikey, PROTAC, molecular glues)."""

    canonical_smiles: Optional[str] = None
    pubchem_cid: Optional[int] = None
    chembl_id: Optional[str] = None
    drugbank_id: Optional[str] = None
    inchikey: Optional[str] = None
    parent_inchikey: Optional[str] = None
    cas_number: Optional[str] = None
    unii: Optional[str] = None
    targets: list[str] = field(default_factory=list)
    moa: Optional[str] = None
    synonyms: list[str] = field(default_factory=list)
    e3_ligase: Optional[str] = None
    e3_ligand: Optional[str] = None
    linker: Optional[str] = None
    is_protac: bool = False
    is_molecular_glue: bool = False
    chemberta_embedding: Optional[list[float]] = None
    l1000_profile: Optional[dict] = None
    prism_profile: Optional[dict] = None
    jump_profile: Optional[dict] = None
    gdsc_sensitivity: Optional[dict] = None
    stitch_interactions: list[dict] = field(default_factory=list)
    bindingdb_affinity: Optional[dict] = None
    drugbank_annotations: Optional[dict] = None
    fda_adverse_events: Optional[dict] = None
    molecular_properties: dict = field(default_factory=dict)

    def __post_init__(self):
        self.modality = MODALITY_SMALL_MOLECULE


# Semantic alias
SmallMoleculeEntity = DrugEntity


@dataclass
class GeneticEntity(PerturbationEntity):
    """Genetic and synthetic biology perturbation entity (target genes, mechanics, epigenetic effectors, opto/chemogenetics)."""

    gene_symbol: Optional[str] = None
    action: Optional[str] = None  # KO, KD, OE, CRISPRi, CRISPRa, mutant...
    ensembl_id: Optional[str] = None
    entrez_id: Optional[str] = None
    guide_id: Optional[str] = None
    mutation: Optional[str] = None
    system_type: Optional[str] = None  # crispr_ko, epigenome_edit, optogenetics, chemogenetics...
    effector: Optional[str] = None     # dCas9-KRAB, dCas9-TET1, ChR2, hM3Dq...
    ncbi_annotation: Optional[dict] = None
    ncbi_summary: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    esm2_embedding: Optional[list[float]] = None
    string_partners: list[dict] = field(default_factory=list)
    depmap_dependency: Optional[dict] = None
    reactome_pathways: list[dict] = field(default_factory=list)
    gene_ontology: list[dict] = field(default_factory=list)
    opentargets: Optional[dict] = None
    jaspar: Optional[dict] = None
    hpa: Optional[dict] = None
    clinvar: Optional[dict] = None
    encode: Optional[dict] = None

    def __post_init__(self):
        self.modality = MODALITY_GENETIC
        if not self.canonical_name and self.gene_symbol:
            self.canonical_name = self.gene_symbol


@dataclass
class DeliveryVehicleEntity(PerturbationEntity):
    """Vehicle and nanoparticle delivery system entity (LNP, AAV, Exosome, Lentivirus, Liposome, etc.)."""

    carrier_type: Optional[str] = None  # lnp, aav, exosome, lentivirus, adenovirus, liposome...
    serotype: Optional[str] = None      # e.g., AAV9, AAV-PHP.eB, AAV2, AAV8...
    cargo: list[str] = field(default_factory=list)  # e.g., ["mRNA", "Cas9-RNP", "siRNA"]
    formulation: list[str] = field(default_factory=list)  # e.g., ["MC3", "DSPC", "Cholesterol", "DMG-PEG2000"]
    tropism: Optional[str] = None       # e.g., "liver", "cns", "muscle"

    def __post_init__(self):
        self.modality = MODALITY_DELIVERY


@dataclass
class BiologicEntity(PerturbationEntity):
    """Biologic macromolecule entity (cytokines, growth factors, therapeutic antibodies, ADCs, RNAi)."""

    category: Optional[str] = None  # cytokine, antibody, adc, rnai, etc.
    hgnc_or_uniprot: Optional[str] = None
    target_antigen: Optional[str] = None

    def __post_init__(self):
        self.modality = MODALITY_BIOLOGIC


@dataclass
class EnvironmentalEntity(PerturbationEntity):
    """Physical and microenvironmental perturbation entity (hypoxia, starvation, heat shock, shear stress, osmolarity, etc.)."""

    category: Optional[str] = None  # oxygen, temperature, starvation, mechanical...
    parameter: Optional[str] = None  # e.g., '1% O2', '42°C', '10 dyn/cm2'

    def __post_init__(self):
        self.modality = MODALITY_ENVIRONMENTAL


# 4. Matching Candidates and Alignment Results
@dataclass
class Candidate:
    """A candidate alignment match: entity + confidence score + match source pathway."""

    entity: PerturbationEntity
    score: float  # 0..1
    score_breakdown: dict[str, float] = field(default_factory=dict)
    matched_by: str = ""  # ExactMatcher / AliasMatcher / NormKeyMatcher
    matched_surface: str = ""

    def as_dict(self) -> dict:
        return {
            "entity": self.entity.as_dict() if self.entity else None,
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "matched_by": self.matched_by,
            "matched_surface": self.matched_surface,
        }

    @property
    def canonical_name(self) -> Optional[str]:
        return self.entity.canonical_name if self.entity else None

    @property
    def canonical_smiles(self) -> Optional[str]:
        return getattr(self.entity, "canonical_smiles", None)

    @property
    def inchikey(self) -> Optional[str]:
        return getattr(self.entity, "inchikey", None)

    @property
    def pubchem_cid(self) -> Optional[int]:
        return getattr(self.entity, "pubchem_cid", None)

    @property
    def parent_inchikey(self) -> Optional[str]:
        return getattr(self.entity, "parent_inchikey", None)


@dataclass
class TreatmentStage:
    """Treatment stage definition in temporal or pulse-chase experimental regimens."""

    stage_index: int
    action_type: str = "treatment"  # pretreat, treatment, pulse, chase, washout, recovery
    duration: Optional[float] = None
    duration_unit: Optional[str] = None
    result: Optional["AlignResult"] = None
    description: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "stage_index": self.stage_index,
            "action_type": self.action_type,
            "duration": self.duration,
            "duration_unit": self.duration_unit,
            "result": self.result.as_dict() if self.result else None,
            "description": self.description,
        }


# 5. Multi-Omics Profiles and Embeddings
@dataclass
class NCBIGeneAnnotation:
    """NCBI Entrez Gene official annotation."""
    gene_id: Optional[Union[int, str]] = None
    symbol: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None
    chromosome: Optional[str] = None
    map_location: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    ensembl_id: Optional[str] = None
    omim_id: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class STRINGInteraction:
    """Pairwise protein-protein interaction information."""
    partner: str
    combined_score: float
    physical_score: Optional[float] = None
    functional_score: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class STRINGNetwork:
    """STRING protein-protein interaction (PPI) network profile."""
    query_gene: str
    interactors: list[str] = field(default_factory=list)
    interactions: list[STRINGInteraction] = field(default_factory=list)
    network_degree: int = 0
    top_partners: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "query_gene": self.query_gene,
            "interactors": self.interactors,
            "interactions": [i.as_dict() for i in self.interactions],
            "network_degree": self.network_degree,
            "top_partners": self.top_partners,
        }


@dataclass
class DepMapDependency:
    """Broad DepMap (Achilles) whole-genome CRISPR gene dependency score (Chronos)."""
    gene_symbol: str
    chronos_score: Optional[float] = None  # Lower negative score indicates higher dependency (< -0.5)
    dependency_probability: Optional[float] = None
    is_common_essential: bool = False
    is_strongly_selective: bool = False
    dependent_cell_lines_count: int = 0
    top_correlated_genes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProteinEmbedding:
    """ESM-2 (Evolutionary Scale Modeling) protein representation vector."""
    target: str
    model_name: str = "esm2_t6_8M_UR50D"
    dim: int = 320
    vector: list[float] = field(default_factory=list)
    pooling: str = "mean"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChemicalEmbedding:
    """ChemBERTa structural self-supervised representation vector."""
    target: str
    model_name: str = "ChemBERTa-77M-MLM"
    dim: int = 768
    vector: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class L1000Signature:
    """LINCS L1000 transcriptomic expression signature profile."""
    compound: str
    tas: float = 0.0  # Transcriptional Activity Score
    signature_strength: float = 0.0
    top_upregulated_genes: list[str] = field(default_factory=list)
    top_downregulated_genes: list[str] = field(default_factory=list)
    landmark_vector: Optional[list[float]] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PRISMSummary:
    """DepMap PRISM 500+ cancer cell line drug viability sensitivity profile."""
    compound: str
    median_auc: Optional[float] = None
    selective_index: float = 0.0
    sensitive_lineages: list[str] = field(default_factory=list)
    resistant_lineages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class JUMPFeatures:
    """JUMP-Cell Painting multi-channel morphological profile."""
    compound_or_pert: str
    mas: float = 0.0  # Morphological Activity Score
    phenotypic_strength: float = 0.0
    channel_impacts: dict[str, float] = field(default_factory=dict)
    top_phenotypes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeneticProfile:
    """Genetic perturbation multi-omics profile (NCBI, ESM2, STRING, DepMap)."""
    gene_symbol: str
    action: Optional[str] = None
    ncbi: Optional[NCBIGeneAnnotation] = None
    esm2_embedding: Optional[ProteinEmbedding] = None
    string_network: Optional[STRINGNetwork] = None
    depmap: Optional[DepMapDependency] = None
    extra_metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "gene_symbol": self.gene_symbol,
            "action": self.action,
            "ncbi": self.ncbi.as_dict() if self.ncbi else None,
            "esm2_embedding": self.esm2_embedding.as_dict() if self.esm2_embedding else None,
            "string_network": self.string_network.as_dict() if self.string_network else None,
            "depmap": self.depmap.as_dict() if self.depmap else None,
            "extra_metadata": self.extra_metadata,
        }

    @property
    def embedding_vector(self) -> Optional[list[float]]:
        return self.esm2_embedding.vector if self.esm2_embedding else None


@dataclass
class ChemicalProfile:
    """Small molecule perturbation multi-dimensional profile (PubChem, ChemBERTa, L1000, PRISM, JUMP)."""
    canonical_name: str
    smiles: Optional[str] = None
    inchikey: Optional[str] = None
    pubchem_cid: Optional[int] = None
    chembl_id: Optional[str] = None
    targets: list[str] = field(default_factory=list)
    moa: Optional[str] = None
    chemberta_embedding: Optional[ChemicalEmbedding] = None
    l1000: Optional[L1000Signature] = None
    prism: Optional[PRISMSummary] = None
    jump_cp: Optional[JUMPFeatures] = None
    extra_metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "pubchem_cid": self.pubchem_cid,
            "chembl_id": self.chembl_id,
            "targets": self.targets,
            "moa": self.moa,
            "chemberta_embedding": self.chemberta_embedding.as_dict() if self.chemberta_embedding else None,
            "l1000": self.l1000.as_dict() if self.l1000 else None,
            "prism": self.prism.as_dict() if self.prism else None,
            "jump_cp": self.jump_cp.as_dict() if self.jump_cp else None,
            "extra_metadata": self.extra_metadata,
        }

    @property
    def embedding_vector(self) -> Optional[list[float]]:
        return self.chemberta_embedding.vector if self.chemberta_embedding else None


@dataclass
class PerturbationProfile:
    """Unified wrapper for multi-omics perturbation profiles across all modalities."""
    query: str
    modality: str
    genetic_profile: Optional[GeneticProfile] = None
    chemical_profile: Optional[ChemicalProfile] = None

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "modality": self.modality,
            "genetic_profile": self.genetic_profile.as_dict() if self.genetic_profile else None,
            "chemical_profile": self.chemical_profile.as_dict() if self.chemical_profile else None,
        }

    @property
    def embedding_vector(self) -> Optional[list[float]]:
        if self.genetic_profile and self.genetic_profile.esm2_embedding:
            return self.genetic_profile.esm2_embedding.vector
        if self.chemical_profile and self.chemical_profile.chemberta_embedding:
            return self.chemical_profile.chemberta_embedding.vector
        return None


@dataclass
class AlignResult:
    """Complete alignment result for an input perturbation record."""

    input: str
    status: Union[AlignStatus, str]
    is_control: bool = False
    modality: Union[Modality, str] = MODALITY_SMALL_MOLECULE
    parsed_dose: Optional[float] = None
    parsed_unit: Optional[str] = None
    parsed_duration: Optional[float] = None
    parsed_duration_unit: Optional[str] = None
    parsed_name: Optional[str] = None
    genetic_target: Optional[str] = None
    genetic_action: Optional[str] = None
    sources_resolved: list[str] = field(default_factory=list)
    note: Optional[str] = None
    entity: Optional[PerturbationEntity] = None
    candidates: list[Candidate] = field(default_factory=list)
    parts: list["AlignResult"] = field(default_factory=list)
    regime_type: str = "single"  # single, simultaneous, sequential, pulse_chase
    stages: list[TreatmentStage] = field(default_factory=list)
    profile: Optional[Union[GeneticProfile, ChemicalProfile, PerturbationProfile]] = None

    def as_dict(self) -> dict:
        return {
            "input": self.input,
            "status": str(self.status),
            "is_control": self.is_control,
            "modality": str(self.modality),
            "parsed_dose": self.parsed_dose,
            "parsed_unit": self.parsed_unit,
            "parsed_duration": self.parsed_duration,
            "parsed_duration_unit": self.parsed_duration_unit,
            "parsed_name": self.parsed_name,
            "genetic_target": self.genetic_target,
            "genetic_action": self.genetic_action,
            "sources_resolved": self.sources_resolved,
            "note": self.note,
            "entity": self.entity.as_dict() if self.entity else None,
            "candidates": [c.as_dict() for c in self.candidates],
            "parts": [p.as_dict() for p in self.parts],
            "regime_type": self.regime_type,
            "stages": [s.as_dict() for s in self.stages],
            "profile": self.profile.as_dict() if (self.profile and hasattr(self.profile, "as_dict")) else None,
        }

    to_dict = as_dict

    @property
    def combo_names(self) -> list[str]:
        """List of canonical component names for combination perturbations."""
        if not self.parts:
            return []
        return [
            p.entity.canonical_name if (p.entity and p.entity.canonical_name) else (p.parsed_name or p.input)
            for p in self.parts
        ]

    @property
    def combo_smiles(self) -> list[Optional[str]]:
        """List of canonical SMILES strings for components in combination perturbations."""
        if not self.parts:
            return []
        return [
            getattr(p.entity, "canonical_smiles", None) if p.entity else None
            for p in self.parts
        ]

    @property
    def combo_doses(self) -> list[Optional[float]]:
        """List of doses for components in combination perturbations."""
        if not self.parts:
            return []
        return [p.parsed_dose for p in self.parts]

    @property
    def combo_units(self) -> list[Optional[str]]:
        """List of dose units for components in combination perturbations."""
        if not self.parts:
            return []
        return [p.parsed_unit for p in self.parts]

    @property
    def combo_targets(self) -> list[str]:
        """Deduplicated union of targets for all components in combination perturbations."""
        if not self.parts:
            return []
        targets = []
        for p in self.parts:
            if p.entity and getattr(p.entity, "targets", None):
                targets.extend(p.entity.targets)
            elif p.genetic_target:
                targets.append(p.genetic_target)
        return list(dict.fromkeys(targets))

    @property
    def stage_names(self) -> list[str]:
        """List of stage names or descriptions in temporal / pulse-chase regimens."""
        if not self.stages:
            return []
        names = []
        for s in self.stages:
            if s.result and (s.result.parsed_name or s.result.input):
                names.append(s.result.parsed_name or s.result.input)
            elif s.description:
                names.append(s.description)
            else:
                names.append(f"Stage {s.stage_index} ({s.action_type})")
        return names

    @property
    def stage_durations(self) -> list[tuple[float | None, str | None]]:
        """List of (duration, duration_unit) pairs in temporal / pulse-chase regimens."""
        return [(s.duration, s.duration_unit) for s in self.stages]


# 6. Knowledge-Enriched Perturbation Entity
@dataclass
class EnrichedPerturbation:
    """Standard knowledge-enriched perturbation entity.

    Encapsulates:
    1. Standardized entity cleaning attributes (canonical_name, dose, duration, modality);
    2. Global persistent identifiers (PubChem CID, SMILES, InChIKey, ChEMBL ID, Entrez ID, Ensembl, UniProt);
    3. Multi-source domain knowledge profiles (physicochemical properties, targets, MOA, clinical phase, pathways, PPI).
    """
    raw_input: str
    canonical_name: str
    modality: str
    is_control: bool = False
    dose: Optional[float] = None
    dose_unit: Optional[str] = None
    duration: Optional[float] = None
    duration_unit: Optional[str] = None
    identifiers: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_input": self.raw_input,
            "canonical_name": self.canonical_name,
            "modality": self.modality,
            "is_control": self.is_control,
            "dose": self.dose,
            "dose_unit": self.dose_unit,
            "duration": self.duration,
            "duration_unit": self.duration_unit,
            "identifiers": self.identifiers,
            "properties": self.properties,
            "extra_metadata": self.extra_metadata,
        }

    to_dict = as_dict

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=indent)
