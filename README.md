# PerturbAlign 🧬

[English](README.md) | [中文](README_zh.md)

**PerturbAlign** is a specialized toolkit for perturbation data cleaning, entity alignment, causal knowledge enrichment, and embedded graph querying designed for single-cell perturbation biology (Perturb-seq, Sci-Plex, Mix-seq, L1000) and multi-omics knowledge systems.

It directly addresses core challenges:
1. **Perturbation Metadata Standardization**: Cleans and disambiguates noisy raw experimental labels (containing salt forms, solvents, dosages, timepoints, trade names, and investigational codes) into controlled fields adhering to the **UPDS (Unified Perturbation Data Schema v1.0)** specification.
2. **Causal Regulatory Knowledge Enrichment**: Connects perturbed targets to downstream causal regulatory chains (**Kinase $\to$ Transcription Factors (TFs) $\to$ DoRothEA Curated Target Regulons**), supporting host cell line mutation-specific context pruning.
3. **Embedded Graph Database Backend**: Built-in ACID SQLite graph database (`SQLiteGraphStore`) supporting SQL Recursive Common Table Expressions (CTE) for ultra-fast multi-hop causal path search without requiring external graph server infrastructure.
4. **Multimodal Deep Embeddings**: Native continuous biological representations via pretrained ESM2 protein language models and ChemBERTa molecular embeddings, alongside pluggable provider interfaces for single-cell AnnData and MuData pipelines.

> 📖 **Full Interactive Tutorial**: Refer to the accompanying Jupyter Notebook: [notebooks/quickstart_tutorial.ipynb](notebooks/quickstart_tutorial.ipynb).

---

## 1. Supported Perturbation Modalities & Data Structures

### 1.1 Modalities (`pert_modality`)

PerturbAlign categorizes perturbations strictly by biological mechanism into the following controlled modalities:

| Modality (`pert_modality`) | Description & Supported Nomenclature | Typical Input Examples | Standardization Objective |
| :--- | :--- | :--- | :--- |
| **Small Molecule**<br>`small_molecule` | International Nonproprietary Names (INN), trade names, SMILES, investigational codes (AZD, GDC, BMS, etc.), vendor catalog numbers | `Doxorubicin HCl 10nM 24h`<br>`AZD-6244 1uM`<br>`Selleck-S1001` | Strip salt forms and vendor prefixes, resolve investigational codes to canonical names (e.g. `Selumetinib`), output canonical SMILES and HGNC targets |
| **Genetic Perturbation**<br>`crispr_ko` / `crispr_a` / `crispr_i`<br>`rnai` / `overexpression` | CRISPR knockout/activation/inhibition, RNAi knockdown, plasmid overexpression; cross-species gene symbols | `sgTrp53 KO`<br>`MYC_OE`<br>`shBRCA1` | Identify mechanism of action (KO/a/i/OE/shRNA), extract target gene, map orthologs to human HGNC Symbol (`TP53`) |
| **Biologics & Cytokines**<br>`biologic` | Recombinant cytokines, interleukins, interferons, monoclonal antibodies (`-mab`), receptor fusion proteins (`-cept`) | `TNF-alpha 50ng/ml`<br>`IL-6 100ng/ml`<br>`Trastuzumab` | Standardize nomenclature, match surface membrane receptors (e.g., `TNFRSF1A`) and downstream signaling pathways |
| **Environmental & Physical Stress**<br>`environmental` | Oxygen levels, temperature, pH, nutrient starvation, and cell culture microenvironmental stresses | `Hypoxia 1% O2 12h`<br>`Heat shock 42C`<br>`Glucose starvation` | Normalize to MSigDB Hallmark stress systems, associate stress-sensing transcription factors (e.g., `HIF1A`, `HSF1`) |
| **Combinations**<br>`combination` | Multi-drug combinations, cocktail regimens, or dual-gene knockouts | `Trametinib + Pictilisib`<br>`sgCDKN1A + sgTP53` | Decompose components using safety delimiter rules, resolve individually, and assemble synergistic regulatory profiles |
| **Negative Control**<br>`control` | Vehicle controls, blank vectors, non-targeting sgRNA controls | `DMSO 0.1%`<br>`Vehicle`<br>`sgCtrl (non-targeting)` | Mark as negative control (`pert_is_control=True`), zero out baseline features |

### 1.2 Supported Data Structures

- **`AnnData`**: Single-cell standard container; automatically extracts perturbation columns from `adata.obs` and writes back standard columns and the `adata.uns["perturbalign"]` knowledge namespace;
- **Pandas `Series` / `DataFrame`**: Standard tabular data with vectorized batch processing support;
- **Python Lists / Strings**: Single or batch perturbation queries.

---

## 2. Perturbation Cleaning & Alignment (External Vocabulary System)

Biological experiment nomenclature involves extensive historical discrepancy and noise. PerturbAlign decouples all cleaning, disambiguation, and classification rules into **22 transparent, independently maintainable offline TSV vocabularies**, avoiding opaque hardcoding:

### 2.1 The 22 External TSV Vocabularies

```
resources/
├── Disambiguation & Prefix Stripping
│   ├── investigational_codes.tsv  # 180+ pharma codes (AZD, GDC, BMS, ABT, PD, PF, etc.) mapped to INN & canonical SMILES
│   ├── vendor_prefixes.tsv        # 28 commercial chemical vendor prefixes (Selleck, MCE, Tocris, Sigma, etc.)
│   └── formulation_codes.tsv      # Pharmaceutical formulation and control codes (XR, SR, CIV)
├── Chemical Desalting & Counterion Safeguards
│   ├── salt_suffixes.tsv          # Salt/acid counterion suffixes (HCl, mesylate, tartrate, sodium, etc.)
│   ├── metal_cations.tsv          # Metal cation salt prefix identification (Sodium, Potassium, etc.)
│   ├── inorganic_compounds.tsv    # Inorganic compound counterion protection (preserves NaCl, KCl, etc.)
│   ├── stereo_tokens.tsv          # Stereochemical descriptor normalization ((R)-, (S)-, cis-, trans-)
│   └── greek_transliteration.tsv  # Greek letter transliteration (α -> alpha, β -> beta)
├── Genetic Action & Control Identification
│   ├── genetic_actions.tsv        # Genetic perturbation mechanism terms (KO, CRISPRa, CRISPRi, RNAi, OE)
│   ├── genetic_control_targets.tsv# Genetic negative control targets (NT, Scramble, AAVS1, Luciferase)
│   ├── sgrna_reference.tsv        # sgRNA nomenclature and prefix reference table
│   └── control_tokens.tsv         # Vehicle and solvent controls (DMSO, Vehicle, Untreated, Water, etc.)
├── Biologics, Delivery & Microenvironment
│   ├── biologics_tokens.tsv       # Cytokines and biologic macromolecule tokens (TNF, IL-6, Interferon, etc.)
│   ├── biologic_suffixes.tsv      # Biologic INN suffixes (-mab monoclonal antibody, -cept fusion protein)
│   ├── delivery_vehicles.tsv      # Delivery vehicles (LNP, Liposome, Electroporation)
│   ├── environmental_tokens.tsv   # Microenvironmental and physical stress tokens (Hypoxia, Starvation, Acidosis, etc.)
│   └── protac_e3_ligands.tsv      # PROTAC degraders and E3 ubiquitin ligase ligands
└── Dimensions & Combination Parsing
    ├── dose_units.tsv             # Concentration units and conversion factors (nM, uM, mM, M, mg/ml)
    ├── duration_units.tsv         # Time duration units and conversion factors (h, hr, min, d, days)
    ├── combo_separators.tsv       # Safe combination delimiters (+, /, ;, AND) with quality gates
    ├── cocktails.tsv              # Classic multi-drug cocktail regimens (R-CHOP, FOLFOX, etc.)
    └── perturbation_protocols.tsv # Public perturbation protocol metadata taxonomy (LINCS L1000, scPerturb)
```

### 2.2 Alignment Output: UPDS (Unified Perturbation Data Schema v1.0) Specification

After processing with `perturbalign.standardize_anndata(adata)`, 10 standardized columns are populated in `adata.obs`:

1. `pert_canonical_name`: Standardized official name (all-caps canonical term, e.g., `SELUMETINIB`, `TP53`)
2. `pert_modality`: Controlled modality enum (`small_molecule`, `crispr_ko`, `crispr_a`, `crispr_i`, `rnai`, `overexpression`, `biologic`, `environmental`, `combination`, `control`)
3. `pert_is_control`: Boolean flag indicating negative control status (`True` / `False`)
4. `pert_dose_val`: Parsed raw numeric dosage
5. `pert_dose_unit`: Raw dosage unit string
6. `pert_dose_norm_um`: **Strictly standardized concentration in micromolar ($\mu\text{M}$)** (supports automatic lossless conversion from nM, mM)
7. `pert_duration_h`: **Strictly standardized duration in hours ($h$)** (supports automatic lossless conversion from min, d)
8. `pert_human_ortholog`: **Cross-species aligned human HGNC Symbol** (e.g., mouse `Trp53` $\to$ `TP53`)
9. `pert_primary_targets`: Semicolon-delimited list of primary HGNC targets (e.g., `MAP2K1;MAP2K2`)
10. `pert_smiles`: Canonical stereochemically unambiguous SMILES string

---

## 3. Multi-Dimensional Knowledge Enrichment & Causal Cascades

Following cleaning and alignment, PerturbAlign bridges discrete perturbation labels to biological knowledge graphs to infer mechanistic causal cascades.

### 3.1 Integrated Knowledge Providers

- **Chemistry & Pharmacology**: PubChem (physicochemical properties/SMILES), DrugBank (targets/MoA), ChEMBL (clinical phase/indications), openFDA (adverse events), BindingDB (affinity constants);
- **Genes & Pathways**: NCBI Gene (Entrez ID/chromosomal locus), Reactome (biological pathways), STRING (protein-protein interactions / PPI), DepMap (CRISPR dependency scores);
- **Causal Regulation & Cellular Context**: DoRothEA (Confidence A/B curated TF-target regulons), Cellosaurus / DepMap (cell line driver mutations and genetic background).

### 3.2 Mechanism: Causal Signaling Cascade

Perturbation response in a cell is not a black box. PerturbAlign constructs mechanistic causal chains from direct targets down to transcriptional outputs:

$$\text{Primary Perturbed Target} \longrightarrow \text{Kinase/Receptor Signaling Pathways} \longrightarrow \text{Core Convergent Transcription Factors (TFs)} \longrightarrow \text{DoRothEA Curated Responsive Regulons}$$

### 3.3 Context-Specific Pruning for Host Cell Lines

Identical perturbations produce distinct downstream responses across varying host genetic backgrounds. PerturbAlign incorporates host cell line context awareness:

- **TP53-Deficiency Pruning**: In **TP53-mutant / deficient cell lines** (e.g., `K562`, `Jurkat`, `HL-60`), when exposed to DNA-damaging agents (such as Doxorubicin / TOP2A inhibitors), the dysfunctional `TP53` branch is automatically pruned from active transcription factors into `pruned_tfs`. Exclusively p53-regulated downstream genes are removed from responsive gene sets, and confidence is attenuated to `0.65 (Context-Attenuated)`.
- **Driver Mutation Reinforcement**: When targets coincide with host oncogenic drivers (e.g., MEK/KRAS pathway inhibition in an A549 lung adenocarcinoma context), causal confidence is boosted accordingly up to `0.97`.

---

## 4. Quick Start

### 4.1 Python API Example

```python
import scanpy as sc
import perturbalign

# 1. Standardize single-cell AnnData (generates UPDS 10 standard columns)
adata = sc.read_h5ad("raw_screen.h5ad")
adata = perturbalign.standardize_anndata(
    adata,
    obs_col="perturbation",      # Raw perturbation column
    cell_line_col="cell_line",   # Host cell line column (e.g., K562, A549)
    online=False,                # Offline mode prioritizes local TSV dictionaries and cache
)

# 2. Infer causal cascade and perform cell line-specific pruning
cascade = perturbalign.infer_causal_cascade(["TOP2A"], cell_line="K562")
print("Pruned inactive TFs:", cascade["pruned_tfs"])         # Flags TP53 as pruned due to mutation
print("Causal confidence score:", cascade["cascade_confidence_score"])  # 0.65

# 3. Offline investigational pharma code disambiguation
drug = perturbalign.query_compound("AZD-6244", online=False)
print("Canonical Name:", drug["pubchem"]["canonical_name"])      # Selumetinib
print("SMILES:", drug["pubchem"]["canonical_smiles"])            # Canonical stereochemical structure

# 4. Cross-species ortholog mapping
gene = perturbalign.query_gene("Trp53", organism="mouse")
print("Human Ortholog:", gene["ncbi"]["symbol"])                 # TP53
```

### 4.2 Embedded Graph Database & Causal Path Querying

```python
# Create or persist knowledge graph to an embedded SQLite database
kg = perturbalign.HeterogeneousKnowledgeGraph(db_path="perturbation_kg.sqlite")

# Query multi-hop causal paths using SQLite Recursive CTE
paths = kg.find_paths_sql("Compound:Trametinib", "Gene:MAP2K1", max_hops=2)
for p in paths:
    print(p.format_path())
```

### 4.3 Tutorial Notebook

For complete end-to-end demonstrations (including embedded SQLite graph querying and multi-hop causal traversal), launch the Jupyter tutorial:
```bash
jupyter notebook notebooks/quickstart_tutorial.ipynb
```

---

## 5. Pluggable Architecture & Hot-Swappable Extensions

PerturbAlign is architected with a decoupled, hot-swappable extension system across all foundational layers, enabling seamless integration of proprietary models, in-house databases, custom assays, and experimental protocols:

### 5.1 Pluggable Embedding Models (`register_embedding_provider`)
Swap or extend perturbation embedding models (e.g., Uni-Mol, MolFormer, Geneformer, scGPT, or custom internal representations):
```python
from perturbalign import BaseEmbeddingProvider, register_embedding_provider, MODALITY_SMALL_MOLECULE

class UniMolProvider(BaseEmbeddingProvider):
    name = "unimol"
    def embed(self, target: str, model=None, cache=None, **kwargs) -> list[float]:
        return custom_unimol_encoder(target)

register_embedding_provider(MODALITY_SMALL_MOLECULE, UniMolProvider())
```

### 5.2 Dynamic Assay Profile Providers (`register_profile_provider`)
Plug in novel experimental assays (e.g., DIA Proteomics, Perturb-ATAC peak accessibility, or custom cellular screens):
```python
from perturbalign import BaseProfileProvider, register_profile_provider

class ProteomicsProvider(BaseProfileProvider):
    assay_name = "proteomics"
    def get_profile(self, target: str, cache=None, **kwargs):
        return {"assay": "proteomics", "target": target, "hits": ["AKT1", "MTOR"]}

register_profile_provider("proteomics", ProteomicsProvider(), aliases=["dia_proteomics"])
```

### 5.3 Universal Vocabulary Registry (`register_vocabulary`)
All domain dictionaries (investigational codes `investigational_codes`, vendor prefixes `vendor_prefixes`, control tokens `control_tokens`, salt suffixes `salt_suffixes`, biologics `biologics_tokens`, or arbitrary custom entity tables) can be registered, extended, or overridden uniformly via `register_vocabulary` (supporting TSV filepaths or in-memory dict/set/list):
```python
from perturbalign import register_vocabulary, get_vocabulary, resolve_investigational_code, align

# 1. Register or override in-house investigational compound codes
register_vocabulary("investigational_codes", {
    "CUSTOM-999": {
        "code": "CUSTOM-999",
        "canonical_name": "Antigravib",
        "targets": ["BRAF", "KRAS"],
        "class_name": "Kinase Inhibitor",
        "developer": "InHouse",
    }
})
# Immediately resolved across the alignment and candidate engine
res = resolve_investigational_code("CUSTOM 999")
assert res["canonical_name"] == "Antigravib"

# 2. Extend vendor prefixes or control tokens
register_vocabulary("vendor_prefixes", ["MYVENDOR-", "CHEM-"])

# 3. Register custom domain dictionaries
register_vocabulary("my_disease_indications", {"AML": "Acute Myeloid Leukemia"})
```

### 5.4 Epigenetic Elements, Causal Cascades, and Target Safety Extensions
- **Cis-Regulatory Elements (cCREs)**: `register_ccre(gene, ccre)` to link custom promoter/enhancer elements
- **Signaling Cascades & Regulons**: `register_target_cascade(kinase, tfs, pathway)` & `register_dorothea_regulon(tf, targets)`
- **Host Cell Line & Safety Screen Profiles**: `register_cell_line(entity)`, `register_safety_target(symbol, profile)`, & `register_safety_compound(drug, profile)`
- **Reference Transcriptional Signatures**: `register_reference_signature(drug, signature_dict)`

---

## 6. Architectural Ecosystem

- **`perturbalign`** (Full Package): Complete feature suite including AnnData featurization, continuous deep embeddings (requiring loaded model weights), heterogeneous causal inference, and embedded SQLite graph database (`SQLiteGraphStore`).
- **`perturbalign-core`** (Core Sub-package): **Zero heavyweight dependencies**, relying only on the Python standard library and 22 offline TSV vocabularies. Ideal for high-throughput production microservices and ETL preprocessing pipelines.
- **AI Agent Integration (MCP Server)**: Built-in Model Context Protocol server, launched via `perturbalign mcp`, enabling direct tool calling from LLM clients like Cursor and Claude Desktop.

---

## License

This project is licensed under the [MIT License](LICENSE).

