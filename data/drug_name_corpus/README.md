# PerturbAlign Drug Name Corpus (drug_name_corpus)

Drug and compound perturbation labels downloaded from 17 public data sources, merged and deduplicated into **205,958** unique labels.
Used for protocol coverage review and benchmarking of the PerturbAlign standardization middleware.

## File Structure

```
drug_name_corpus/
├── README.md                  # This file: Data source inventory
├── all_drug_names.txt         # Deduplicated corpus of all labels (205,958 lines, one per line)
├── download_sources.py        # Minimal downloader script (direct connection from each source)
├── merge_names.py             # Minimal merge script (deduplicates and merges to all_drug_names.txt)
├── scan_patterns.py           # Minimal pattern scanning script (salt forms / doses / controls / combos)
└── raw/                       # Raw downloads from each source (retained as needed)
```

## Data Sources Inventory

### 1. LINCS L1000 (Perturbation Labels)
- **Source**: NIH GEO GSE92742 Broad LINCS siginfo
- **URL**: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz
- **Label Count**: 28,927 (pert_iname column, containing small molecules / controls / genetic perturbations)
- **File**: `raw/lincs_sig_info.txt.gz` (10.9 MB, raw gzip)
- **Features**: Largest perturbation source, containing extensive control tokens (DMSO/GFP/empty vector) and compound names.

### 2. Broad Drug Repurposing Hub (= PRISM Compound Library)
- **Source**: Clue.io S3 Storage
- **Drugs Table URL**: https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt
- **Samples Table URL**: https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_samples_20200324.txt
- **Label Count**: 6,798 (unique drug names) + 13,563 (samples table, including vendor catalog numbers)
- **Files**: `raw/repurposing_drugs_20200324.txt`, `raw/repurposing_samples_20200324.txt`
- **Features**: Contains numerous vendor catalog IDs (~9,000 BRD-/NSC-/SA- codes).

### 3. ChEMBL (Aggregated Endpoints)
- **Source**: EBI ChEMBL API (https://www.ebi.ac.uk/chembl/api/data)
- **Endpoints**:
  - FDA Drugs: `/drug?limit=100` paginated -> 1,276 synonyms
  - Full Molecules: `/molecule?molecule_type=Small molecule` paginated -> 14,821
  - Molecule Endpoint: `/molecule/{id}` -> 1,585
  - Molecule by initial letter a-z: -> 11,880 (v3), 26,232 (v5, x3 pages)
  - Clinical Phases 1-4: `/molecule?molecule_properties__max_phase=N` -> 38,241
  - Bioactivity: `/activity?standard_type=IC50` -> 10
- **Files**: `raw/chembl_*.txt` (7 files)
- **Features**: One of the largest drug name sources, containing INN generic names, brand names, investigational codes, and salt form variants.

### 4. KEGG (Multi-Database Compilation)
- **Source**: KEGG REST API (https://rest.kegg.jp/list/{db})
- **Databases**:
  - DRUG: `/list/drug` -> 22,049 (contains (JAN)/(TN) pharmacopeia notations, salts, radiopharmaceuticals)
  - COMPOUND + GLYCAN: `/list/compound` + `/list/glycan` -> 41,098 (contains glycan abbreviations, ion charges, polymers)
  - ENZYME + REACTION: `/list/enzyme` + `/list/reaction` -> 50,938 (contains enzyme names, chemical reaction formulas, coenzyme flags)
- **Files**: `raw/kegg_drug_names.txt`, `raw/kegg_compound_glycan_names.txt`, `raw/kegg_enzyme_reaction_names.txt`
- **Features**: Largest compound/metabolite source, containing 12,434 chemical reaction formulas (`<=>`), glycan abbreviations, and phosphorylation position markers.

### 5. openFDA Drug Label
- **Source**: openFDA API (https://api.fda.gov/drug/label.json)
- **URL**: `https://api.fda.gov/drug/label.json?search=openfda.generic_name:*&limit=100&skip=N`
- **Label Count**: 6,882
- **File**: `raw/openfda_drug_names.txt`
- **Features**: Contains oral dosage units (MG/MCG/ML), homeopathic multi-component names, viral strain designations, and USP pharmacopeia suffixes.

### 6. Wikidata
- **Source**: Wikidata SPARQL (https://query.wikidata.org/sparql)
- **Label Count**: 901 (v1) + 3,154 (v2, including compounds/pharmaceuticals/biomedical substances)
- **Files**: `raw/wikidata_drug_names.txt`, `raw/wikidata_drug_names_v2.txt`
- **Features**: Contains biologics (species/brand parentheses), and generic chemical names.

### 7. SIDER (Adverse Drug Reactions)
- **Source**: SIDER Database (http://sideeffects.embl.de/)
- **Label Count**: 1,344
- **File**: `raw/sider_drug_names.txt`
- **Features**: Drug name and adverse effect associations with standardized naming.

### 8. RxNorm
- **Source**: NLM RxNorm API (https://rxnav.nlm.nih.gov/)
- **Label Count**: 378
- **File**: `raw/rxnorm_drug_names.txt`
- **Features**: US prescription generic names including brand names.

### 9. DailyMed
- **Source**: DailyMed API (https://dailymed.nlm.nih.gov/dailymed/)
- **Label Count**: 95
- **File**: `raw/dailymed_drug_names.txt`
- **Features**: US drug labeling, containing leading % concentrations.

### 10. PubChem (Sampled)
- **Source**: PubChem PUG REST (https://pubchem.ncbi.nlm.nih.gov/rest/pug)
- **Label Count**: 194 (sampled due to rate limits)
- **File**: `raw/pubchem_drug_names.txt`
- **Features**: Compound generic names.

### 11. GEO Perturbations (Sampled)
- **Source**: NCBI GEO
- **Label Count**: 2
- **File**: `raw/geo_perturbation_names.txt`

## Verified Inactive / Unavailable Data Sources (to avoid repeated attempts)

| Source | Failure Reason |
|--------|----------------|
| GDSC (Sanger) | API 410 Gone, FTP 404 |
| DrugBank | 403 Login required, API 401 |
| BindingDB | All endpoints 404 |
| HMDB | All endpoints 403 |
| PharmGKB | SSL UNEXPECTED_EOF |
| TTD | All download URLs ReadTimeout |
| ChEBI | API connection error / no output |
| NCI Drug Dictionary | HTML structure mismatch, JSON parse error |
| ClinicalTrials v2 | 0 drug names (parameter mismatch) |
| DepMap API | compound_metadata returns empty |
| CTRPv2 (Broad) | 404 |
| ChemPert (GitLab) | 404 |
| LOTUS | GitHub/Zenodo URLs inactive |
| STITCH | Memory allocation error |
| MetaboLights | metabolites endpoint returns string instead of list |
| Reactome | 771 disease names (non-drug names) |
| DisGeNET | JSON parse error |
| OpenTargets | GraphQL 400 |
| DrugCentral | 404 |
| DGIdb | Returned non-JSON |
| ZINC / EPA CompTox / WHO EML | API changed / no output |
| IUPHAR Guide to Pharmacology | 401 |
| PDBe ligands | 404 |

## Protocol Coverage

Through 10 rounds of comprehensive auditing, **66 naming protocol categories** have been identified.
Key categories include: salt suffixes (~93 types), dose units (~20 types), control tokens (~30 types), combination separators,
stereochemical notation, gene mutation tags, vendor catalog codes, formulation codes, chemical reaction formulas, glycan abbreviations, etc.

## Reproduction Steps

```bash
# 1. Download all sources
python download_sources.py

# 2. Merge and deduplicate
python merge_names.py

# 3. Pattern scan (salt forms / doses / controls / combos)
python scan_patterns.py
```

## Summary Statistics

- **Total Labels**: 205,958
- **Data Sources**: 17 successful (23 raw files)
- **File Size**: all_drug_names.txt ~5.3 MB
- **Audit Rounds**: 10 rounds of full manual and automated audit
- **Protocol Categories**: 66 categories
- **Benchmark Cases**: 148 cases
