# PerturbAlign Data Directory

This directory contains test benchmark fixtures, mapping tables, and reproducible scripts for building external perturbation corpora.

## Structure

```
data/
├── benchmark/
│   └── benchmark_cases.json    # Standardized benchmark evaluation test cases (68 KB)
├── drug_name_corpus/
│   ├── download_sources.py     # Script to download raw labels from 17 public sources
│   ├── merge_names.py          # Script to merge and deduplicate downloaded sources
│   ├── scan_patterns.py        # Script to inspect pattern frequencies in downloaded corpus
│   └── README.md               # Detailed documentation of public data sources
└── reference/
    ├── cdsdb_smiles_map.final.tsv
    ├── cdsdb_smiles_map.tsv
    ├── combosciplex_drug_aliases.tsv
    └── combosciplex_smiles.tsv
```

## Large Data Archive Policy

To ensure the repository remains ultra-lightweight (< 2 MB) and fast to clone/install:
- Raw downloads (`raw/*.txt`, `lincs_sig_info.txt.gz`) and generated corpora (`all_drug_names.txt`, ~205k labels) are archived outside the core repository.
- Intermediate evaluation scans (`external_unmatched_scan.json`, `auto_vocab_candidates.json`) are stored in the external archive.
- You can regenerate or fetch the raw corpus anytime by running:
  ```bash
  python data/drug_name_corpus/download_sources.py
  python data/drug_name_corpus/merge_names.py
  ```
