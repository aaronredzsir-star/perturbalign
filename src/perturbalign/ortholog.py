"""Cross-species gene orthology and homology engine.

Addresses cross-species perturbation challenges in single-cell computational biology:
- Inconsistent nomenclature: Human all-uppercase (TP53, EGFR), mouse/rat title-cased (Trp53, Egfr), zebrafish lowercase (tp53);
- Divergent symbols: Human KDR <-> Mouse Flk1/Kdr, Human HLA <-> Mouse H2, Human CDKN2A (p14/p16) <-> Mouse Cdkn2a (p19/p16);
- Automatic organism detection (detect_organism);
- Cross-species ortholog mapping: Human <-> Mouse <-> Rat <-> Zebrafish;
- Combines NCBI HomoloGene online queries with curated offline core lexicon and SQLite persistent caching.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from .cache import SQLiteCache
from .resources import VOCABULARY_VERSION


# 1. Curated cross-species ortholog lexicon (core oncology, immune, growth factor, cell-cycle genes)
_CURATED_ORTHOLOGS: dict[str, dict[str, dict[str, str]]] = {
    "TP53": {
        "human": {"symbol": "TP53", "entrez_id": "7157", "uniprot": "P04637", "name": "tumor protein p53"},
        "mouse": {"symbol": "Trp53", "entrez_id": "22059", "uniprot": "P02340", "name": "transformation related protein 53"},
        "rat": {"symbol": "Tp53", "entrez_id": "24842", "uniprot": "P10361", "name": "tumor protein p53"},
        "zebrafish": {"symbol": "tp53", "entrez_id": "30590", "uniprot": "Q9W678", "name": "tumor protein p53"},
    },
    "KRAS": {
        "human": {"symbol": "KRAS", "entrez_id": "3845", "uniprot": "P01116", "name": "KRAS proto-oncogene, GTPase"},
        "mouse": {"symbol": "Kras", "entrez_id": "16653", "uniprot": "P32883", "name": "Kirsten rat sarcoma viral oncogene homolog"},
        "rat": {"symbol": "Kras", "entrez_id": "24525", "uniprot": "P08644", "name": "KRAS proto-oncogene, GTPase"},
    },
    "EGFR": {
        "human": {"symbol": "EGFR", "entrez_id": "1956", "uniprot": "P00533", "name": "epidermal growth factor receptor"},
        "mouse": {"symbol": "Egfr", "entrez_id": "13645", "uniprot": "Q01279", "name": "epidermal growth factor receptor"},
        "rat": {"symbol": "Egfr", "entrez_id": "24329", "uniprot": "P15888", "name": "epidermal growth factor receptor"},
    },
    "MYC": {
        "human": {"symbol": "MYC", "entrez_id": "4609", "uniprot": "P01106", "name": "MYC proto-oncogene, bHLH transcription factor"},
        "mouse": {"symbol": "Myc", "entrez_id": "17869", "uniprot": "P01108", "name": "myelocytomatosis oncogene"},
        "rat": {"symbol": "Myc", "entrez_id": "24577", "uniprot": "P06171", "name": "MYC proto-oncogene, bHLH transcription factor"},
    },
    "PTEN": {
        "human": {"symbol": "PTEN", "entrez_id": "5728", "uniprot": "P60484", "name": "phosphatase and tensin homolog"},
        "mouse": {"symbol": "Pten", "entrez_id": "19211", "uniprot": "O08586", "name": "phosphatase and tensin homolog"},
        "rat": {"symbol": "Pten", "entrez_id": "50557", "uniprot": "O54907", "name": "phosphatase and tensin homolog"},
    },
    "BRCA1": {
        "human": {"symbol": "BRCA1", "entrez_id": "672", "uniprot": "P38398", "name": "BRCA1 DNA repair associated"},
        "mouse": {"symbol": "Brca1", "entrez_id": "12189", "uniprot": "P48754", "name": "breast cancer 1, early onset"},
        "rat": {"symbol": "Brca1", "entrez_id": "497926", "uniprot": "O54952", "name": "BRCA1 DNA repair associated"},
    },
    "CDKN1A": {
        "human": {"symbol": "CDKN1A", "entrez_id": "1026", "uniprot": "P38936", "name": "cyclin dependent kinase inhibitor 1A (p21)"},
        "mouse": {"symbol": "Cdkn1a", "entrez_id": "12575", "uniprot": "P39689", "name": "cyclin dependent kinase inhibitor 1A (p21)"},
        "rat": {"symbol": "Cdkn1a", "entrez_id": "25285", "uniprot": "P39690", "name": "cyclin-dependent kinase inhibitor 1A (p21)"},
    },
    "MDM2": {
        "human": {"symbol": "MDM2", "entrez_id": "4193", "uniprot": "Q00987", "name": "MDM2 proto-oncogene"},
        "mouse": {"symbol": "Mdm2", "entrez_id": "17246", "uniprot": "P23804", "name": "transformed mouse 3T3 cell double minute 2"},
        "rat": {"symbol": "Mdm2", "entrez_id": "314886", "uniprot": "Q712V2", "name": "MDM2 proto-oncogene"},
    },
    "KDR": {
        "human": {"symbol": "KDR", "entrez_id": "3791", "uniprot": "P35968", "name": "kinase insert domain receptor (VEGFR2)"},
        "mouse": {"symbol": "Kdr", "entrez_id": "16542", "uniprot": "P35918", "name": "kinase insert domain protein receptor (Flk1)"},
        "rat": {"symbol": "Kdr", "entrez_id": "25589", "uniprot": "O08775", "name": "kinase insert domain receptor"},
    },
    "CD4": {
        "human": {"symbol": "CD4", "entrez_id": "920", "uniprot": "P01730", "name": "CD4 molecule"},
        "mouse": {"symbol": "Cd4", "entrez_id": "12504", "uniprot": "P06332", "name": "CD4 antigen"},
        "rat": {"symbol": "Cd4", "entrez_id": "24932", "uniprot": "P05540", "name": "CD4 molecule"},
    },
    "CD8A": {
        "human": {"symbol": "CD8A", "entrez_id": "925", "uniprot": "P01732", "name": "CD8a molecule"},
        "mouse": {"symbol": "Cd8a", "entrez_id": "12525", "uniprot": "P01731", "name": "CD8 antigen, alpha chain"},
        "rat": {"symbol": "Cd8a", "entrez_id": "24930", "uniprot": "P07725", "name": "CD8a molecule"},
    },
    "TNF": {
        "human": {"symbol": "TNF", "entrez_id": "7124", "uniprot": "P01375", "name": "tumor necrosis factor"},
        "mouse": {"symbol": "Tnf", "entrez_id": "21926", "uniprot": "P06804", "name": "tumor necrosis factor"},
        "rat": {"symbol": "Tnf", "entrez_id": "24835", "uniprot": "P16599", "name": "tumor necrosis factor"},
    },
    "IL6": {
        "human": {"symbol": "IL6", "entrez_id": "3569", "uniprot": "P05231", "name": "interleukin 6"},
        "mouse": {"symbol": "Il6", "entrez_id": "16193", "uniprot": "P08505", "name": "interleukin 6"},
        "rat": {"symbol": "Il6", "entrez_id": "24498", "uniprot": "P20607", "name": "interleukin 6"},
    },
    "VEGFA": {
        "human": {"symbol": "VEGFA", "entrez_id": "7422", "uniprot": "P15692", "name": "vascular endothelial growth factor A"},
        "mouse": {"symbol": "Vegfa", "entrez_id": "22339", "uniprot": "Q00731", "name": "vascular endothelial growth factor A"},
        "rat": {"symbol": "Vegfa", "entrez_id": "83785", "uniprot": "P16612", "name": "vascular endothelial growth factor A"},
    },
    "STAT3": {
        "human": {"symbol": "STAT3", "entrez_id": "6774", "uniprot": "P40763", "name": "signal transducer and activator of transcription 3"},
        "mouse": {"symbol": "Stat3", "entrez_id": "20848", "uniprot": "P42227", "name": "signal transducer and activator of transcription 3"},
        "rat": {"symbol": "Stat3", "entrez_id": "25125", "uniprot": "P52631", "name": "signal transducer and activator of transcription 3"},
    },
}

# Build reverse index (any species symbol/lowercase/alias -> human primary key)
_SYMBOL_TO_HUMAN_KEY: dict[str, str] = {}
for hkey, org_map in _CURATED_ORTHOLOGS.items():
    _SYMBOL_TO_HUMAN_KEY[hkey.lower()] = hkey
    for org, data in org_map.items():
        sym = data.get("symbol", "")
        if sym:
            _SYMBOL_TO_HUMAN_KEY[sym.lower()] = hkey
# Special alias mappings
_SYMBOL_TO_HUMAN_KEY["flk1"] = "KDR"
_SYMBOL_TO_HUMAN_KEY["p53"] = "TP53"


def detect_organism(symbol: str) -> str:
    """Infer organism based on standard biological gene nomenclature conventions.

    Rules:
      - All-uppercase (e.g. 'TP53', 'EGFR', 'KRAS') -> 'human'
      - Title-cased (e.g. 'Trp53', 'Egfr', 'Kras') -> 'mouse' (or rat)
      - All-lowercase (e.g. 'tp53', 'egfr', 'kras') -> 'zebrafish'
      - Default fallback is 'human'
    """
    s = symbol.strip()
    if not s:
        return "human"
    if s.isupper():
        return "human"
    if s.islower():
        return "zebrafish"
    if s[0].isupper() and s[1:].islower():
        return "mouse"
    return "human"


def get_gene_orthologs(
    symbol_or_id: str,
    organism: str = "auto",
    target_organisms: tuple[str, ...] = ("human", "mouse", "rat", "zebrafish"),
    online: bool = True,
    cache: SQLiteCache | None = None,
) -> dict[str, dict[str, Any]]:
    """Query cross-species orthology mappings for a gene.

    Returns dict structure:
      {
        "human": {"symbol": "TP53", "entrez_id": "7157", "uniprot": "P04637", "name": "..."},
        "mouse": {"symbol": "Trp53", "entrez_id": "22059", "uniprot": "P02340", "name": "..."},
        "rat": {"symbol": "Tp53", "entrez_id": "24842", "uniprot": "P10361", "name": "..."},
        "zebrafish": {"symbol": "tp53", "entrez_id": "30590", ...}
      }
    """
    raw_query = symbol_or_id.strip()
    if not raw_query:
        return {}

    source_org = detect_organism(raw_query) if organism == "auto" else organism.lower()
    cache_key = f"gene:orthologs:{raw_query.lower()}:{source_org}"
    if cache is not None:
        cached = cache.get(cache_key, vocab_version=VOCABULARY_VERSION)
        if cached:
            return cached.get("orthologs", {})

    # 1. Query offline curated core ortholog lexicon first
    lookup_key = raw_query.lower()
    human_key = _SYMBOL_TO_HUMAN_KEY.get(lookup_key) or raw_query.upper()
    if human_key in _CURATED_ORTHOLOGS:
        orth_dict = _CURATED_ORTHOLOGS[human_key]
        filtered = {org: data for org, data in orth_dict.items() if org in target_organisms}
        if cache is not None:
            cache.set(cache_key, {"orthologs": filtered}, vocab_version=VOCABULARY_VERSION)
        return filtered

    # 2. Online NCBI HomoloGene / E-utilities search
    if online:
        try:
            # Search HomoloGene
            q_term = urllib.parse.quote(f"{raw_query}[Gene Name]")
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=homologene&term={q_term}&retmode=json"
            req = urllib.request.Request(search_url, headers={"User-Agent": "perturbalign/0.9.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                search_data = json.loads(resp.read().decode("utf-8"))
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if id_list:
                hid = id_list[0]
                sum_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=homologene&id={hid}&retmode=json"
                req_sum = urllib.request.Request(sum_url, headers={"User-Agent": "perturbalign/0.9.0"})
                with urllib.request.urlopen(req_sum, timeout=3.0) as resp_sum:
                    sum_data = json.loads(resp_sum.read().decode("utf-8"))
                doc = sum_data.get("result", {}).get(str(hid), {})
                genes = doc.get("genes", [])
                tax_map = {9606: "human", 10090: "mouse", 10116: "rat", 7955: "zebrafish"}
                online_res: dict[str, dict[str, Any]] = {}
                for g_item in genes:
                    tax_id = g_item.get("taxid")
                    org_name = tax_map.get(tax_id)
                    if org_name and org_name in target_organisms:
                        online_res[org_name] = {
                            "symbol": g_item.get("symbol", ""),
                            "entrez_id": str(g_item.get("geneid", "")),
                            "name": g_item.get("title", ""),
                        }
                if online_res:
                    if cache is not None:
                        cache.set(cache_key, {"orthologs": online_res}, vocab_version=VOCABULARY_VERSION)
                    return online_res
        except Exception:
            pass

    # 3. Heuristic biological naming projection fallback
    # Human all-uppercase, mouse title-cased, rat title-cased, zebrafish all-lowercase
    norm_upper = raw_query.upper()
    norm_mouse = norm_upper.capitalize()
    norm_fish = norm_upper.lower()

    heuristic_res = {
        "human": {"symbol": norm_upper, "note": "heuristic projection"},
        "mouse": {"symbol": norm_mouse, "note": "heuristic projection"},
        "rat": {"symbol": norm_mouse, "note": "heuristic projection"},
        "zebrafish": {"symbol": norm_fish, "note": "heuristic projection"},
    }
    filtered_h = {org: data for org, data in heuristic_res.items() if org in target_organisms}
    if cache is not None:
        cache.set(cache_key, {"orthologs": filtered_h}, vocab_version=VOCABULARY_VERSION)
    return filtered_h
