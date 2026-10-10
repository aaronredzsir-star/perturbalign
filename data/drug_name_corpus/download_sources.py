# -*- coding: utf-8 -*-
"""PerturbAlign Drug Name Corpus - Minimal Downloader Script.
Downloads raw perturbation labels directly from 17 public data sources into raw/ directory.
Usage: python download_sources.py
"""
import os, time, gzip, requests

RAW = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
NP = {"http": None, "https": None}  # Direct connection, no proxy

def get(url, timeout=60, **kw):
    return requests.get(url, headers=H, proxies=NP, timeout=timeout, **kw)

def save(names, fname):
    uniq = sorted({n.strip() for n in names if n and len(n.strip()) > 1})
    with open(os.path.join(RAW, fname), "w", encoding="utf-8") as f:
        f.write("\n".join(uniq))
    print(f"  {fname}: {len(uniq)}")

# 1. LINCS L1000 siginfo
print("[1] LINCS L1000...")
r = get("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz")
if r.status_code == 200:
    with open(os.path.join(RAW, "lincs_sig_info.txt.gz"), "wb") as f:
        f.write(r.content)
    text = gzip.decompress(r.content).decode("utf-8", errors="replace")
    names = []
    for line in text.split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            names.append(parts[2])  # pert_iname
    save(names, "lincs_pert_iname.txt")

# 2. Broad Repurposing Hub
print("[2] Repurposing Hub...")
for url, fname in [
    ("https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt", "repurposing_drugs.txt"),
    ("https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_samples_20200324.txt", "repurposing_samples.txt"),
]:
    r = get(url)
    if r.status_code == 200:
        with open(os.path.join(RAW, fname), "wb") as f:
            f.write(r.content)
        names = []
        for line in r.text.split("\n"):
            if line.startswith("!") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                names.append(parts[1])  # pert_iname / broad_name
        save(names, fname.replace(".txt", "_names.txt"))

# 3. ChEMBL（      ，     ）
print("[3] ChEMBL...")
chembl = []
for letter in "abcdefghijklmnopqrstuvwxyz":
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule?limit=1000&format=json&molecule_type=Small molecule&pref_name__istartswith={letter}"
    try:
        r = get(url, timeout=30)
        if r.status_code == 200:
            for m in r.json().get("molecules", []):
                if m.get("pref_name"):
                    chembl.append(m["pref_name"])
                for s in m.get("molecule_synonyms", []):
                    if s.get("molecule_synonym"):
                        chembl.append(s["molecule_synonym"])
        time.sleep(0.15)
    except:
        pass
save(chembl, "chembl_molecule_names.txt")

# 4. KEGG（DRUG + COMPOUND + GLYCAN + ENZYME + REACTION）
print("[4] KEGG...")
for db, fname in [("drug", "kegg_drug"), ("compound", "kegg_compound"),
                  ("glycan", "kegg_glycan"), ("enzyme", "kegg_enzyme"), ("reaction", "kegg_reaction")]:
    r = get(f"https://rest.kegg.jp/list/{db}", timeout=60)
    if r.status_code == 200:
        names = []
        for line in r.text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                for n in parts[1].split(";"):
                    names.append(n.strip())
        save(names, f"{fname}_names.txt")

# 5. openFDA Drug Label
print("[5] openFDA...")
fda = []
for skip in range(0, 7000, 100):
    try:
        r = get(f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:*&limit=100&skip={skip}", timeout=20)
        if r.status_code != 200:
            break
        for res in r.json().get("results", []):
            for n in res.get("openfda", {}).get("generic_name", []):
                fda.append(n)
        time.sleep(0.1)
    except:
        break
save(fda, "openfda_drug_names.txt")

# 6. Wikidata（SPARQL）
print("[6] Wikidata...")
sparql = """SELECT ?item ?label WHERE {
  ?item wdt:P31 wd:Q12140; rdfs:label ?label.
  FILTER(LANG(?label) = "en")
} LIMIT 5000"""
r = get("https://query.wikidata.org/sparql", params={"query": sparql, "format": "json"}, timeout=60)
if r.status_code == 200:
    names = [b["label"]["value"] for b in r.json().get("results", {}).get("bindings", [])]
    save(names, "wikidata_drug_names.txt")

# 7. SIDER
print("[7] SIDER...")
r = get("http://sideeffects.embl.de/media/download/drug_names.tsv", timeout=30)
if r.status_code == 200:
    names = [line.split("\t")[0] for line in r.text.strip().split("\n")[1:] if "\t" in line]
    save(names, "sider_drug_names.txt")

# 8. RxNorm
print("[8] RxNorm...")
r = get("https://rxnav.nlm.nih.gov/REST/allstatus?type=IN", timeout=30)
if r.status_code == 200:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(r.text)
    names = [c.text for c in root.iter("name") if c.text]
    save(names, "rxnorm_drug_names.txt")

# 9. DailyMed
print("[9] DailyMed...")
r = get("https://dailymed.nlm.nih.gov/dailymed/services/v2/drugnames.cfm?pagesize=100", timeout=30)
if r.status_code == 200:
    names = [d.get("name") for d in r.json().get("data", []) if d.get("name")]
    save(names, "dailymed_drug_names.txt")

print("\nDownload completed. Raw files saved in raw/ directory.")
