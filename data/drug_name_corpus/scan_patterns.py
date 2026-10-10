# -*- coding: utf-8 -*-
"""PerturbAlign Drug Name Corpus - Minimal Pattern Scanning Script.
Classifies all_drug_names.txt by naming protocol: salts/doses/controls/combos/stereochemistry/vendor codes.
Usage: python scan_patterns.py
"""
import os, re
from collections import Counter

HERE = os.path.dirname(__file__)
NAMES = os.path.join(HERE, "all_drug_names.txt")

with open(NAMES, encoding="utf-8") as f:
    labels = [l.strip() for l in f if l.strip()]
print(f"Scanning {len(labels)} labels\n")

# 1. Salt suffix
print("=" * 60)
print("1. Salt / Ester Suffixes (trailing word match, >= 20)")
print("=" * 60)
salt_tails = Counter()
for lab in labels:
    toks = re.split(r"[\s_]+", lab.strip())
    if len(toks) >= 2:
        last = toks[-1].lower().strip(".,;:")
        if re.search(r"(ate|ide|ine|ium|hydrate|base|salt|oxide|sulfide)$", last):
            salt_tails[last] += 1
for t, c in salt_tails.most_common(25):
    if c >= 20:
        print(f"  {c:6d}  {t}")

# 2. Dose unit
print("\n" + "=" * 60)
print("2. Dose Units (trailing number + unit, >= 5)")
print("=" * 60)
units = Counter()
for lab in labels:
    m = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)\s*$", lab)
    if m:
        u = m.group(2).lower()
        if len(u) >= 2:
            units[u] += 1
for u, c in units.most_common(20):
    if c >= 5:
        print(f"  {c:6d}  {u}")

# 3. Control tokens
print("\n" + "=" * 60)
print("3. Control / Solvent Tokens (whole word match)")
print("=" * 60)
ctrl = Counter()
ctrl_set = {"dmso", "vehicle", "control", "untreated", "mock", "gfp",
            "egfp", "luciferase", "rfp", "lacZ", "empty", "sgnt", "plko",
            "h2o", "water", "pbs", "dmem", "fbs", "bfp"}
for lab in labels:
    low = lab.lower().strip()
    if low in ctrl_set:
        ctrl[low] += 1
for t, c in ctrl.most_common():
    print(f"  {c:6d}  {t}")

# 4. Combination separators
print("\n" + "=" * 60)
print("4. Combination Delimiters (label count containing delimiter)")
print("=" * 60)
for sep, name in [("+", "plus"), (" and ", "and"), (" & ", "amp"),
                  ("/", "slash"), (" + ", "plus_space"), ("_", "underscore")]:
    count = sum(1 for l in labels if sep in l)
    print(f"  {count:6d}  '{sep}' ({name})")

# 5. Stereochemistry markers
print("\n" + "=" * 60)
print("5. Stereochemistry Markers")
print("=" * 60)
for pat, name in [(r"\([RS]\)", "(R)/(S)"), (r"\([+-]\)", "(+)/(-)"),
                  (r"\d+[a-zA-Z]", "digit + letter"), (r"cis|trans", "cis/trans"),
                  (r"\([SE]\)-\([+-]\)-", "(S)-(+)- prefix")]:
    count = sum(1 for l in labels if re.search(pat, l, re.I))
    print(f"  {count:6d}  {name}")

# 6. Vendor / Pharma code prefixes
print("\n" + "=" * 60)
print("6. Vendor / Pharma Code Prefixes (>= 50)")
print("=" * 60)
prefixes = Counter()
for lab in labels:
    m = re.match(r"^([A-Z]{2,5}-)", lab)
    if m:
        prefixes[m.group(1)] += 1
for p, c in prefixes.most_common(20):
    if c >= 50:
        print(f"  {c:6d}  {p}")

# 7. Chemical reaction formulas
print("\n" + "=" * 60)
print("7. Chemical Reaction Formulas (containing <=>)")
print("=" * 60)
rxn = [l for l in labels if "<=>" in l]
print(f"  {len(rxn)} reaction labels")
for l in rxn[:3]:
    print(f"    {l[:70]}")

# 8. Label length distribution
print("\n" + "=" * 60)
print("8. Label Length Distribution")
print("=" * 60)
lens = [len(l) for l in labels]
print(f"  Min: {min(lens)}, Max: {max(lens)}, Mean: {sum(lens)/len(lens):.1f}")
for threshold in [10, 20, 30, 50, 100]:
    count = sum(1 for l in lens if l <= threshold)
    print(f"  <= {threshold:3d} chars: {count:6d} ({count/len(lens)*100:.1f}%)")

print("\nScan completed.")
