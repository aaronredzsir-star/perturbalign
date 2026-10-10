# -*- coding: utf-8 -*-
"""PerturbAlign Drug Name Corpus - Minimal Merge Script.
Reads all *_names.txt from raw/, deduplicates and merges into all_drug_names.txt.
Usage: python merge_names.py
"""
import os, glob

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "all_drug_names.txt")

all_names = set()
files = sorted(glob.glob(os.path.join(RAW, "*_names.txt")))
print(f"Found {len(files)} name files")

for fp in files:
    with open(fp, encoding="utf-8", errors="replace") as f:
        count = 0
        for line in f:
            name = line.strip()
            if name and len(name) > 1:
                all_names.add(name)
                count += 1
    print(f"  {os.path.basename(fp)}: {count} lines")

merged = sorted(all_names)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(merged))

print(f"\nMerge completed: {len(merged)} unique labels -> {OUT}")
print(f"File size: {os.path.getsize(OUT) / 1024:.1f} KB")
