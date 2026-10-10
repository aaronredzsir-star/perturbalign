"""Command-Line Interface (CLI) for perturbalign.

Provides turnkey CLI tools:
- perturbalign align: Batch clean and align tabular data (CSV/TSV), supporting independent dose/time columns and combo explosion.
- perturbalign audit: Rapidly profile and audit perturbation quality and modality distribution.
- perturbalign cache: Manage local SQLite cache (stats, export for internal distribution, import cold-start corpus, clear).
- perturbalign mcp: Launch Model Context Protocol (MCP) server for AI Agent integration (Claude/Cursor/Antigravity).
- perturbalign version: Show installed version and vocabulary SHA.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from . import __version__
from .resources import VOCABULARY_VERSION


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"perturbalign v{__version__} (vocabulary SHA: {VOCABULARY_VERSION})")
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    from .cache import get_default_cache

    cache = get_default_cache()
    action = args.cache_action

    if action == "stats":
        cnt = cache.count()
        file_size_kb = os.path.getsize(cache.path) / 1024 if os.path.exists(cache.path) else 0.0
        print(f"Cache location : {cache.path}")
        print(f"Cached entries : {cnt}")
        print(f"Cache file size: {file_size_kb:.1f} KB")
        return 0

    if action == "export":
        if not args.path:
            print("Error: export requires destination file path, e.g.: perturbalign cache export corpus.sqlite")
            return 1
        n = cache.export_corpus(args.path)
        print(f"Successfully exported {n} unexpired valid entries to: {args.path}")
        return 0

    if action == "import":
        if not args.path:
            print("Error: import requires source file path, e.g.: perturbalign cache import corpus.sqlite")
            return 1
        n = cache.import_corpus(args.path)
        print(f"Successfully imported {n} entries from {args.path} into local cache.")
        return 0

    if action == "clear":
        n = cache.clear()
        print(f"Cleared local cache, removed {n} entries.")
        return 0

    print(f"Unknown cache command: {action}")
    return 1


def _cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import PerturbAlignMCPServer

    server = PerturbAlignMCPServer()
    server.run_stdio()
    return 0


def _cmd_align(args: argparse.Namespace) -> int:
    import pandas as pd
    from .adapter import annotate_dataframe, explode_combos

    in_path = args.input
    if not os.path.exists(in_path):
        print(f"Error: Input file does not exist: {in_path}")
        return 1

    # Detect delimiter
    sep = "\t" if in_path.endswith(".tsv") else ","
    df = pd.read_csv(in_path, sep=sep)

    pert_col = args.pert_col
    if pert_col not in df.columns:
        print(f"Error: Column {pert_col!r} not in file columns; available: {list(df.columns)}")
        return 1

    online = not args.offline
    fuzzy = args.fuzzy

    out_df, audit = annotate_dataframe(
        df,
        pert_col=pert_col,
        dose_col=args.dose_col,
        unit_col=args.unit_col,
        time_col=args.time_col,
        time_unit_col=args.time_unit_col,
        prefix=args.prefix,
        standardize_doses=args.standardize_doses,
        target_dose_unit=args.target_dose_unit,
        mw_col=args.mw_col,
        online=online,
        fuzzy=fuzzy,
        enrich_chembl=args.enrich_chembl,
        enrich_rdkit=args.enrich_rdkit,
        return_audit=True,
    )

    if args.explode_combos:
        out_df = explode_combos(out_df, prefix=args.prefix)

    out_path = args.output
    if not out_path:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_aligned{ext}"

    out_sep = "\t" if out_path.endswith(".tsv") else ","
    out_df.to_csv(out_path, sep=out_sep, index=False)
    print(f"Alignment completed! Wrote {len(out_df)} rows to: {out_path}")
    if audit:
        stats = audit.get("stats", {})
        print(f"Audit Summary: Matched={audit.get('n_total', 0) - stats.get('unmatched', 0)}, "
              f"Control={stats.get('control', 0)}, Combo={stats.get('combo', 0)}, "
              f"Unmatched={stats.get('unmatched', 0)}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    import pandas as pd
    from .align import align

    in_path = args.input
    if not os.path.exists(in_path):
        print(f"Error: Input file does not exist: {in_path}")
        return 1

    sep = "\t" if in_path.endswith(".tsv") else ","
    df = pd.read_csv(in_path, sep=sep)
    pert_col = args.pert_col
    if pert_col not in df.columns:
        print(f"Error: Column {pert_col!r} not in file columns; available: {list(df.columns)}")
        return 1

    labels = df[pert_col].tolist()
    results = align(labels, online=False, fuzzy=args.fuzzy)

    modality_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    unmatched: list[str] = []

    for r in results:
        modality_counts[r.modality] = modality_counts.get(r.modality, 0) + 1
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
        if r.status == "unmatched":
            unmatched.append(r.input)

    print("=" * 60)
    print(f"perturbalign Audit Report: {in_path}")
    print(f"Total samples: {len(results)} (Unique perturbation labels: {len(set(labels))})")
    print("-" * 60)
    print("Modality Distribution:")
    for mod, count in sorted(modality_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(results)) * 100
        print(f"  - {mod:<20}: {count:>6} ({pct:>5.1f}%)")
    print("-" * 60)
    print("Status Breakdown:")
    for st, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(results)) * 100
        print(f"  - {st:<20}: {count:>6} ({pct:>5.1f}%)")
    if unmatched:
        print("-" * 60)
        from collections import Counter
        top_unmatched = Counter(unmatched).most_common(10)
        print("Top Unmatched:")
        for tag, cnt in top_unmatched:
            print(f"  - {tag:<30}: {cnt} times")
    print("=" * 60)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="perturbalign",
        description="perturbalign: Single-cell and high-throughput perturbation metadata alignment and annotation toolkit",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")

    # 1. version
    p_ver = subparsers.add_parser("version", help="Show version and vocabulary SHA")
    p_ver.set_defaults(func=_cmd_version)

    # 2. align
    p_align = subparsers.add_parser("align", help="Align and standardize tabular perturbation data (CSV/TSV)")
    p_align.add_argument("input", help="Path to input CSV or TSV file")
    p_align.add_argument("-o", "--output", help="Output file path (defaults to <name>_aligned.<ext>)")
    p_align.add_argument("--pert-col", default="perturbation", help="Column name containing perturbation labels (default: perturbation)")
    p_align.add_argument("--dose-col", help="Optional: column name for independent numeric dose")
    p_align.add_argument("--unit-col", help="Optional: column name for independent dose unit")
    p_align.add_argument("--time-col", help="Optional: column name for independent duration")
    p_align.add_argument("--time-unit-col", help="Optional: column name for independent duration unit")
    p_align.add_argument("--mw-col", help="Optional: column name for molecular weight (for molar/mass dimension conversion)")
    p_align.add_argument("--prefix", default="perturbalign_", help="Prefix for newly added columns (default: perturbalign_)")
    p_align.add_argument("--fuzzy", action="store_true", help="Enable Level-2 fuzzy spelling error-tolerant matching")
    p_align.add_argument("--offline", action="store_true", help="Offline-only mode (queries local cache and lexicons only)")
    p_align.add_argument("--enrich-chembl", action="store_true", help="Enable ChEMBL target and mechanism of action enrichment")
    p_align.add_argument("--enrich-rdkit", action="store_true", help="Enable RDKit parent InChIKey computation")
    p_align.add_argument("--standardize-doses", action="store_true", help="Automatically normalize doses and units by physical dimension")
    p_align.add_argument("--target-dose-unit", help="Target dose unit (e.g. µM, µg/ml, mg/kg)")
    p_align.add_argument("--explode-combos", action="store_true", help="Explode combination rows into multiple individual component rows")
    p_align.set_defaults(func=_cmd_align)

    # 3. audit
    p_audit = subparsers.add_parser("audit", help="Audit perturbation quality and modality distribution")
    p_audit.add_argument("input", help="Path to input CSV or TSV file")
    p_audit.add_argument("--pert-col", default="perturbation", help="Column name containing perturbation labels (default: perturbation)")
    p_audit.add_argument("--fuzzy", action="store_true", help="Enable fuzzy matching")
    p_audit.set_defaults(func=_cmd_audit)

    # 4. cache
    p_cache = subparsers.add_parser("cache", help="Manage local SQLite cache")
    p_cache.add_argument("cache_action", choices=["stats", "clear", "export", "import"], help="Cache operation")
    p_cache.add_argument("path", nargs="?", help="Export destination or import source file path")
    p_cache.set_defaults(func=_cmd_cache)

    # 5. mcp
    p_mcp = subparsers.add_parser("mcp", help="Launch MCP (Model Context Protocol) stdio server for AI Agent integration")
    p_mcp.set_defaults(func=_cmd_mcp)

    args = parser.parse_args(argv)
    if not args.subcommand:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
