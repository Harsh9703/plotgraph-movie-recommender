#!/usr/bin/env python3
"""Lightweight CLI Utility to Compare Two Completed PlotGraph Pilot Runs.

Usage:
    python scripts/compare_pilots.py \
        --dir-a data/processed/ml-1m/kg_l/pilot \
        --dir-b data/processed/ml-1m/kg_l/pilot_v1_1 \
        --name-a "Prompt v1.0.0" \
        --name-b "Prompt v1.1.0"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.knowledge_graph.pilot_comparison import (
    compare_pilot_runs,
    format_comparison_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("compare_pilots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two completed PlotGraph pilot extraction runs (e.g. v1.0.0 vs v1.1.0)."
    )
    parser.add_argument(
        "--dir-a",
        type=str,
        default="data/processed/ml-1m/kg_l/pilot",
        help="Directory containing Pilot A artifacts (default: data/processed/ml-1m/kg_l/pilot).",
    )
    parser.add_argument(
        "--dir-b",
        type=str,
        default="data/processed/ml-1m/kg_l/pilot_v1_1",
        help="Directory containing Pilot B artifacts (default: data/processed/ml-1m/kg_l/pilot_v1_1).",
    )
    parser.add_argument(
        "--csv-a",
        type=str,
        default=None,
        help="Direct path to Pilot A review CSV (overrides --dir-a).",
    )
    parser.add_argument(
        "--csv-b",
        type=str,
        default=None,
        help="Direct path to Pilot B review CSV (overrides --dir-b).",
    )
    parser.add_argument(
        "--name-a",
        type=str,
        default="Prompt v1.0.0",
        help="Display label for Pilot A (default: 'Prompt v1.0.0').",
    )
    parser.add_argument(
        "--name-b",
        type=str,
        default="Prompt v1.1.0",
        help="Display label for Pilot B (default: 'Prompt v1.1.0').",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional destination path for structured comparison JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_a = args.csv_a or args.dir_a
    source_b = args.csv_b or args.dir_b

    try:
        comparison = compare_pilot_runs(
            source_a=source_a,
            source_b=source_b,
            name_a=args.name_a,
            name_b=args.name_b,
        )
    except Exception as err:
        logger.error(f"Failed to compare pilot runs: {err}", exc_info=True)
        return 1

    report_table = format_comparison_table(comparison)
    print("\n" + report_table + "\n")

    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison JSON to {out_json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
