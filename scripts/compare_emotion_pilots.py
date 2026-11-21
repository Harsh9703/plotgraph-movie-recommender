#!/usr/bin/env python3
"""Lightweight CLI Utility to Compare Three Emotion Pilot Approaches Side-by-Side.

Compares:
1. Combined Prompt v1.0.0 (Phase 3D pilot)
2. Combined Prompt v1.1.0 (Phase 3D pilot refinement)
3. Dedicated Evidence-Grounded Emotion Prompt v1.0.0 (Phase 3E pilot)

Usage:
    python scripts/compare_emotion_pilots.py \
        --dir-v1-0 data/processed/ml-1m/kg_l/pilot \
        --dir-v1-1 data/processed/ml-1m/kg_l/pilot_v1_1 \
        --dir-emotion-v1 data/processed/ml-1m/kg_l/emotion_pilot_v1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.knowledge_graph.pilot_comparison import (
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    compare_three_way_emotion_pilots,
    format_three_way_emotion_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("compare_emotion_pilots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare emotion distributions and grounding across Phase 3D and Phase 3E pilots."
    )
    parser.add_argument(
        "--dir-v1-0",
        type=str,
        default="data/processed/ml-1m/kg_l/pilot",
        help="Directory containing Combined Prompt v1.0 artifacts (default: data/processed/ml-1m/kg_l/pilot).",
    )
    parser.add_argument(
        "--dir-v1-1",
        type=str,
        default="data/processed/ml-1m/kg_l/pilot_v1_1",
        help="Directory containing Combined Prompt v1.1 artifacts (default: data/processed/ml-1m/kg_l/pilot_v1_1).",
    )
    parser.add_argument(
        "--dir-emotion-v1",
        type=str,
        default="data/processed/ml-1m/kg_l/emotion_pilot_v1",
        help="Directory containing Dedicated Emotion Prompt v1.0 artifacts (default: data/processed/ml-1m/kg_l/emotion_pilot_v1).",
    )
    parser.add_argument(
        "--csv-v1-0",
        type=str,
        default=None,
        help="Direct path to Pilot v1.0 review CSV (overrides --dir-v1-0).",
    )
    parser.add_argument(
        "--csv-v1-1",
        type=str,
        default=None,
        help="Direct path to Pilot v1.1 review CSV (overrides --dir-v1-1).",
    )
    parser.add_argument(
        "--csv-emotion-v1",
        type=str,
        default=None,
        help="Direct path to Dedicated Emotion review CSV (overrides --dir-emotion-v1).",
    )
    parser.add_argument(
        "--name-v1-0",
        type=str,
        default="Combined v1.0",
        help="Display label for Pilot v1.0 (default: 'Combined v1.0').",
    )
    parser.add_argument(
        "--name-v1-1",
        type=str,
        default="Combined v1.1",
        help="Display label for Pilot v1.1 (default: 'Combined v1.1').",
    )
    parser.add_argument(
        "--name-emotion-v1",
        type=str,
        default="Dedicated v1.0 (Evidence)",
        help="Display label for Dedicated Emotion Pilot (default: 'Dedicated v1.0 (Evidence)').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_EMOTION_EDGE_THRESHOLD,
        help=f"Threshold for prominent emotion edge creation (default: {DEFAULT_EMOTION_EDGE_THRESHOLD}; revised from 0.5 based on pilot analysis).",
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

    source_1_0 = args.csv_v1_0 or args.dir_v1_0
    source_1_1 = args.csv_v1_1 or args.dir_v1_1
    source_emo = args.csv_emotion_v1 or args.dir_emotion_v1

    logger.info("Comparing emotion pilots:")
    logger.info("  Combined v1.0:   %s", source_1_0)
    logger.info("  Combined v1.1:   %s", source_1_1)
    logger.info("  Dedicated v1.0:  %s", source_emo)

    try:
        comparison = compare_three_way_emotion_pilots(
            source_v1_0=source_1_0,
            source_v1_1=source_1_1,
            source_emotion_v1=source_emo,
            name_v1_0=args.name_v1_0,
            name_v1_1=args.name_v1_1,
            name_emotion_v1=args.name_emotion_v1,
            emotion_threshold=args.threshold,
        )
    except Exception as err:
        logger.error(f"Failed to compare emotion pilot runs: {err}", exc_info=True)
        return 1

    report_table = format_three_way_emotion_table(comparison)
    print("\n" + report_table + "\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison JSON saved to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
