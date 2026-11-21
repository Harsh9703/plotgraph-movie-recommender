#!/usr/bin/env python3
"""CLI Runner for Phase 3D: Qwen2.5-7B-Instruct 30-Movie Pilot.

Usage:
    # Run locally with mock backend (macOS / no GPU required):
    python scripts/run_phase3d_qwen_pilot.py --backend mock

    # Run on ASU SOL GPU node with real vLLM inference:
    python scripts/run_phase3d_qwen_pilot.py --backend vllm
"""

import argparse
import logging
import sys
from pathlib import Path

from src.knowledge_graph.qwen_pilot import MAX_PILOT_MOVIES, run_qwen_pilot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_phase3d_qwen_pilot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 3D Qwen2.5-7B-Instruct 30-movie pilot extraction."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kg_l.yaml",
        help="Path to KG-L configuration YAML file (default: configs/kg_l.yaml).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mock", "vllm"],
        default="mock",
        help="Inference backend: 'mock' for local CPU/testing, 'vllm' for ASU SOL GPU (default: mock).",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to input sample parquet file (default: from config, max 30 movies).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for pilot artifacts (default: resolved from prompt version, e.g. pilot_v1_1 for v1.1.0).",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        choices=["v1.0.0", "v1.1.0"],
        help="Prompt template version ('v1.0.0' or 'v1.1.0', default: from config).",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=MAX_PILOT_MOVIES,
        help=f"Maximum movies to process in pilot mode (strictly <= {MAX_PILOT_MOVIES}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        summary = run_qwen_pilot(
            config_path=args.config,
            backend_type=args.backend,
            input_path=args.input_path,
            output_dir=args.output_dir,
            prompt_version=args.prompt_version,
            max_movies=args.max_movies,
        )
    except Exception as err:
        logger.error(f"Pilot execution failed with error: {err}", exc_info=True)
        return 1

    # Pretty-print final summary table
    out_dir = args.output_dir or (
        "data/processed/ml-1m/kg_l/pilot_v1_1"
        if summary.get("prompt_version") == "v1.1.0"
        else "data/processed/ml-1m/kg_l/pilot"
    )
    print("\n" + "=" * 65)
    print("           PHASE 3D QWEN PILOT RUN SUMMARY")
    print("=" * 65)
    print(f"Backend                     : {summary.get('backend')}")
    print(f"Model ID                    : {summary.get('model_id')}")
    print(f"Prompt Version              : {summary.get('prompt_version')}")
    print(f"Schema Version              : {summary.get('schema_version')}")
    print(f"Total Movies in Sample      : {summary.get('total_movies_in_sample')}")
    print(f"Successfully Validated      : {summary.get('successfully_validated_count')}")
    print(f"Failed Extractions          : {summary.get('failed_count')}")
    print(f"Total Retries Used          : {summary.get('total_retries_used')}")
    print(f"Success Rate                : {summary.get('success_rate'):.1f}%")
    print(f"Total Runtime (seconds)     : {summary.get('total_pilot_runtime_seconds'):.2f}s")
    print(f"Avg Runtime / Movie (s)     : {summary.get('average_runtime_per_movie_seconds'):.3f}s")
    print("-" * 65)
    print("Artifacts Written:")
    print(f"  - Raw Outputs             : {out_dir}/raw_outputs.jsonl")
    print(f"  - Validated Extractions   : {out_dir}/validated_extractions.jsonl")
    print(f"  - Failures Log            : {out_dir}/failures.jsonl")
    print(f"  - Pilot Summary           : {out_dir}/pilot_summary.json")
    print(f"  - Human Review CSV        : {out_dir}/pilot_review.csv")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
