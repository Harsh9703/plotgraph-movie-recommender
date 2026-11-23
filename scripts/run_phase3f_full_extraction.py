#!/usr/bin/env python3
"""CLI Entry Point for Phase 3F Full Production KG-L Extraction.

Executes production-scale, resumable, batched Qwen2.5-7B-Instruct extraction
across all eligible MovieLens-1M movies with Phase 3A plot overviews.

Usage:
  # On ASU SOL GPU Compute (via SLURM):
  python scripts/run_phase3f_full_extraction.py --config configs/kg_l.yaml --backend vllm

  # Dry-run runtime estimation (no model loading or GPU required):
  python scripts/run_phase3f_full_extraction.py --config configs/kg_l.yaml --estimate-runtime

  # Completion validation of completed run:
  python scripts/run_phase3f_full_extraction.py --config configs/kg_l.yaml --validate-only

  # Chunked execution (e.g. chunk 1 of 4):
  python scripts/run_phase3f_full_extraction.py --config configs/kg_l.yaml --chunk-id 0 --num-chunks 4

  # Local mock testing (writes to isolated mock directory):
  python scripts/run_phase3f_full_extraction.py --config configs/kg_l.yaml --backend mock --output-dir data/processed/ml-1m/kg_l/mock_extraction
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge_graph.qwen_production import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PRODUCTION_OUTPUT_DIR,
    estimate_production_runtime,
    load_eligible_production_movies,
    print_runtime_estimation,
    run_production_extraction,
    validate_production_run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_phase3f_full_extraction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3F: Production Batched Qwen KG-L Extraction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kg_l.yaml",
        help="Path to kg_l.yaml configuration file.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "mock"],
        default="vllm",
        help="Inference backend: 'vllm' for real GPU execution on SOL, 'mock' for CPU testing.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override path to movies_with_tmdb.parquet input.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Override output directory (default from config or {DEFAULT_PRODUCTION_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Inference batch size for vLLM (default from config or {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help="0-indexed chunk identifier for partitioned execution.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Total number of chunks for partitioned execution.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="0-indexed row start index for range partitioning.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="0-indexed row end index for range partitioning.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-attempt movies that previously logged permanent failures.",
    )
    parser.add_argument(
        "--estimate-runtime",
        action="store_true",
        help="Dry run: print conservative runtime estimation and exit without loading model.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run completion validation utility on existing output directory and exit.",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Disable automatic export of compact Parquet files upon completion.",
    )
    parser.add_argument(
        "--allow-mock-in-production-dir",
        action="store_true",
        help="Safety override allowing mock backend to write to production directory (testing only).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Handle Dry-Run Runtime Estimation Mode
    if args.estimate_runtime:
        metadata_path = args.input or "data/processed/ml-1m/metadata/movies_with_tmdb.parquet"
        df_eligible = load_eligible_production_movies(metadata_path)
        bs = args.batch_size or DEFAULT_BATCH_SIZE
        estimates = estimate_production_runtime(len(df_eligible), batch_size=bs)
        print_runtime_estimation(estimates)
        return 0

    # 2. Handle Completion Validation Only Mode
    if args.validate_only:
        out_dir = args.output_dir or DEFAULT_PRODUCTION_OUTPUT_DIR
        meta_p = args.input or "data/processed/ml-1m/metadata/movies_with_tmdb.parquet"
        val_result = validate_production_run(output_dir=out_dir, metadata_path=meta_p)
        return 0 if val_result["is_valid"] else 1

    # 3. Standard Production Extraction Mode
    try:
        summary = run_production_extraction(
            config_path=args.config,
            backend_type=args.backend,
            input_path=args.input,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            chunk_id=args.chunk_id,
            num_chunks=args.num_chunks,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            retry_failed=args.retry_failed,
            export_parquet=(not args.no_parquet),
            allow_mock_in_production_dir=args.allow_mock_in_production_dir,
        )
        logger.info(
            f"Phase 3F Extraction Complete: "
            f"Semantic: {summary['session_metrics']['semantic']['successes']} ok, "
            f"Emotion: {summary['session_metrics']['emotion']['successes']} ok, "
            f"Total Runtime: {summary['session_metrics']['total_runtime_seconds']}s"
        )
        return 0
    except Exception as exc:
        logger.error(f"Fatal error during Phase 3F production extraction: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
