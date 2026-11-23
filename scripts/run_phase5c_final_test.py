#!/usr/bin/env python3
"""Phase 5C: One-Time Final MovieLens Test Evaluation Execution Script.

Strict read-only evaluation:
1. Verifies SHA-256 hashes of all frozen Phase 5B artifacts against manifest.
2. Fails immediately if any artifact differs.
3. Loads TEST split for the first and only time.
4. Reports candidate pool Recall@100 ceiling.
5. Evaluates frozen EASE baseline vs frozen PlotGraph reranker.
6. Computes paired user-level bootstrap confidence intervals (N=1,000, seed=42).
7. Analyzes per-user win/loss and relevant item rank movements.
8. Persists all final results and audit record.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from src.final_eval.model_refit import compute_file_sha256
from src.final_eval.test_evaluator import (
    analyze_rank_movements,
    analyze_win_loss,
    compute_test_bootstrap_uncertainty,
    compute_test_coverage_stats,
    evaluate_test_predictions,
    verify_manifest_and_frozen_artifacts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_phase5c_final_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5C One-Time Final MovieLens Test Evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/final_test.yaml",
        help="Path to final test evaluation configuration file",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify frozen artifacts against manifest and exit without opening TEST",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    manifest_path = Path(cfg["inputs"]["manifest_path"])
    results_dir = Path(cfg["outputs"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" PlotGraph Phase 5C: One-Time Final MovieLens Test Evaluation")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Pre-Evaluation Manifest Verification
    # -------------------------------------------------------------------------
    print("\n[Step 1/8] Verifying frozen artifacts against Phase 5B manifest...")
    try:
        manifest = verify_manifest_and_frozen_artifacts(manifest_path)
    except Exception as exc:
        print(f"\nFATAL AUDIT FAILURE: Manifest verification failed: {exc}")
        print("Refusing to open or inspect TEST split.")
        sys.exit(1)

    print("\n" + "*" * 80)
    print(" FROZEN ARTIFACTS VERIFIED — BEGINNING ONE-TIME TEST EVALUATION")
    print("*" * 80)

    if args.verify_only:
        print("\nVerification mode complete. Exiting without opening TEST.")
        return

    # -------------------------------------------------------------------------
    # STEP 2: Permitted TEST Access
    # -------------------------------------------------------------------------
    test_path = Path(cfg["inputs"]["test_split_path"])
    print(f"\n[Step 2/8] Loading TEST split for one-time evaluation: {test_path}...")
    t0 = time.time()
    test_df = pd.read_parquet(test_path)
    test_sha256 = compute_file_sha256(test_path)
    print(f"  * Loaded {len(test_df):,} TEST interactions in {time.time() - t0:.2f}s")
    print(f"  * TEST split SHA-256: {test_sha256}")

    # -------------------------------------------------------------------------
    # STEP 3: Test Coverage Report & Candidate Pool Ceiling
    # -------------------------------------------------------------------------
    print("\n[Step 3/8] Analyzing TEST set coverage and candidate pool ceiling...")
    cand_path = Path(cfg["inputs"]["candidates_path"])
    candidates_df = pd.read_parquet(cand_path)
    pos_thresh = float(cfg["evaluation"].get("positive_threshold", 4.0))

    coverage_stats = compute_test_coverage_stats(test_df, candidates_df, positive_threshold=pos_thresh)
    print(f"  * Total TEST interactions:                   {coverage_stats['total_test_interactions']:,}")
    print(f"  * Total positive TEST interactions (>= {pos_thresh}): {coverage_stats['positive_test_interactions']:,}")
    print(f"  * Users with >=1 positive TEST interaction:   {coverage_stats['users_with_test_positive']:,} (Population A: End-to-End)")
    print(f"  * Positive TEST items captured in top-100:    {coverage_stats['captured_test_positives']:,}")
    print(f"  * Global candidate Recall@100:                {coverage_stats['candidate_pool_recall100_global']:.4f}")
    print(f"  * Mean user candidate Recall@100 (CEILING):   {coverage_stats['candidate_pool_recall100_mean_user']:.4f}")
    print(f"  * Users with >=1 captured positive candidate: {coverage_stats['users_with_captured_positive']:,} ({coverage_stats['candidate_coverage_pct']}%) (Population B: Conditional)")
    print(f"  * Users with 0 captured positive candidates:  {coverage_stats['users_with_zero_captured_positive']:,}")

    # -------------------------------------------------------------------------
    # STEP 4: Evaluate Frozen EASE vs Frozen PlotGraph
    # -------------------------------------------------------------------------
    print("\n[Step 4/8] Evaluating frozen EASE baseline and PlotGraph reranker...")
    unlabeled_preds_path = Path(cfg["inputs"]["unlabeled_predictions_path"])
    unlabeled_preds_df = pd.read_parquet(unlabeled_preds_path)

    cutoffs = cfg["evaluation"].get("cutoffs", [10, 20])
    e2e_results, cond_results, per_user_df = evaluate_test_predictions(
        unlabeled_preds_df=unlabeled_preds_df,
        test_df=test_df,
        cutoffs=cutoffs,
        positive_threshold=pos_thresh,
    )

    # -------------------------------------------------------------------------
    # STEP 5: Paired User-Level Bootstrap Significance
    # -------------------------------------------------------------------------
    print("\n[Step 5/8] Computing paired user-level bootstrap uncertainty (1,000 resamples)...")
    n_bootstraps = int(cfg["evaluation"].get("n_bootstraps", 1000))
    b_seed = int(cfg["evaluation"].get("bootstrap_seed", 42))

    bootstrap_results = compute_test_bootstrap_uncertainty(
        metric_arrays=e2e_results["_arrays"],
        n_bootstraps=n_bootstraps,
        seed=b_seed,
    )

    # -------------------------------------------------------------------------
    # STEP 6: Win / Loss Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 6/8] Analyzing user-level win/loss distribution on NDCG@10...")
    win_loss_stats = analyze_win_loss(per_user_df)
    print(f"  * Improved Users (PlotGraph > EASE): {win_loss_stats['improved_users_count']:,} ({win_loss_stats['improved_users_pct']}%) | Mean Δ = {win_loss_stats['mean_delta_among_improved']:+.4f}")
    print(f"  * Tied Users     (PlotGraph = EASE): {win_loss_stats['tied_users_count']:,} ({win_loss_stats['tied_users_pct']}%)")
    print(f"  * Worsened Users (PlotGraph < EASE): {win_loss_stats['worsened_users_count']:,} ({win_loss_stats['worsened_users_pct']}%) | Mean Δ = {win_loss_stats['mean_delta_among_worsened']:+.4f}")

    # -------------------------------------------------------------------------
    # STEP 7: Rank-Movement Analysis for Relevant TEST Items
    # -------------------------------------------------------------------------
    print("\n[Step 7/8] Analyzing rank movements for relevant TEST items in candidate pool...")
    movement_summary, movements_df = analyze_rank_movements(
        unlabeled_preds_df=unlabeled_preds_df,
        test_df=test_df,
        positive_threshold=pos_thresh,
    )
    print(f"  * Relevant items in top-100:       {movement_summary['total_relevant_items_in_pool']:,}")
    print(f"  * Mean EASE Rank vs PlotGraph:     {movement_summary['mean_ease_rank']} -> {movement_summary['mean_plotgraph_rank']}")
    print(f"  * Median Rank Change:              {movement_summary['median_rank_change']:+.1f}")
    print(f"  * Items Moved Upward:              {movement_summary['items_moved_upward_count']:,} ({movement_summary['items_moved_upward_pct']}%)")
    print(f"  * Items Moved Downward:            {movement_summary['items_moved_downward_count']:,} ({movement_summary['items_moved_downward_pct']}%)")
    print(f"  * Items Crossed INTO Top-10:       {movement_summary['crossed_into_top10_count']:,}")
    print(f"  * Items Crossed OUT of Top-10:     {movement_summary['crossed_out_of_top10_count']:,}")
    print(f"  * Net Top-10 Gain:                 {movement_summary['net_top10_gain']:+d}")

    # -------------------------------------------------------------------------
    # STEP 8: Serialize Output Artifacts & Final Audit Record
    # -------------------------------------------------------------------------
    print("\n[Step 8/8] Serializing final results and audit record...")

    # Strip raw array from summary before serialization
    raw_arrays = e2e_results.pop("_arrays")

    # Combine results
    final_results = {
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "ndcg@10",
        "coverage_summary": coverage_stats,
        "end_to_end_results": e2e_results,
        "conditional_results": cond_results,
        "win_loss_analysis": win_loss_stats,
        "rank_movement_summary": movement_summary,
        "bootstrap_results": bootstrap_results,
    }

    # 1. JSON results
    res_json_path = Path(cfg["outputs"]["final_results_json"])
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"  * Saved final test results JSON: {res_json_path}")

    # 2. CSV results
    res_csv_path = Path(cfg["outputs"]["final_results_csv"])
    csv_rows = []
    for k in cutoffs:
        for m in ["ndcg", "recall", "hr", "mrr"]:
            key = f"{m}@{k}"
            csv_rows.append({
                "metric": key,
                "cutoff": k,
                "ease_e2e": e2e_results[f"ease_{key}"],
                "plotgraph_e2e": e2e_results[f"plotgraph_{key}"],
                "delta_e2e_abs": e2e_results[f"delta_{key}_abs"],
                "delta_e2e_rel_pct": e2e_results[f"delta_{key}_rel_pct"],
                "ease_conditional": cond_results[f"ease_{key}"],
                "plotgraph_conditional": cond_results[f"plotgraph_{key}"],
                "delta_cond_abs": cond_results[f"delta_{key}_abs"],
                "delta_cond_rel_pct": cond_results[f"delta_{key}_rel_pct"],
            })
    pd.DataFrame(csv_rows).to_csv(res_csv_path, index=False)
    print(f"  * Saved final test results CSV:  {res_csv_path}")

    # 3. Per-user metrics Parquet
    per_user_path = Path(cfg["outputs"]["per_user_metrics_parquet"])
    per_user_df.to_parquet(per_user_path, index=False)
    print(f"  * Saved per-user test metrics:   {per_user_path} ({len(per_user_df):,} rows)")

    # 4. Rank movements Parquet
    movements_path = Path(cfg["outputs"]["rank_movements_parquet"])
    movements_df.to_parquet(movements_path, index=False)
    print(f"  * Saved rank movements Parquet:  {movements_path} ({len(movements_df):,} rows)")

    # 5. Bootstrap results JSON
    bootstrap_json_path = Path(cfg["outputs"]["bootstrap_results_json"])
    with open(bootstrap_json_path, "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"  * Saved bootstrap results JSON:  {bootstrap_json_path}")

    # 6. Final Audit Record JSON
    audit_record_path = Path(cfg["outputs"]["final_audit_record_json"])
    audit_record = {
        "audit_version": "1.0.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 5C - Final MovieLens Test Evaluation",
        "certification_statement": (
            "TEST labels were accessed only after methodology and artifacts were frozen. "
            "No model selection, feature selection, candidate tuning, or hyperparameter tuning "
            "was performed using TEST."
        ),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "test_split_sha256": test_sha256,
        "frozen_artifact_checksums": manifest.get("artifact_checksums"),
        "test_interaction_counts": coverage_stats,
        "primary_metric_ndcg10": {
            "ease": e2e_results["ease_ndcg@10"],
            "plotgraph": e2e_results["plotgraph_ndcg@10"],
            "delta_abs": e2e_results["delta_ndcg@10_abs"],
            "delta_rel_pct": e2e_results["delta_ndcg@10_rel_pct"],
            "bootstrap_95_ci": [
                bootstrap_results["delta_ndcg@10"]["ci_lower_95"],
                bootstrap_results["delta_ndcg@10"]["ci_upper_95"],
            ],
            "paired_p_value": bootstrap_results["delta_ndcg@10"]["paired_p_value"],
            "is_significant_95": bootstrap_results["delta_ndcg@10"]["is_significant_95"],
        },
        "all_final_metrics": e2e_results,
    }
    with open(audit_record_path, "w", encoding="utf-8") as f:
        json.dump(audit_record, f, indent=2)
    print(f"  * Saved final audit record JSON: {audit_record_path}")

    # -------------------------------------------------------------------------
    # PRINT CONCISE FINAL COMPARISON TABLE
    # -------------------------------------------------------------------------
    b_ndcg = bootstrap_results["delta_ndcg@10"]
    sig_str = "SIGNIFICANT (p < 0.05)" if b_ndcg["is_significant_95"] else "NOT significant (p >= 0.05)"

    print("\n" + "=" * 90)
    print(" FINAL ONE-TIME MOVIELENS-1M TEST EVALUATION RESULTS (N = 5,640 Users)")
    print("=" * 90)
    print(f"{'Metric':<14} | {'EASE Baseline':<14} | {'PlotGraph (FULL+NRC)':<20} | {'Absolute Δ':<12} | {'Relative Δ (%)':<15}")
    print("-" * 90)
    metrics_display = [
        ("NDCG@10 [PRIMARY]", "ndcg@10"),
        ("Recall@10 (e2e)", "recall@10"),
        ("HR@10", "hr@10"),
        ("MRR@10", "mrr@10"),
        ("NDCG@20", "ndcg@20"),
        ("Recall@20 (e2e)", "recall@20"),
        ("HR@20", "hr@20"),
        ("MRR@20", "mrr@20"),
    ]
    for label, m_key in metrics_display:
        e_val = e2e_results[f"ease_{m_key}"]
        p_val = e2e_results[f"plotgraph_{m_key}"]
        d_abs = e2e_results[f"delta_{m_key}_abs"]
        d_rel = e2e_results[f"delta_{m_key}_rel_pct"]
        print(f"{label:<18} | {e_val:14.4f} | {p_val:20.4f} | {d_abs:+12.4f} | {d_rel:+14.2f}%")
    print("-" * 90)
    print(f"Candidate Recall@100 Ceiling (Mean Per-User): {coverage_stats['candidate_pool_recall100_mean_user']:.4f}")
    print(f"Paired Bootstrap Δ NDCG@10 (N=1,000, Seed=42): {b_ndcg['point_delta']:+.4f} [{b_ndcg['ci_lower_95']:+.4f}, {b_ndcg['ci_upper_95']:+.4f}]")
    print(f"Statistical Significance: {sig_str} (p = {b_ndcg['paired_p_value']:.4f}, Superiority Prob = {b_ndcg['prob_a_superior']*100:.1f}%)")
    print("=" * 90)
    print("\nPhase 5C Final Test Evaluation completed successfully.")


if __name__ == "__main__":
    main()
