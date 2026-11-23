#!/usr/bin/env python3
"""Phase 5A Learning-to-Rank LightGBM Reranker Training and Validation Ablations.

Executes:
1. Zero TEST access verification.
2. Candidate pool limitation and Recall@100 reporting.
3. Deterministic user-level partitioning (train/eval and fit/inner-val).
4. EASE baseline evaluation on held-out evaluation users.
5. Training of 11 LightGBM LambdaRank ablation models.
6. Evaluation across all ranking metrics at K=10 and K=20 (NDCG, Recall e2e & cond, HR, MRR).
7. User-level bootstrap significance analysis (1,000 resamples for 95% CI).
8. Best model selection, feature importance analysis, and artifact serialization.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import yaml

from src.reranker.dataset import (
    assert_no_test_access,
    build_lgbm_dataset,
    build_val_labels,
    create_user_splits,
    get_candidate_pool_stats,
)
from src.reranker.evaluator import (
    compute_user_bootstrap,
    evaluate_candidate_rankings,
)
from src.reranker.trainer import (
    ABLATION_CONFIGS,
    compute_feature_importance,
    train_lgbm_ranker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5A LightGBM Reranker and Validation Ablation Experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reranker.yaml",
        help="Path to reranker configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Enforce strict TEST split isolation
    candidates_path = cfg["inputs"]["candidates_path"]
    val_path = cfg["inputs"]["val_path"]
    assert_no_test_access(candidates_path)
    assert_no_test_access(val_path)
    assert_no_test_access(str(cfg))

    output_dir = Path(cfg["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" PlotGraph Phase 5A: LightGBM Reranker Training & Ablation Experiments")
    print("=" * 80)
    print(f"Candidate Features: {candidates_path}")
    print(f"Validation Split:   {val_path}")
    print(f"Output Directory:   {output_dir}")
    print("Strict Zero-Test Guard: ACTIVE (TEST split is never loaded or referenced)")
    print("-" * 80)

    # 2. Load candidate features and validation split
    print("\n[Step 1/8] Loading candidate features and validation labels...")
    t0 = time.time()
    cand_df = pd.read_parquet(candidates_path)
    val_df = pd.read_parquet(val_path)
    print(f"Loaded {len(cand_df):,} candidate rows ({cand_df['user_idx'].nunique():,} users) in {time.time() - t0:.2f}s")
    print(f"Loaded {len(val_df):,} validation interactions in {time.time() - t0:.2f}s")

    # 3. Candidate pool limitation and statistics
    print("\n[Step 2/8] Analyzing candidate pool limitations and coverage...")
    pos_thresh = cfg["splits"].get("positive_threshold", 4.0)
    cand_df = build_val_labels(cand_df, val_df, positive_threshold=pos_thresh)
    pool_stats = get_candidate_pool_stats(cand_df, val_df, positive_threshold=pos_thresh)

    print(f"  * Total VAL interactions:                   {pool_stats['total_val_interactions']:,}")
    print(f"  * Total positive VAL interactions (>= {pos_thresh}): {pool_stats['total_val_positives']:,}")
    print(f"  * Users with >=1 positive VAL interaction:  {pool_stats['users_with_val_positive']:,}")
    print(f"  * Positive VAL items in top-100 pool:       {pool_stats['captured_val_positives']:,}")
    print(f"  * Global candidate-pool Recall@100:         {pool_stats['candidate_pool_recall100_global']:.4f}")
    print(f"  * Mean user candidate-pool Recall@100:      {pool_stats['candidate_pool_recall100_mean_user']:.4f}")
    print(f"  * Users with >=1 positive candidate:        {pool_stats['users_with_captured_positive']:,} ({pool_stats['evaluable_user_ratio'] * 100:.2f}%)")
    print(f"  * Users with 0 positive candidates:         {pool_stats['users_with_zero_captured_positive']:,}")

    # Build validation ground truth lookup dict
    if "positive" in val_df.columns:
        val_pos_df = val_df[val_df["positive"] == 1]
    else:
        val_pos_df = val_df[val_df["rating"] >= pos_thresh]
    val_pos_dict: Dict[int, set] = val_pos_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    # 4. User-level partition
    print("\n[Step 3/8] Creating deterministic user-level partitions...")
    eval_size = cfg["splits"].get("eval_size", 0.20)
    inner_val_size = cfg["splits"].get("inner_val_size", 0.20)
    seed = cfg["splits"].get("seed", 42)

    user_splits = create_user_splits(
        cand_df,
        eval_size=eval_size,
        inner_val_size=inner_val_size,
        seed=seed,
    )

    print(f"  * Evaluable users (>=1 positive in pool):  {len(user_splits.all_evaluable_users):,}")
    print(f"  * Held-out evaluation users (20%):          {len(user_splits.eval_users):,}")
    print(f"  * Reranker training users (80%):            {len(user_splits.train_users):,}")
    print(f"    - Fit partition (80% of train):           {len(user_splits.fit_users):,}")
    print(f"    - Inner-validation partition (early stop): {len(user_splits.inner_val_users):,}")

    # Prepare held-out evaluation dataframe
    eval_df = cand_df[cand_df["user_idx"].isin(set(user_splits.eval_users))].copy()
    eval_df = eval_df.sort_values(["user_idx", "candidate_rank"])

    # 5. EASE Baseline Evaluation
    print("\n[Step 4/8] Evaluating EASE baseline on held-out evaluation users...")
    # Original EASE ranking: candidates are already ordered by candidate_rank (1 to 100).
    # We assign ease_score or negative candidate_rank as the score.
    eval_df["score_ease"] = eval_df["ease_score"]

    cutoffs = cfg["evaluation"].get("cutoffs", [10, 20])
    ease_metrics, ease_user_metrics = evaluate_candidate_rankings(
        eval_df,
        score_col="score_ease",
        val_pos_dict=val_pos_dict,
        cutoffs=cutoffs,
    )

    print(f"  EASE Baseline -> NDCG@10: {ease_metrics['ndcg@10']:.4f} | Recall@10 (e2e): {ease_metrics['recall@10']:.4f} | Recall@10 (cond): {ease_metrics['recall_cond@10']:.4f} | HR@10: {ease_metrics['hr@10']:.4f} | MRR@10: {ease_metrics['mrr@10']:.4f}")

    # 6. Train and Evaluate Feature Ablation Models
    print("\n[Step 5/8] Training LightGBM LambdaRank models across 11 feature ablations...")
    lgbm_params = cfg.get("lightgbm", {})

    ablation_results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}
    model_user_ndcg10: Dict[str, np.ndarray] = {"EASE": ease_user_metrics["ndcg@10"]}

    # Store baseline EASE entry
    ease_entry = {
        "model_name": "EASE_BASELINE",
        "feature_family": "CANDIDATE_GENERATOR",
        "num_features": 0,
        "fit_duration_seconds": 0.0,
        "inference_duration_seconds": 0.0,
        "best_iteration": 0,
        **ease_metrics,
        "delta_ndcg@10_abs": 0.0,
        "delta_ndcg@10_rel_pct": 0.0,
    }
    ablation_results.append(ease_entry)

    # Iterate over ablations
    for name, feature_list in ABLATION_CONFIGS.items():
        print(f"\n--- Training Ablation: {name} ({len(feature_list)} features) ---")

        # Build fit, inner-val, and eval matrices
        X_fit, y_fit, g_fit, _ = build_lgbm_dataset(cand_df, user_splits.fit_users, feature_list)
        X_inner, y_inner, g_inner, _ = build_lgbm_dataset(cand_df, user_splits.inner_val_users, feature_list)
        X_eval, _, _, _ = build_lgbm_dataset(cand_df, user_splits.eval_users, feature_list)

        # Train model
        model, train_info = train_lgbm_ranker(
            X_fit=X_fit,
            y_fit=y_fit,
            group_fit=g_fit,
            X_inner=X_inner,
            y_inner=y_inner,
            group_inner=g_inner,
            feature_names=feature_list,
            params=lgbm_params,
        )
        trained_models[name] = model

        # Predict on held-out eval users
        t_inf = time.time()
        preds = model.predict(X_eval)
        inf_duration = time.time() - t_inf

        pred_col = f"score_{name.lower()}"
        eval_df[pred_col] = preds

        # Evaluate ranking
        metrics, user_metrics = evaluate_candidate_rankings(
            eval_df,
            score_col=pred_col,
            val_pos_dict=val_pos_dict,
            cutoffs=cutoffs,
        )
        model_user_ndcg10[name] = user_metrics["ndcg@10"]

        # Delta vs EASE
        delta_abs = metrics["ndcg@10"] - ease_metrics["ndcg@10"]
        delta_rel_pct = (delta_abs / ease_metrics["ndcg@10"] * 100.0) if ease_metrics["ndcg@10"] > 0 else 0.0

        print(
            f"  Result -> NDCG@10: {metrics['ndcg@10']:.4f} (Δ={delta_abs:+.4f}, {delta_rel_pct:+.2f}%) | "
            f"Recall@10 (e2e): {metrics['recall@10']:.4f} | Recall@10 (cond): {metrics['recall_cond@10']:.4f} | "
            f"HR@10: {metrics['hr@10']:.4f} | MRR@10: {metrics['mrr@10']:.4f} | Fit: {train_info['fit_duration_seconds']:.2f}s"
        )

        entry = {
            "model_name": name,
            "feature_family": name.split("_")[1] if "_" in name else name,
            "num_features": len(feature_list),
            "fit_duration_seconds": train_info["fit_duration_seconds"],
            "inference_duration_seconds": round(inf_duration, 4),
            "best_iteration": train_info["best_iteration"],
            **metrics,
            "delta_ndcg@10_abs": round(delta_abs, 6),
            "delta_ndcg@10_rel_pct": round(delta_rel_pct, 4),
        }
        ablation_results.append(entry)

    # 7. Select Best Model by held-out NDCG@10
    print("\n[Step 6/8] Selecting best model and conducting bootstrap significance testing...")
    # Find best learned model (excluding baseline EASE)
    learned_entries = [e for e in ablation_results if e["model_name"] != "EASE_BASELINE"]
    best_entry = max(learned_entries, key=lambda x: x["ndcg@10"])
    best_model_name = best_entry["model_name"]
    best_model = trained_models[best_model_name]

    print(f"  Selected Best Model: {best_model_name} with NDCG@10 = {best_entry['ndcg@10']:.4f}")

    # Add best model rank column to eval_df
    best_score_col = f"score_{best_model_name.lower()}"
    eval_df["pred_selected"] = eval_df[best_score_col]
    # Compute rank (1 to 100) per user for selected model
    eval_df["reranked_rank_selected"] = (
        eval_df.groupby("user_idx")[best_score_col]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    # Bootstrap testing (1,000 resamples)
    n_bootstraps = cfg["evaluation"].get("n_bootstraps", 1000)
    b_seed = cfg["evaluation"].get("bootstrap_seed", 42)

    comparisons_to_test = [
        ("Best Model vs EASE", best_model_name, "EASE"),
        ("Qwen Emotion vs NRC Emotion (BASE)", "BASE_QWEN_EMOTION", "BASE_NRC_EMOTION"),
        ("KG-M + KG-L vs BASE", "BASE_KGM_KGL", "BASE"),
        ("Multimodal Qwen vs Multimodal NRC", "BASE_KGM_KGL_QWEN", "BASE_KGM_KGL_NRC"),
        ("Best Model vs BASE", best_model_name, "BASE"),
    ]

    bootstrap_results: Dict[str, Any] = {}
    print(f"\n  Running User-Level Paired Bootstrap (N={n_bootstraps}, Seed={b_seed}):")
    for comp_name, mod_a, mod_b in comparisons_to_test:
        vec_a = model_user_ndcg10[mod_a]
        vec_b = model_user_ndcg10[mod_b]
        b_res = compute_user_bootstrap(vec_a, vec_b, n_bootstraps=n_bootstraps, seed=b_seed)
        b_res["model_a"] = mod_a
        b_res["model_b"] = mod_b
        bootstrap_results[comp_name] = b_res
        sig_str = "SIGNIFICANT (p < 0.05)" if b_res["is_significant_95"] else "NOT significant (p >= 0.05)"
        print(
            f"    * {comp_name:<38}: Δ NDCG@10 = {b_res['point_delta']:+.4f} "
            f"[{b_res['ci_lower_95']:+.4f}, {b_res['ci_upper_95']:+.4f}] | p = {b_res['paired_p_value']:.4f} ({sig_str})"
        )

    # 8. Feature Importance for Best Model
    print(f"\n[Step 7/8] Computing feature importance for {best_model_name}...")
    best_features = ABLATION_CONFIGS[best_model_name]
    df_importance = compute_feature_importance(best_model, best_features)
    print("  Top 10 Features by Gain:")
    for idx, row in df_importance.head(10).iterrows():
        print(f"    {idx+1:2d}. {row['feature']:<36} ({row['feature_family']:<12}) -> Gain: {row['gain']:.2f} ({row['gain_pct']:.2f}%)")

    # Family-level gain aggregation
    family_importance = (
        df_importance.groupby("feature_family")
        .agg(total_gain=("gain", "sum"), total_gain_pct=("gain_pct", "sum"), feature_count=("feature", "count"))
        .sort_values("total_gain", ascending=False)
        .reset_index()
    )
    print("\n  Gain Importance by Feature Family:")
    for _, row in family_importance.iterrows():
        print(f"    * {row['feature_family']:<14} ({row['feature_count']:2d} features) -> {row['total_gain_pct']:.2f}% total gain")

    # 9. Serialize Artifacts
    print("\n[Step 8/8] Serializing artifacts...")
    # Results DataFrame and CSV
    df_results = pd.DataFrame(ablation_results)
    csv_path = Path(cfg["outputs"]["ablation_results_csv"])
    df_results.to_csv(csv_path, index=False)
    print(f"  * Saved ablation CSV: {csv_path}")

    # JSON results
    json_path = Path(cfg["outputs"]["ablation_results_json"])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"  * Saved ablation JSON: {json_path}")

    # Predictions Parquet
    pred_path = Path(cfg["outputs"]["eval_predictions_parquet"])
    eval_df.to_parquet(pred_path, index=False)
    print(f"  * Saved evaluation predictions: {pred_path} ({len(eval_df):,} rows)")

    # Feature Importance CSV
    fi_path = Path(cfg["outputs"]["feature_importance_csv"])
    df_importance.to_csv(fi_path, index=False)
    print(f"  * Saved feature importance: {fi_path}")

    # Save Best Model
    joblib_path = Path(cfg["outputs"]["selected_model_joblib"])
    joblib.dump(best_model, joblib_path)
    txt_model_path = Path(cfg["outputs"]["selected_model_txt"])
    best_model.booster_.save_model(str(txt_model_path))
    print(f"  * Saved model joblib: {joblib_path}")
    print(f"  * Saved model text:   {txt_model_path}")

    # Run summary
    run_summary = {
        "candidate_pool_stats": pool_stats,
        "splits": {
            "eval_size": eval_size,
            "inner_val_size": inner_val_size,
            "seed": seed,
            "eval_users_count": len(user_splits.eval_users),
            "train_users_count": len(user_splits.train_users),
            "fit_users_count": len(user_splits.fit_users),
            "inner_val_users_count": len(user_splits.inner_val_users),
        },
        "ease_baseline": ease_entry,
        "selected_best_model": {
            "model_name": best_model_name,
            "num_features": len(best_features),
            "metrics": best_entry,
            "features": best_features,
        },
        "bootstrap_significance": bootstrap_results,
        "family_importance": family_importance.to_dict(orient="records"),
    }

    summary_path = Path(cfg["outputs"]["run_summary_json"])
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print(f"  * Saved run summary: {summary_path}")

    # Print Final Markdown Results Table
    print("\n" + "=" * 105)
    print(f"{'Model':<20} | {'Feats':<5} | {'NDCG@10':<7} | {'Δ EASE':<7} | {'Rec@10(e2e)':<11} | {'Rec@10(c)':<9} | {'HR@10':<6} | {'MRR@10':<6} | {'NDCG@20':<7} | {'Rec@20(e2e)':<11}")
    print("-" * 105)
    for r in ablation_results:
        m_name = r["model_name"]
        n_feat = r["num_features"]
        ndcg10 = r["ndcg@10"]
        delta_str = f"{r['delta_ndcg@10_abs']:+.4f}" if m_name != "EASE_BASELINE" else "  ---  "
        rec10_e2e = r["recall@10"]
        rec10_c = r["recall_cond@10"]
        hr10 = r["hr@10"]
        mrr10 = r["mrr@10"]
        ndcg20 = r["ndcg@20"]
        rec20_e2e = r["recall@20"]
        print(f"{m_name:<20} | {n_feat:5d} | {ndcg10:7.4f} | {delta_str:<7} | {rec10_e2e:11.4f} | {rec10_c:9.4f} | {hr10:6.4f} | {mrr10:6.4f} | {ndcg20:7.4f} | {rec20_e2e:11.4f}")
    print("=" * 105)
    print("\nPhase 5A Learning-to-Rank Reranker pipeline completed successfully.")


if __name__ == "__main__":
    main()
