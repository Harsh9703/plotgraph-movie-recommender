"""Phase 5C Final MovieLens Test Evaluation and Audit Engine.

Handles:
- Cryptographic verification of frozen Phase 5B manifest and artifact checksums.
- Test-set coverage and candidate pool ceiling analysis.
- Read-only evaluation of frozen EASE vs frozen PlotGraph reranker.
- End-to-end and conditional population metric calculations.
- Paired user-level bootstrap significance analysis (95% CI).
- Detailed per-user win/loss and relevant item rank-movement analysis.
- Production of final audit record.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.evaluation.ranking_metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.final_eval.model_refit import compute_file_sha256
from src.reranker.evaluator import compute_user_bootstrap

logger = logging.getLogger(__name__)


def verify_manifest_and_frozen_artifacts(
    manifest_path: Union[str, Path] = "data/processed/ml-1m/final_eval/final_evaluation_manifest.json",
) -> Dict[str, Any]:
    """Verify all SHA-256 checksums and properties recorded in the Phase 5B manifest.

    Fails with ValueError if any artifact is missing or has been altered.
    """
    p_manifest = Path(manifest_path)
    if not p_manifest.exists():
        raise FileNotFoundError(f"Final evaluation manifest not found: {p_manifest}")

    with open(p_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info(f"Loaded manifest version {manifest.get('manifest_version')} created at {manifest.get('created_at_utc')}")

    # Check methodology fields
    meth = manifest.get("frozen_methodology", {})
    if meth.get("num_model_features") != 89:
        raise ValueError(f"Manifest specifies {meth.get('num_model_features')} features, expected 89!")

    cand_summary = manifest.get("candidate_pool_summary", {})
    if cand_summary.get("total_candidate_rows") != 604000:
        raise ValueError(f"Manifest candidate rows {cand_summary.get('total_candidate_rows')} != 604,000!")

    # Verify checksums for each recorded artifact
    checksums = manifest.get("artifact_checksums", {})
    if not checksums:
        raise ValueError("Manifest contains no artifact checksums!")

    verified_artifacts = {}
    for name, item in checksums.items():
        file_path = Path(item["path"])
        expected_sha = item["sha256"]
        expected_size = item["size_bytes"]

        if not file_path.exists():
            raise FileNotFoundError(f"Frozen artifact '{name}' missing at {file_path}")

        actual_size = file_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"ARTIFACT TAMPER DETECTED: '{name}' size mismatch: expected {expected_size}, got {actual_size}!"
            )

        actual_sha = compute_file_sha256(file_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"SECURITY AUDIT FAILURE: SHA-256 mismatch for frozen artifact '{name}'!\n"
                f"  Expected: {expected_sha}\n"
                f"  Actual:   {actual_sha}"
            )
        verified_artifacts[name] = actual_sha
        logger.info(f"Verified artifact '{name}' -> SHA-256: {actual_sha[:16]}... (OK)")

    manifest["manifest_sha256"] = compute_file_sha256(p_manifest)
    return manifest


def compute_test_coverage_stats(
    test_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> Dict[str, Any]:
    """Compute test set interaction statistics and candidate pool ceiling."""
    if "positive" in test_df.columns:
        test_pos = test_df[test_df["positive"] == 1]
    else:
        test_pos = test_df[test_df["rating"] >= positive_threshold]

    total_test_interactions = int(len(test_df))
    total_positive_interactions = int(len(test_pos))
    users_with_test_positive = int(test_pos["user_idx"].nunique())

    # Map user_idx -> set of positive movie_idx in TEST
    test_pos_dict = test_pos.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    # Match candidate table with test positives
    cand_pairs = set(zip(candidates_df["user_idx"], candidates_df["movie_idx"]))
    test_pos_pairs = set(zip(test_pos["user_idx"], test_pos["movie_idx"]))

    captured_positives = len(cand_pairs & test_pos_pairs)
    global_recall100 = (
        float(captured_positives) / float(total_positive_interactions)
        if total_positive_interactions > 0
        else 0.0
    )

    # Per-user recall calculation
    user_cand_dict = candidates_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    user_recalls = []
    users_with_captured = 0

    for u, pos_set in test_pos_dict.items():
        if len(pos_set) > 0:
            cands = user_cand_dict.get(u, set())
            overlap_count = len(pos_set & cands)
            user_recalls.append(float(overlap_count) / float(len(pos_set)))
            if overlap_count > 0:
                users_with_captured += 1

    mean_user_recall100 = float(np.mean(user_recalls)) if user_recalls else 0.0
    users_with_zero_captured = users_with_test_positive - users_with_captured

    return {
        "total_test_interactions": total_test_interactions,
        "positive_test_interactions": total_positive_interactions,
        "users_with_test_positive": users_with_test_positive,
        "captured_test_positives": captured_positives,
        "candidate_pool_recall100_global": round(global_recall100, 4),
        "candidate_pool_recall100_mean_user": round(mean_user_recall100, 4),
        "users_with_captured_positive": users_with_captured,
        "users_with_zero_captured_positive": users_with_zero_captured,
        "candidate_coverage_pct": round(users_with_captured / users_with_test_positive * 100.0, 2) if users_with_test_positive > 0 else 0.0,
    }


def evaluate_test_predictions(
    unlabeled_preds_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cutoffs: Sequence[int] = (10, 20),
    positive_threshold: float = 4.0,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Evaluate frozen EASE vs frozen PlotGraph rankings on TEST.

    Evaluates across:
    - Population A (End-to-End): All users with >=1 positive TEST interaction.
    - Population B (Conditional): Users with >=1 positive TEST candidate in top-100 pool.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]
        - end_to_end_results: Metrics for Population A.
        - conditional_results: Metrics for Population B.
        - per_user_df: Detailed per-user metric table.
    """
    if "positive" in test_df.columns:
        test_pos = test_df[test_df["positive"] == 1]
    else:
        test_pos = test_df[test_df["rating"] >= positive_threshold]

    test_pos_dict: Dict[int, Set[int]] = test_pos.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    eval_users = sorted(test_pos_dict.keys())
    n_eval_users = len(eval_users)

    # Sort candidates contiguously by user_idx
    sorted_df = unlabeled_preds_df.sort_values(["user_idx", "candidate_rank"]).copy()
    grouped = {u: g for u, g in sorted_df.groupby("user_idx", sort=False)}

    # Container for per-user records
    per_user_records: List[Dict[str, Any]] = []

    # Vectors for aggregation
    ease_e2e_metrics: Dict[str, List[float]] = {f"{m}@{k}": [] for k in cutoffs for m in ["ndcg", "recall", "hr", "mrr"]}
    plotgraph_e2e_metrics: Dict[str, List[float]] = {f"{m}@{k}": [] for k in cutoffs for m in ["ndcg", "recall", "hr", "mrr"]}

    ease_cond_metrics: Dict[str, List[float]] = {f"{m}@{k}": [] for k in cutoffs for m in ["ndcg", "recall", "hr", "mrr"]}
    plotgraph_cond_metrics: Dict[str, List[float]] = {f"{m}@{k}": [] for k in cutoffs for m in ["ndcg", "recall", "hr", "mrr"]}

    conditional_mask: List[bool] = []

    for u in eval_users:
        all_pos = test_pos_dict[u]
        n_pos = len(all_pos)

        user_rows = grouped.get(u)
        if user_rows is None or user_rows.empty:
            pool_pos = set()
            recs_ease: List[int] = []
            recs_plotgraph: List[int] = []
        else:
            pool_items = set(user_rows["movie_idx"])
            pool_pos = all_pos & pool_items

            # EASE list: sorted by candidate_rank ascending (1 to 100)
            recs_ease = user_rows.sort_values("candidate_rank")["movie_idx"].tolist()
            # PlotGraph list: sorted by reranked_rank ascending (1 to 100)
            recs_plotgraph = user_rows.sort_values("reranked_rank")["movie_idx"].tolist()

        is_cond = len(pool_pos) > 0
        conditional_mask.append(is_cond)

        u_record: Dict[str, Any] = {
            "user_idx": int(u),
            "num_positive_test_items": n_pos,
            "num_captured_in_top100": len(pool_pos),
            "has_captured_positive": is_cond,
        }

        # Compute metrics across cutoffs
        for k in cutoffs:
            # End-to-end metrics against all test positives
            e_ndcg = ndcg_at_k(recs_ease, all_pos, k)
            e_rec = recall_at_k(recs_ease, all_pos, k)
            e_hr = hit_rate_at_k(recs_ease, all_pos, k)
            e_mrr = mrr_at_k(recs_ease, all_pos, k)

            p_ndcg = ndcg_at_k(recs_plotgraph, all_pos, k)
            p_rec = recall_at_k(recs_plotgraph, all_pos, k)
            p_hr = hit_rate_at_k(recs_plotgraph, all_pos, k)
            p_mrr = mrr_at_k(recs_plotgraph, all_pos, k)

            ease_e2e_metrics[f"ndcg@{k}"].append(e_ndcg)
            ease_e2e_metrics[f"recall@{k}"].append(e_rec)
            ease_e2e_metrics[f"hr@{k}"].append(e_hr)
            ease_e2e_metrics[f"mrr@{k}"].append(e_mrr)

            plotgraph_e2e_metrics[f"ndcg@{k}"].append(p_ndcg)
            plotgraph_e2e_metrics[f"recall@{k}"].append(p_rec)
            plotgraph_e2e_metrics[f"hr@{k}"].append(p_hr)
            plotgraph_e2e_metrics[f"mrr@{k}"].append(p_mrr)

            u_record[f"ease_ndcg@{k}"] = round(e_ndcg, 6)
            u_record[f"plotgraph_ndcg@{k}"] = round(p_ndcg, 6)
            u_record[f"delta_ndcg@{k}"] = round(p_ndcg - e_ndcg, 6)

            u_record[f"ease_recall@{k}"] = round(e_rec, 6)
            u_record[f"plotgraph_recall@{k}"] = round(p_rec, 6)
            u_record[f"delta_recall@{k}"] = round(p_rec - e_rec, 6)

            u_record[f"ease_hr@{k}"] = round(e_hr, 6)
            u_record[f"plotgraph_hr@{k}"] = round(p_hr, 6)
            u_record[f"delta_hr@{k}"] = round(p_hr - e_hr, 6)

            u_record[f"ease_mrr@{k}"] = round(e_mrr, 6)
            u_record[f"plotgraph_mrr@{k}"] = round(p_mrr, 6)
            u_record[f"delta_mrr@{k}"] = round(p_mrr - e_mrr, 6)

            # Conditional metrics against in-pool positives
            if is_cond:
                e_ndcg_c = ndcg_at_k(recs_ease, pool_pos, k)
                e_rec_c = recall_at_k(recs_ease, pool_pos, k)
                e_hr_c = hit_rate_at_k(recs_ease, pool_pos, k)
                e_mrr_c = mrr_at_k(recs_ease, pool_pos, k)

                p_ndcg_c = ndcg_at_k(recs_plotgraph, pool_pos, k)
                p_rec_c = recall_at_k(recs_plotgraph, pool_pos, k)
                p_hr_c = hit_rate_at_k(recs_plotgraph, pool_pos, k)
                p_mrr_c = mrr_at_k(recs_plotgraph, pool_pos, k)

                ease_cond_metrics[f"ndcg@{k}"].append(e_ndcg_c)
                ease_cond_metrics[f"recall@{k}"].append(e_rec_c)
                ease_cond_metrics[f"hr@{k}"].append(e_hr_c)
                ease_cond_metrics[f"mrr@{k}"].append(e_mrr_c)

                plotgraph_cond_metrics[f"ndcg@{k}"].append(p_ndcg_c)
                plotgraph_cond_metrics[f"recall@{k}"].append(p_rec_c)
                plotgraph_cond_metrics[f"hr@{k}"].append(p_hr_c)
                plotgraph_cond_metrics[f"mrr@{k}"].append(p_mrr_c)

        per_user_records.append(u_record)

    per_user_df = pd.DataFrame(per_user_records)

    # Helper for summarizing results table
    def build_summary(ease_dict: Dict[str, List[float]], pg_dict: Dict[str, List[float]], pop_name: str, n_users: int) -> Dict[str, Any]:
        res = {"population": pop_name, "num_users": n_users}
        for k in cutoffs:
            for m in ["ndcg", "recall", "hr", "mrr"]:
                key = f"{m}@{k}"
                e_val = float(np.mean(ease_dict[key])) if ease_dict[key] else 0.0
                p_val = float(np.mean(pg_dict[key])) if pg_dict[key] else 0.0
                delta = p_val - e_val
                pct = (delta / e_val * 100.0) if e_val > 0 else 0.0
                res[f"ease_{key}"] = round(e_val, 6)
                res[f"plotgraph_{key}"] = round(p_val, 6)
                res[f"delta_{key}_abs"] = round(delta, 6)
                res[f"delta_{key}_rel_pct"] = round(pct, 4)
        return res

    e2e_summary = build_summary(
        ease_e2e_metrics, plotgraph_e2e_metrics, "End-to-End (All Users with >=1 Positive TEST Interaction)", n_eval_users
    )
    n_cond = sum(conditional_mask)
    cond_summary = build_summary(
        ease_cond_metrics, plotgraph_cond_metrics, "Conditional (Users with >=1 Positive TEST Candidate in Pool)", n_cond
    )

    # Attach raw metric arrays for bootstrap calculations
    e2e_summary["_arrays"] = {
        "ease": {k: np.array(v, dtype=np.float64) for k, v in ease_e2e_metrics.items()},
        "plotgraph": {k: np.array(v, dtype=np.float64) for k, v in plotgraph_e2e_metrics.items()},
    }

    return e2e_summary, cond_summary, per_user_df


def analyze_win_loss(per_user_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze distribution of improved, tied, and worsened users under NDCG@10."""
    total_users = len(per_user_df)
    deltas = per_user_df["delta_ndcg@10"].to_numpy()

    improved = deltas > 0.000001
    worsened = deltas < -0.000001
    tied = np.abs(deltas) <= 0.000001

    n_improved = int(np.sum(improved))
    n_worsened = int(np.sum(worsened))
    n_tied = int(np.sum(tied))

    mean_improved_delta = float(np.mean(deltas[improved])) if n_improved > 0 else 0.0
    mean_worsened_delta = float(np.mean(deltas[worsened])) if n_worsened > 0 else 0.0

    return {
        "total_evaluated_users": total_users,
        "improved_users_count": n_improved,
        "improved_users_pct": round(n_improved / total_users * 100.0, 2),
        "tied_users_count": n_tied,
        "tied_users_pct": round(n_tied / total_users * 100.0, 2),
        "worsened_users_count": n_worsened,
        "worsened_users_pct": round(n_worsened / total_users * 100.0, 2),
        "mean_delta_among_improved": round(mean_improved_delta, 6),
        "mean_delta_among_worsened": round(mean_worsened_delta, 6),
        "net_improved_ratio": round(n_improved / n_worsened, 2) if n_worsened > 0 else None,
    }


def analyze_rank_movements(
    unlabeled_preds_df: pd.DataFrame,
    test_df: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Analyze how PlotGraph moves relevant TEST items relative to EASE."""
    if "positive" in test_df.columns:
        test_pos = test_df[test_df["positive"] == 1]
    else:
        test_pos = test_df[test_df["rating"] >= positive_threshold]

    test_pos_pairs = set(zip(test_pos["user_idx"], test_pos["movie_idx"]))

    # Match candidate rows
    cand_user_movie = list(zip(unlabeled_preds_df["user_idx"], unlabeled_preds_df["movie_idx"]))
    is_relevant_cand = [pair in test_pos_pairs for pair in cand_user_movie]

    rel_cand_df = unlabeled_preds_df[is_relevant_cand].copy()
    if rel_cand_df.empty:
        return {"total_relevant_items_in_pool": 0}, pd.DataFrame()

    ease_ranks = rel_cand_df["candidate_rank"].to_numpy(dtype=np.int32)
    pg_ranks = rel_cand_df["reranked_rank"].to_numpy(dtype=np.int32)

    # Rank delta: positive means rank decreased (e.g. 20 -> 5 = +15, improved towards 1)
    rank_deltas = ease_ranks - pg_ranks

    moved_up = rank_deltas > 0
    moved_down = rank_deltas < 0
    unchanged = rank_deltas == 0

    crossed_into_10 = (ease_ranks > 10) & (pg_ranks <= 10)
    crossed_out_of_10 = (ease_ranks <= 10) & (pg_ranks > 10)

    crossed_into_20 = (ease_ranks > 20) & (pg_ranks <= 20)
    crossed_out_of_20 = (ease_ranks <= 20) & (pg_ranks > 20)

    rel_cand_df["ease_rank"] = ease_ranks
    rel_cand_df["plotgraph_rank"] = pg_ranks
    rel_cand_df["rank_delta"] = rank_deltas
    rel_cand_df["movement"] = np.where(moved_up, "UP", np.where(moved_down, "DOWN", "SAME"))
    rel_cand_df["crossed_into_top10"] = crossed_into_10
    rel_cand_df["crossed_out_of_top10"] = crossed_out_of_10

    n_total = len(rel_cand_df)
    summary = {
        "total_relevant_items_in_pool": n_total,
        "mean_ease_rank": round(float(np.mean(ease_ranks)), 2),
        "mean_plotgraph_rank": round(float(np.mean(pg_ranks)), 2),
        "mean_rank_change": round(float(np.mean(rank_deltas)), 2),
        "median_rank_change": round(float(np.median(rank_deltas)), 2),
        "items_moved_upward_count": int(np.sum(moved_up)),
        "items_moved_upward_pct": round(np.sum(moved_up) / n_total * 100.0, 2),
        "items_moved_downward_count": int(np.sum(moved_down)),
        "items_moved_downward_pct": round(np.sum(moved_down) / n_total * 100.0, 2),
        "items_unchanged_count": int(np.sum(unchanged)),
        "items_unchanged_pct": round(np.sum(unchanged) / n_total * 100.0, 2),
        "crossed_into_top10_count": int(np.sum(crossed_into_10)),
        "crossed_out_of_top10_count": int(np.sum(crossed_out_of_10)),
        "net_top10_gain": int(np.sum(crossed_into_10) - np.sum(crossed_out_of_10)),
        "crossed_into_top20_count": int(np.sum(crossed_into_20)),
        "crossed_out_of_top20_count": int(np.sum(crossed_out_of_20)),
        "net_top20_gain": int(np.sum(crossed_into_20) - np.sum(crossed_out_of_20)),
    }

    out_cols = [
        "user_idx",
        "movie_idx",
        "ease_rank",
        "plotgraph_rank",
        "rank_delta",
        "movement",
        "crossed_into_top10",
        "crossed_out_of_top10",
    ]
    return summary, rel_cand_df[out_cols].copy()


def compute_test_bootstrap_uncertainty(
    metric_arrays: Dict[str, Dict[str, np.ndarray]],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute deterministic paired user-level bootstrap confidence intervals for PlotGraph - EASE."""
    ease_arrs = metric_arrays["ease"]
    pg_arrs = metric_arrays["plotgraph"]

    comparisons = ["ndcg@10", "recall@10", "hr@10", "mrr@10", "ndcg@20", "recall@20", "hr@20", "mrr@20"]
    bootstrap_results = {}

    for metric in comparisons:
        if metric in pg_arrs and metric in ease_arrs:
            vec_pg = pg_arrs[metric]
            vec_ease = ease_arrs[metric]
            res = compute_user_bootstrap(vec_pg, vec_ease, n_bootstraps=n_bootstraps, seed=seed)
            bootstrap_results[f"delta_{metric}"] = res

    return bootstrap_results
