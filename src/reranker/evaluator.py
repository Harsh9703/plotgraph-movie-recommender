"""Validation-stage evaluation metrics and bootstrap statistical analysis for Phase 5A."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.evaluation.ranking_metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def evaluate_candidate_rankings(
    eval_df: pd.DataFrame,
    score_col: str,
    val_pos_dict: Dict[int, Set[int]],
    cutoffs: Sequence[int] = (10, 20),
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate candidate rankings across held-out evaluation users.

    Ranks each user's candidate pool by `score_col` descending, breaking ties
    with `candidate_rank` ascending (preserving EASE ordering when scores tie).

    Computes:
    - End-to-End Validation metrics (evaluated against all user positives in VAL):
      * NDCG@K
      * Recall@K (End-to-End Recall@K, ceiling = Recall@100 of candidate generator)
      * HR@K
      * MRR@K
    - Conditional Reranking metrics (evaluated against positives captured in candidate pool):
      * Conditional NDCG@K
      * Conditional Recall@K (ceiling = 1.0)

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, np.ndarray]]
        - mean_metrics: Dictionary of population-averaged metrics across eval users.
        - per_user_metrics: Dictionary of per-user metric vectors (useful for bootstrap testing).
    """
    if "user_idx" not in eval_df.columns or "movie_idx" not in eval_df.columns:
        raise KeyError("eval_df must contain 'user_idx' and 'movie_idx'")
    if "candidate_rank" not in eval_df.columns:
        raise KeyError("eval_df must contain 'candidate_rank'")
    if score_col not in eval_df.columns:
        raise KeyError(f"eval_df must contain score column '{score_col}'")

    # Sort stably: highest score first; if tied, lowest original candidate_rank first
    sorted_df = eval_df.sort_values(
        by=["user_idx", score_col, "candidate_rank"],
        ascending=[True, False, True],
    )

    per_user_data: Dict[str, List[float]] = {}
    for k in cutoffs:
        per_user_data[f"ndcg@{k}"] = []
        per_user_data[f"ndcg_cond@{k}"] = []
        per_user_data[f"recall@{k}"] = []
        per_user_data[f"recall_e2e@{k}"] = []
        per_user_data[f"recall_cond@{k}"] = []
        per_user_data[f"hr@{k}"] = []
        per_user_data[f"mrr@{k}"] = []

    # Group by user_idx
    for user_idx, group in sorted_df.groupby("user_idx", sort=False):
        recs = group["movie_idx"].tolist()
        val_all_pos = val_pos_dict.get(int(user_idx), set())
        pool_pos = set(group[group["val_label"] == 1]["movie_idx"])

        for k in cutoffs:
            # End-to-end metrics against all user positive items in VAL
            ndcg_val = ndcg_at_k(recs, val_all_pos, k)
            rec_e2e = recall_at_k(recs, val_all_pos, k)
            hr_val = hit_rate_at_k(recs, val_all_pos, k)
            mrr_val = mrr_at_k(recs, val_all_pos, k)

            # Conditional metrics against positive items captured in candidate pool
            ndcg_cond_val = ndcg_at_k(recs, pool_pos, k)
            rec_cond = recall_at_k(recs, pool_pos, k)

            per_user_data[f"ndcg@{k}"].append(ndcg_val)
            per_user_data[f"ndcg_cond@{k}"].append(ndcg_cond_val)
            per_user_data[f"recall@{k}"].append(rec_e2e)
            per_user_data[f"recall_e2e@{k}"].append(rec_e2e)
            per_user_data[f"recall_cond@{k}"].append(rec_cond)
            per_user_data[f"hr@{k}"].append(hr_val)
            per_user_data[f"mrr@{k}"].append(mrr_val)

    # Aggregate means
    mean_metrics: Dict[str, float] = {}
    per_user_arrays: Dict[str, np.ndarray] = {}
    for key, vals in per_user_data.items():
        arr = np.array(vals, dtype=np.float64)
        mean_metrics[key] = round(float(np.mean(arr)), 6)
        per_user_arrays[key] = arr

    return mean_metrics, per_user_arrays


def compute_user_bootstrap(
    metric_user_a: np.ndarray,
    metric_user_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute user-level paired bootstrap confidence interval for Delta = A - B.

    Parameters
    ----------
    metric_user_a : np.ndarray
        Metric array per evaluation user for model A.
    metric_user_b : np.ndarray
        Metric array per evaluation user for model B.
    n_bootstraps : int
        Number of bootstrap resamples (default: 1,000).
    seed : int
        Deterministic random seed.

    Returns
    -------
    Dict[str, Any]
        Bootstrap statistics including 95% CI and paired probability.
    """
    if len(metric_user_a) != len(metric_user_b):
        raise ValueError("Arrays must have identical user count for paired bootstrap")

    n_users = len(metric_user_a)
    diff = metric_user_a - metric_user_b
    point_delta = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n_users, size=(n_bootstraps, n_users))
    sample_deltas = np.mean(diff[resample_indices], axis=1)

    ci_lower = float(np.percentile(sample_deltas, 2.5))
    ci_upper = float(np.percentile(sample_deltas, 97.5))
    prob_a_superior = float(np.mean(sample_deltas > 0.0))
    p_cross_zero = float(min(np.mean(sample_deltas <= 0.0), np.mean(sample_deltas >= 0.0)))
    two_tailed_p = float(min(1.0, 2.0 * p_cross_zero))

    return {
        "point_delta": round(point_delta, 6),
        "ci_lower_95": round(ci_lower, 6),
        "ci_upper_95": round(ci_upper, 6),
        "prob_a_superior": round(prob_a_superior, 4),
        "paired_p_value": round(two_tailed_p, 4),
        "n_bootstraps": n_bootstraps,
        "n_users": n_users,
        "is_significant_95": bool(ci_lower > 0.0 or ci_upper < 0.0),
    }
