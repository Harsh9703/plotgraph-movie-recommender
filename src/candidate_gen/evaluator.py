"""Evaluation harness, grid search tuning, and candidate pool export utilities."""

import itertools
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import pandas as pd
from src.evaluation.ranking_metrics import evaluate_all_users, recall_at_k
from .base import BaseCandidateGenerator


def evaluate_candidate_model(
    model: BaseCandidateGenerator,
    val_df: pd.DataFrame,
    cutoffs: Sequence[int] = (10, 20),
    exclude_seen: bool = True,
) -> Dict[str, Any]:
    """Evaluate candidate generator against validation positives using full-catalog ranking."""
    if "positive" not in val_df.columns:
        raise ValueError("val_df must contain a 'positive' column.")

    val_users = val_df["user_idx"].unique().tolist()
    total_val_users = len(val_users)

    # Relevant items are positive interactions in validation (r >= 4)
    val_pos = val_df[val_df["positive"] == 1]
    ground_truth = val_pos.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    eval_val_users = len(ground_truth)
    eval_coverage_pct = (eval_val_users / total_val_users * 100.0) if total_val_users > 0 else 0.0

    max_k = max(cutoffs)
    recommendations = model.recommend_all(val_users, k=max_k, exclude_seen=exclude_seen)

    # Full-catalog ranking evaluation across unobserved items
    metrics = evaluate_all_users(
        recommendations=recommendations,
        ground_truth=ground_truth,
        cutoffs=cutoffs,
        exclude_empty_users=True,
    )

    return {
        "model_name": model.name,
        "metrics": metrics,
        "total_val_users": total_val_users,
        "eval_val_users": eval_val_users,
        "eval_coverage_pct": eval_coverage_pct,
        "cutoffs": list(cutoffs),
    }


def tune_grid(
    model_class: Type[BaseCandidateGenerator],
    param_grid: Dict[str, List[Any]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    target_metric: str = "ndcg@10",
    cutoffs: Sequence[int] = (10, 20),
    exclude_seen: bool = True,
    base_name: Optional[str] = None,
) -> Tuple[BaseCandidateGenerator, Dict[str, Any], float, List[Dict[str, Any]]]:
    """Perform validation hyperparameter grid search for a candidate generator.

    Args:
        model_class: The recommender class to instantiate.
        param_grid: Dict mapping hyperparameter names to lists of candidate values.
        train_df: Training DataFrame containing positive interactions.
        val_df: Validation DataFrame for model evaluation.
        num_users: Total universe users.
        num_items: Total universe items.
        target_metric: Metric key to maximize (default 'ndcg@10').
        cutoffs: Ranking cutoffs.
        exclude_seen: Whether to exclude positive training items.
        base_name: Base model display name.

    Returns:
        Tuple of (best_fitted_model, best_params, best_metric_val, all_grid_results).
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    all_results: List[Dict[str, Any]] = []
    best_metric = -float("inf")
    best_params: Dict[str, Any] = {}
    best_model: Optional[BaseCandidateGenerator] = None

    for comb in combinations:
        params = dict(zip(param_names, comb))
        name = base_name or model_class.__name__

        model = model_class(name=name, **params)

        t0 = time.perf_counter()
        model.fit(train_df, num_users=num_users, num_items=num_items)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        eval_res = evaluate_candidate_model(
            model, val_df, cutoffs=cutoffs, exclude_seen=exclude_seen
        )
        eval_time = time.perf_counter() - t0

        metric_val = eval_res["metrics"].get(target_metric, 0.0)

        record = {
            "params": params,
            "metrics": eval_res["metrics"],
            "target_metric": target_metric,
            "target_value": metric_val,
            "train_time": train_time,
            "eval_time": eval_time,
            "total_val_users": eval_res["total_val_users"],
            "eval_val_users": eval_res["eval_val_users"],
        }
        all_results.append(record)

        if metric_val > best_metric:
            best_metric = metric_val
            best_params = params
            best_model = model

    if best_model is None:
        raise RuntimeError("Grid search produced no valid model.")

    return best_model, best_params, best_metric, all_results


def generate_and_save_candidate_pool(
    model: BaseCandidateGenerator,
    all_users: Sequence[int],
    k: int = 100,
    output_path: Optional[Union[str, Path]] = None,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """Generate top-K candidate movies with scores for every user and optionally save to Parquet.

    Preserves:
        - user_idx
        - movie_idx
        - rank (1 to K)
        - score
        - model_name
    """
    recs_with_scores = model.recommend_all_with_scores(all_users, k=k, exclude_seen=exclude_seen)

    rows = []
    for u in all_users:
        user_cand = recs_with_scores.get(u, [])
        for rank, (item_id, score) in enumerate(user_cand, start=1):
            rows.append({
                "user_idx": int(u),
                "movie_idx": int(item_id),
                "rank": int(rank),
                "score": float(score),
                "model_name": str(model.name),
            })

    df_candidates = pd.DataFrame(rows)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_candidates.to_parquet(out_p, index=False)

    return df_candidates


def evaluate_candidate_pool_recall(
    candidates_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    k: int = 100,
) -> Dict[str, Any]:
    """Evaluate candidate pool Recall@K against validation positives and verify absence of seen train items."""
    # Filter candidates to top K ranks
    cand_k = candidates_df[candidates_df["rank"] <= k]
    user_cand = cand_k.groupby("user_idx")["movie_idx"].apply(list).to_dict()

    val_pos = val_df[val_df["positive"] == 1]
    ground_truth = val_pos.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    eval_users = [u for u, gt in ground_truth.items() if len(gt) > 0]

    recalls = []
    for u in eval_users:
        recs = user_cand.get(u, [])
        recalls.append(recall_at_k(recs, ground_truth[u], k=k))

    mean_recall = float(np.mean(recalls)) if recalls else 0.0

    # Verification: check that seen positive train items are completely absent
    seen_leakage_count = 0
    if train_df is not None:
        train_pos = train_df[train_df["positive"] == 1]
        train_seen = train_pos.groupby("user_idx")["movie_idx"].apply(set).to_dict()
        for u in user_cand:
            seen_set = train_seen.get(u, set())
            overlap = set(user_cand[u]) & seen_set
            if overlap:
                seen_leakage_count += len(overlap)

    avg_candidates_per_user = float(len(cand_k) / candidates_df["user_idx"].nunique()) if not candidates_df.empty else 0.0

    return {
        f"recall@{k}": mean_recall,
        "eval_users": len(eval_users),
        "total_users_in_pool": int(candidates_df["user_idx"].nunique()),
        "avg_candidates_per_user": avg_candidates_per_user,
        "seen_train_leakage_count": seen_leakage_count,
    }
