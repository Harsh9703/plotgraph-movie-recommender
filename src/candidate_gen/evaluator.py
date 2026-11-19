"""Evaluation harness for candidate recommendation models on the validation split."""

from typing import Any, Dict, Sequence
import pandas as pd
from src.evaluation.ranking_metrics import evaluate_all_users
from .base import BaseCandidateGenerator


def evaluate_candidate_model(
    model: BaseCandidateGenerator,
    val_df: pd.DataFrame,
    cutoffs: Sequence[int] = (10, 20),
    exclude_seen: bool = True,
) -> Dict[str, Any]:
    """Evaluate candidate generator against validation positives using full-catalog ranking.

    Args:
        model: Fitted candidate recommender.
        val_df: Validation DataFrame with 'user_idx', 'movie_idx', 'positive'.
        cutoffs: Ranking cutoffs K to evaluate.
        exclude_seen: Whether to exclude positive training items.

    Returns:
        Dict containing:
            - 'metrics': Dict with keys like 'ndcg@10', 'recall@10', 'hr@10', 'mrr@10', etc.
            - 'total_val_users': Total unique users in validation.
            - 'eval_val_users': Number of validation users with >=1 positive item.
            - 'eval_coverage_pct': Percentage of validation users with >=1 positive item.
    """
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
