"""EASE candidate generator refit on positive TRAIN+VAL history for Phase 5B."""

import time
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.candidate_gen.ease import EASERecommender
from src.candidate_gen.evaluator import generate_and_save_candidate_pool


def refit_ease_on_train_val(
    train_val_df: pd.DataFrame,
    reg: float = 500.0,
    num_users: int = 6040,
    num_items: int = 3883,
) -> EASERecommender:
    """Refit the frozen EASE candidate generator (reg=500) on positive TRAIN+VAL interactions."""
    ease = EASERecommender(reg=reg, name="EASE")
    t0 = time.time()
    ease.fit(train_val_df, num_users=num_users, num_items=num_items)
    fit_duration = time.time() - t0
    return ease


def generate_test_candidates(
    model: EASERecommender,
    all_users: Sequence[int],
    k: int = 100,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate top-K candidate recommendations for all users excluding seen positive TRAIN+VAL items.

    Returns DataFrame containing:
    ['user_idx', 'movie_idx', 'rank', 'score', 'model_name']
    """
    candidates_df = generate_and_save_candidate_pool(
        model=model,
        all_users=all_users,
        k=k,
        output_path=output_path,
        exclude_seen=True,
    )
    return candidates_df


def validate_test_candidates(
    candidates_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
    expected_users: int = 6040,
    k: int = 100,
) -> dict:
    """Verify integrity of generated test candidate pool.

    Checks:
    - Exact row count = expected_users * k (604,000)
    - Exactly k candidates per user
    - Zero duplicate user/movie pairs
    - Zero positive TRAIN+VAL item leakage
    - Deterministic ranks 1..k
    - Finite EASE scores
    """
    expected_rows = expected_users * k
    actual_rows = len(candidates_df)
    if actual_rows != expected_rows:
        raise ValueError(f"Candidate row count mismatch: expected {expected_rows:,}, got {actual_rows:,}")

    actual_users = candidates_df["user_idx"].nunique()
    if actual_users != expected_users:
        raise ValueError(f"User count mismatch: expected {expected_users:,}, got {actual_users:,}")

    # Check candidates per user
    user_counts = candidates_df.groupby("user_idx").size()
    if not (user_counts == k).all():
        raise ValueError(f"Not every user has exactly {k} candidates!")

    # Check duplicates
    dups = candidates_df.duplicated(subset=["user_idx", "movie_idx"]).sum()
    if dups > 0:
        raise ValueError(f"Found {dups} duplicate (user_idx, movie_idx) candidate pairs!")

    # Check finite scores
    if not np.isfinite(candidates_df["score"].to_numpy()).all():
        raise ValueError("Detected non-finite EASE scores in candidate table!")

    # Check rank range
    if candidates_df["rank"].min() != 1 or candidates_df["rank"].max() != k:
        raise ValueError(f"Candidate ranks out of range: min={candidates_df['rank'].min()}, max={candidates_df['rank'].max()}")

    # Check seen positive TRAIN+VAL leakage
    pos_train_val = train_val_df[train_val_df["positive"] == 1]
    seen_dict = pos_train_val.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    leakage_count = 0
    for u, cand_group in candidates_df.groupby("user_idx"):
        seen_set = seen_dict.get(int(u), set())
        overlap = set(cand_group["movie_idx"]) & seen_set
        if overlap:
            leakage_count += len(overlap)

    if leakage_count > 0:
        raise ValueError(
            f"CRITICAL LEAKAGE: {leakage_count} positive TRAIN+VAL items were recommended in test candidates!"
        )

    return {
        "total_candidate_rows": actual_rows,
        "total_users": actual_users,
        "candidates_per_user": k,
        "duplicate_pairs": int(dups),
        "positive_train_val_leakage_count": leakage_count,
        "ease_score_min": float(candidates_df["score"].min()),
        "ease_score_max": float(candidates_df["score"].max()),
        "ease_score_mean": float(candidates_df["score"].mean()),
    }
