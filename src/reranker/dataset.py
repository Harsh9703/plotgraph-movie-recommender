"""Dataset utilities and leakage guards for Phase 5A LightGBM Reranker.

Handles:
- Loading candidate features and validation labels.
- Deterministic user-level splitting (train vs held-out evaluation, and inner fit vs early-stopping val).
- Candidate-pool coverage statistics.
- Strict leak prevention asserting TEST split is never accessed.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class UserSplitResult:
    """Deterministic user-level partition assignments for reranking."""

    all_evaluable_users: List[int]
    train_users: List[int]
    eval_users: List[int]
    fit_users: List[int]
    inner_val_users: List[int]


def assert_no_test_access(path_or_str: str) -> None:
    """Enforce strict isolation: prevent any access or reference to the TEST split."""
    normalized = str(path_or_str).lower().replace("\\", "/")
    if "test.parquet" in normalized or "splits/test" in normalized:
        raise ValueError(
            f"SECURITY VIOLATION: Test split access is strictly prohibited in Phase 5A! Path: {path_or_str}"
        )


def build_val_labels(
    candidate_df: pd.DataFrame,
    val_df: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """Merge binary validation labels into candidate feature table.

    A candidate (user_idx, movie_idx) receives label 1 if the user rated the movie
    >= positive_threshold in the VAL split; otherwise 0.
    """
    if "user_idx" not in candidate_df.columns or "movie_idx" not in candidate_df.columns:
        raise KeyError("candidate_df must contain 'user_idx' and 'movie_idx'")
    if "user_idx" not in val_df.columns or "movie_idx" not in val_df.columns:
        raise KeyError("val_df must contain 'user_idx' and 'movie_idx'")

    # Extract positive interactions from VAL
    if "positive" in val_df.columns:
        val_pos = val_df[val_df["positive"] == 1][["user_idx", "movie_idx"]].drop_duplicates()
    else:
        val_pos = val_df[val_df["rating"] >= positive_threshold][["user_idx", "movie_idx"]].drop_duplicates()

    val_pos = val_pos.copy()
    val_pos["val_label"] = 1

    merged = pd.merge(
        candidate_df,
        val_pos,
        on=["user_idx", "movie_idx"],
        how="left",
    )
    merged["val_label"] = merged["val_label"].fillna(0).astype(int)
    return merged


def get_candidate_pool_stats(
    candidate_df: pd.DataFrame,
    val_df: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> Dict[str, Any]:
    """Compute candidate-pool limitation and coverage statistics."""
    if "positive" in val_df.columns:
        val_pos = val_df[val_df["positive"] == 1]
    else:
        val_pos = val_df[val_df["rating"] >= positive_threshold]

    total_val_interactions = int(len(val_df))
    total_val_positives = int(len(val_pos))
    users_with_val_positive = int(val_pos["user_idx"].nunique())

    # Map each user to set of positive VAL movie_idx
    val_pos_dict = val_pos.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    # Candidate labels
    if "val_label" not in candidate_df.columns:
        cand_labeled = build_val_labels(candidate_df, val_df, positive_threshold=positive_threshold)
    else:
        cand_labeled = candidate_df

    captured_val_positives = int(cand_labeled["val_label"].sum())
    global_recall100 = (
        float(captured_val_positives) / float(total_val_positives) if total_val_positives > 0 else 0.0
    )

    # Per-user statistics
    user_cand_pos = cand_labeled.groupby("user_idx")["val_label"].sum()
    users_with_captured_positive = int((user_cand_pos >= 1).sum())
    total_candidate_users = int(candidate_df["user_idx"].nunique())
    users_with_zero_captured_positive = total_candidate_users - users_with_captured_positive

    # Mean Recall@100 across users who have >= 1 positive VAL item
    user_recalls = []
    for u, pos_set in val_pos_dict.items():
        if len(pos_set) > 0:
            c_pos = user_cand_pos.get(u, 0)
            user_recalls.append(float(c_pos) / float(len(pos_set)))
    mean_user_recall100 = float(np.mean(user_recalls)) if user_recalls else 0.0

    return {
        "total_val_interactions": total_val_interactions,
        "total_val_positives": total_val_positives,
        "users_with_val_positive": users_with_val_positive,
        "total_candidate_users": total_candidate_users,
        "captured_val_positives": captured_val_positives,
        "candidate_pool_recall100_global": round(global_recall100, 4),
        "candidate_pool_recall100_mean_user": round(mean_user_recall100, 4),
        "users_with_captured_positive": users_with_captured_positive,
        "users_with_zero_captured_positive": users_with_zero_captured_positive,
        "evaluable_user_ratio": round(users_with_captured_positive / total_candidate_users, 4),
    }


def create_user_splits(
    candidate_df: pd.DataFrame,
    eval_size: float = 0.20,
    inner_val_size: float = 0.20,
    seed: int = 42,
) -> UserSplitResult:
    """Create deterministic user-level splits among evaluable users.

    Evaluable users are defined as users with >= 1 positive candidate captured in the top-100 pool,
    since ranking queries with zero positives cannot provide learning gradients or meaningful ranking.

    Partitioning:
    - 1 - eval_size (80%) -> train_users
    - eval_size (20%) -> eval_users (held out strictly for final evaluation)
    - Inside train_users:
      * 1 - inner_val_size (80%) -> fit_users
      * inner_val_size (20%) -> inner_val_users (for early stopping)
    """
    if "val_label" not in candidate_df.columns:
        raise ValueError("candidate_df must have 'val_label' before splitting")

    user_pos_count = candidate_df.groupby("user_idx")["val_label"].sum()
    evaluable_users = user_pos_count[user_pos_count >= 1].index.values
    evaluable_counts = user_pos_count.loc[evaluable_users].values

    # Deterministic stratification by binned positive candidate counts (1, 2, 3, 4, 5+)
    bins = np.minimum(evaluable_counts, 5)

    train_users, eval_users = train_test_split(
        evaluable_users,
        test_size=eval_size,
        random_state=seed,
        stratify=bins,
    )

    # Inner train/val split on train_users for early stopping
    train_pos_counts = user_pos_count.loc[train_users].values
    train_bins = np.minimum(train_pos_counts, 5)

    fit_users, inner_val_users = train_test_split(
        train_users,
        test_size=inner_val_size,
        random_state=seed,
        stratify=train_bins,
    )

    # Integrity verification
    assert len(set(train_users) & set(eval_users)) == 0, "Train and Eval user overlap detected!"
    assert len(set(fit_users) & set(inner_val_users)) == 0, "Fit and Inner-Val user overlap detected!"
    assert set(fit_users) | set(inner_val_users) == set(train_users), "Fit + Inner-Val mismatch!"

    return UserSplitResult(
        all_evaluable_users=evaluable_users.tolist(),
        train_users=train_users.tolist(),
        eval_users=eval_users.tolist(),
        fit_users=fit_users.tolist(),
        inner_val_users=inner_val_users.tolist(),
    )


def build_lgbm_dataset(
    candidate_df: pd.DataFrame,
    user_list: Sequence[int],
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[int], pd.DataFrame]:
    """Filter and prepare contiguous ranking groups for LightGBM.

    Ensures rows are contiguously ordered by `['user_idx', 'candidate_rank']`.
    Returns:
    - X: feature matrix (N, num_features)
    - y: labels (N,)
    - groups: list of group lengths (e.g. [100, 100, ...])
    - filtered_df: sorted DataFrame for the partition
    """
    user_set = set(user_list)
    filtered = candidate_df[candidate_df["user_idx"].isin(user_set)].copy()

    # Stable contiguous sorting by user_idx and candidate_rank
    filtered = filtered.sort_values(["user_idx", "candidate_rank"])

    # Group counts
    group_sizes = filtered.groupby("user_idx", sort=False).size().tolist()

    if sum(group_sizes) != len(filtered):
        raise ValueError(f"Group sizes sum ({sum(group_sizes)}) != row count ({len(filtered)})")

    X = filtered[feature_cols].values.astype(np.float32)
    y = filtered["val_label"].values.astype(np.int32)

    return X, y, group_sizes, filtered
