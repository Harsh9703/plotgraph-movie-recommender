"""Chronological per-user interaction splitting."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_split_sizes(n: int, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[int, int, int]:
    """Calculate exact integer split sizes (n_train, n_val, n_test) for a user with n interactions.

    Behavior for rounding and small histories:
    - If n >= 3: ensures n_train >= 1, n_val >= 1, n_test >= 1.
      Computes n_val = max(1, round(n * val_ratio)), n_test = max(1, round(n * test_ratio)),
      and n_train = n - n_val - n_test. If n_train < 1, boundaries are decremented so n_train >= 1.
    - If n == 2: n_train = 1, n_val = 0, n_test = 1.
    - If n == 1: n_train = 1, n_val = 0, n_test = 0.
    - If n == 0: 0, 0, 0.
    - For all MovieLens-1M users (n >= 20), this yields approximately 80% train, 10% val, 10% test.
      For example:
        n = 20  -> (16, 2, 2)  (exact 80/10/10)
        n = 100 -> (80, 10, 10) (exact 80/10/10)
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    n_train = n - n_val - n_test

    # Safeguard if rounding pushed n_train to 0
    while n_train < 1:
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
        n_train = n - n_val - n_test

    return n_train, n_val, n_test


def split_chronological_per_user(
    df: pd.DataFrame,
    user_col: str = "user_id",
    timestamp_col: str = "timestamp",
    item_col: str = "movie_id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    """Split interactions per user chronologically into train, validation, and test sets.

    1. Interactions for each user are sorted ascending by timestamp.
    2. Ties in timestamps are broken deterministically by item ID ascending.
    3. The sorted history is partitioned: [0 : n_train], [n_train : n_train + n_val], [n_train + n_val : n].
    4. Both original ratings and binary positive indicators are preserved.

    Args:
        df: DataFrame containing user interactions.
        user_col: User ID column name.
        timestamp_col: Timestamp column name.
        item_col: Item ID column name (used for deterministic tie breaking).
        train_ratio: Proportion of interactions in training (default 0.8).
        val_ratio: Proportion of interactions in validation (default 0.1).
        test_ratio: Proportion of interactions in test (default 0.1).

    Returns:
        Dict with keys 'train', 'val', 'test' containing corresponding DataFrames.
    """
    required_cols = {user_col, timestamp_col, item_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Ensure deterministic sorting
    sorted_df = df.sort_values(by=[user_col, timestamp_col, item_col]).reset_index(drop=True)

    train_indices = []
    val_indices = []
    test_indices = []

    # Group by user and split
    # Using groupby with sort=False since sorted_df is already sorted by user_col
    for _, group in sorted_df.groupby(user_col, sort=False):
        n = len(group)
        n_train, n_val, n_test = compute_split_sizes(n, train_ratio, val_ratio, test_ratio)
        idxs = group.index.values

        train_indices.extend(idxs[:n_train])
        val_indices.extend(idxs[n_train : n_train + n_val])
        test_indices.extend(idxs[n_train + n_val :])

    train_df = sorted_df.iloc[train_indices].reset_index(drop=True)
    val_df = sorted_df.iloc[val_indices].reset_index(drop=True)
    test_df = sorted_df.iloc[test_indices].reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def verify_no_temporal_leakage(
    splits: Dict[str, pd.DataFrame],
    user_col: str = "user_id",
    timestamp_col: str = "timestamp",
) -> bool:
    """Verify that no user has interactions in train occurring after validation or test.

    Returns True if valid, raises AssertionError if temporal leakage detected.
    """
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    # Check max train timestamp <= min val timestamp
    if not val.empty:
        max_train = train.groupby(user_col)[timestamp_col].max()
        min_val = val.groupby(user_col)[timestamp_col].min()
        common_users_val = max_train.index.intersection(min_val.index)
        leakage_val = max_train.loc[common_users_val] > min_val.loc[common_users_val]
        if leakage_val.any():
            bad_users = leakage_val[leakage_val].index.tolist()
            raise AssertionError(f"Temporal leakage detected between train and val for users: {bad_users[:5]}")

    # Check max val timestamp <= min test timestamp
    if not val.empty and not test.empty:
        max_val = val.groupby(user_col)[timestamp_col].max()
        min_test = test.groupby(user_col)[timestamp_col].min()
        common_users_test = max_val.index.intersection(min_test.index)
        leakage_test = max_val.loc[common_users_test] > min_test.loc[common_users_test]
        if leakage_test.any():
            bad_users = leakage_test[leakage_test].index.tolist()
            raise AssertionError(f"Temporal leakage detected between val and test for users: {bad_users[:5]}")

    # Check max train timestamp <= min test timestamp (for any common users)
    if not test.empty:
        max_train = train.groupby(user_col)[timestamp_col].max()
        min_test = test.groupby(user_col)[timestamp_col].min()
        common_users = max_train.index.intersection(min_test.index)
        leakage_tt = max_train.loc[common_users] > min_test.loc[common_users]
        if leakage_tt.any():
            bad_users = leakage_tt[leakage_tt].index.tolist()
            raise AssertionError(f"Temporal leakage detected between train and test for users: {bad_users[:5]}")

    return True
