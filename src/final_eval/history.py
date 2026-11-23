"""Historical interaction management and leakage guards for Phase 5B final evaluation preparation.

Handles:
- Combining chronological TRAIN and VAL splits into a single historical interaction dataset.
- Enforcing strict zero TEST access assertions.
- Extracting positive interactions (rating >= 4.0 / positive == 1) for user preference profiling.
"""

from pathlib import Path
from typing import Sequence, Set, Union

import pandas as pd

from src.reranker.dataset import assert_no_test_access


def load_train_val_history(
    train_path: Union[str, Path] = "data/processed/ml-1m/splits/train.parquet",
    val_path: Union[str, Path] = "data/processed/ml-1m/splits/val.parquet",
) -> pd.DataFrame:
    """Load and combine chronological TRAIN and VAL splits into historical interactions.

    Enforces strict zero TEST access assertions: neither train_path nor val_path
    may reference test data.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing columns:
        ['user_idx', 'movie_idx', 'rating', 'timestamp', 'positive'].
    """
    assert_no_test_access(str(train_path))
    assert_no_test_access(str(val_path))

    p_train = Path(train_path)
    p_val = Path(val_path)

    if not p_train.exists():
        raise FileNotFoundError(f"TRAIN split not found: {p_train}")
    if not p_val.exists():
        raise FileNotFoundError(f"VAL split not found: {p_val}")

    df_train = pd.read_parquet(p_train)
    df_val = pd.read_parquet(p_val)

    required_cols = ["user_idx", "movie_idx", "rating", "timestamp"]
    for col in required_cols:
        if col not in df_train.columns:
            raise KeyError(f"TRAIN split missing required column '{col}'")
        if col not in df_val.columns:
            raise KeyError(f"VAL split missing required column '{col}'")

    # Combine chronologically
    combined = pd.concat([df_train, df_val], ignore_index=True)

    # Ensure binary 'positive' column exists
    if "positive" not in combined.columns:
        combined["positive"] = (combined["rating"] >= 4.0).astype(int)

    # Integrity verification
    assert len(combined) == len(df_train) + len(df_val), "Row conservation violation in concat!"
    return combined


def get_positive_history_stats(history_df: pd.DataFrame, positive_threshold: float = 4.0) -> dict:
    """Compute summary statistics for historical interactions."""
    if "positive" in history_df.columns:
        pos_df = history_df[history_df["positive"] == 1]
    else:
        pos_df = history_df[history_df["rating"] >= positive_threshold]

    return {
        "total_interactions": int(len(history_df)),
        "total_positive_interactions": int(len(pos_df)),
        "total_users": int(history_df["user_idx"].nunique()),
        "users_with_positive": int(pos_df["user_idx"].nunique()),
        "total_movies": int(history_df["movie_idx"].nunique()),
        "movies_with_positive": int(pos_df["movie_idx"].nunique()),
    }
