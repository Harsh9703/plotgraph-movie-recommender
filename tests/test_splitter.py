"""Unit tests for chronological interaction splitter."""

import numpy as np
import pandas as pd
import pytest
from src.data.splitter import (
    compute_split_sizes,
    split_chronological_per_user,
    verify_no_temporal_leakage,
)


def test_compute_split_sizes_conservation():
    for n in range(1, 150):
        n_train, n_val, n_test = compute_split_sizes(n, 0.8, 0.1, 0.1)
        assert n_train + n_val + n_test == n
        assert n_train >= 0
        assert n_val >= 0
        assert n_test >= 0
        if n >= 3:
            assert n_train >= 1
            assert n_val >= 1
            assert n_test >= 1


def test_compute_split_sizes_specific_cases():
    assert compute_split_sizes(20) == (16, 2, 2)
    assert compute_split_sizes(100) == (80, 10, 10)
    assert compute_split_sizes(1) == (1, 0, 0)
    assert compute_split_sizes(2) == (1, 0, 1)
    assert compute_split_sizes(3) == (1, 1, 1)
    assert compute_split_sizes(5) == (3, 1, 1)
    assert compute_split_sizes(10) == (8, 1, 1)


@pytest.fixture
def synthetic_interactions():
    # 3 users:
    # User 1 has 20 interactions (like ML-1M minimum)
    # User 2 has 30 interactions
    # User 3 has 5 interactions
    rows = []
    np.random.seed(42)

    for uid, count in [(1, 20), (2, 30), (3, 5)]:
        base_time = uid * 1000
        for i in range(count):
            rows.append({
                "user_id": uid,
                "movie_id": 100 + i,
                "rating": float(np.random.choice([1, 2, 3, 4, 5])),
                "timestamp": base_time + i * 10,
                "positive": 1 if np.random.rand() > 0.4 else 0,
            })

    return pd.DataFrame(rows)


def test_chronological_splitting_proportions_and_no_leakage(synthetic_interactions):
    splits = split_chronological_per_user(
        synthetic_interactions,
        user_col="user_id",
        timestamp_col="timestamp",
        item_col="movie_id",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    # Total conservation
    assert len(train) + len(val) + len(test) == len(synthetic_interactions)

    # Check user 1 (20 interactions -> 16 train, 2 val, 2 test)
    assert len(train[train["user_id"] == 1]) == 16
    assert len(val[val["user_id"] == 1]) == 2
    assert len(test[test["user_id"] == 1]) == 2

    # Verify no temporal leakage
    assert verify_no_temporal_leakage(splits) is True

    # Check per-user timestamp ordering strictly
    for uid in [1, 2, 3]:
        u_train = train[train["user_id"] == uid]
        u_val = val[val["user_id"] == uid]
        u_test = test[test["user_id"] == uid]

        if not u_val.empty:
            assert u_train["timestamp"].max() <= u_val["timestamp"].min()
        if not u_val.empty and not u_test.empty:
            assert u_val["timestamp"].max() <= u_test["timestamp"].min()


def test_leakage_detection_catches_tampered_timestamp(synthetic_interactions):
    splits = split_chronological_per_user(synthetic_interactions)

    # Tamper with train split: inject a future timestamp for user 1 into train
    tampered_train = splits["train"].copy()
    val_min_time = splits["val"].loc[splits["val"]["user_id"] == 1, "timestamp"].min()
    tampered_train.loc[tampered_train["user_id"] == 1, "timestamp"] = val_min_time + 9999

    tampered_splits = {
        "train": tampered_train,
        "val": splits["val"],
        "test": splits["test"],
    }

    with pytest.raises(AssertionError, match="Temporal leakage detected"):
        verify_no_temporal_leakage(tampered_splits)


def test_split_determinism(synthetic_interactions):
    # Shuffle input
    shuffled1 = synthetic_interactions.sample(frac=1.0, random_state=123).reset_index(drop=True)
    shuffled2 = synthetic_interactions.sample(frac=1.0, random_state=999).reset_index(drop=True)

    splits1 = split_chronological_per_user(shuffled1)
    splits2 = split_chronological_per_user(shuffled2)

    pd.testing.assert_frame_equal(splits1["train"], splits2["train"])
    pd.testing.assert_frame_equal(splits1["val"], splits2["val"])
    pd.testing.assert_frame_equal(splits1["test"], splits2["test"])
