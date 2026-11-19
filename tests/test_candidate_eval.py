"""Unit tests for candidate evaluation harness."""

import pandas as pd
import pytest
from src.candidate_gen.evaluator import evaluate_candidate_model
from src.candidate_gen.popularity import PopularityRecommender


@pytest.fixture
def sample_splits():
    # 2 users: user 0, user 1
    # Train: user 0 likes item 0; user 1 likes item 1
    train_df = pd.DataFrame({
        "user_idx": [0, 1],
        "movie_idx": [0, 1],
        "rating": [5.0, 5.0],
        "positive": [1, 1],
    })

    # Val:
    # User 0 has positive on item 2, and non-positive on item 3
    # User 1 has no positive (only non-positive on item 4)
    val_df = pd.DataFrame({
        "user_idx": [0, 0, 1],
        "movie_idx": [2, 3, 4],
        "rating": [4.0, 2.0, 3.0],
        "positive": [1, 0, 0],
    })

    return train_df, val_df


def test_evaluator_uses_only_validation_positives(sample_splits):
    train_df, val_df = sample_splits

    pop = PopularityRecommender()
    pop.fit(train_df, num_users=2, num_items=5)

    eval_result = evaluate_candidate_model(pop, val_df, cutoffs=[2, 5], exclude_seen=True)

    # User 0 has 1 positive validation item (item 2). Item 3 (positive=0) must NOT be counted as relevant.
    # User 1 has 0 positive validation items.
    assert eval_result["total_val_users"] == 2
    assert eval_result["eval_val_users"] == 1
    assert eval_result["eval_coverage_pct"] == 50.0

    metrics = eval_result["metrics"]
    assert "ndcg@2" in metrics
    assert "recall@2" in metrics
    assert "hr@2" in metrics
    assert "mrr@2" in metrics


def test_evaluator_excludes_seen_training_items(sample_splits):
    train_df, val_df = sample_splits

    pop = PopularityRecommender().fit(train_df, num_users=2, num_items=5)

    # For user 0, seen in train is {0}. Popularity ranking is [0, 1, 2, 3, 4].
    # Exclude seen -> [1, 2, 3, 4].
    eval_result = evaluate_candidate_model(pop, val_df, cutoffs=[2], exclude_seen=True)

    # Ground truth for user 0 is {2}.
    # Recommendations for user 0 are [1, 2] (length 2).
    # Item 2 is at rank 2!
    # HR@2 = 1.0, Recall@2 = 1.0, MRR@2 = 1/2 = 0.5
    m = eval_result["metrics"]
    assert m["hr@2"] == 1.0
    assert m["recall@2"] == 1.0
    assert m["mrr@2"] == 0.5


def test_evaluator_missing_positive_column_raises():
    train_df = pd.DataFrame({"user_idx": [0], "movie_idx": [0], "positive": [1]})
    bad_val_df = pd.DataFrame({"user_idx": [0], "movie_idx": [1]})  # missing positive

    pop = PopularityRecommender().fit(train_df, num_users=1, num_items=2)
    with pytest.raises(ValueError, match="'positive' column"):
        evaluate_candidate_model(pop, bad_val_df)
