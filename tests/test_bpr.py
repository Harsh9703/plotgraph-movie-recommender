"""Unit tests for Bayesian Personalized Ranking (BPR) candidate recommendation model."""

import pandas as pd
import pytest
from src.candidate_gen.bpr import BPRRecommender


@pytest.fixture
def small_train_df():
    return pd.DataFrame({
        "user_idx": [0, 0, 0, 1, 1, 2, 2, 2],
        "movie_idx": [0, 1, 2, 1, 2, 0, 1, 3],
        "rating": [5.0, 4.0, 2.0, 4.5, 4.0, 5.0, 4.0, 4.0],
        "positive": [1, 1, 0, 1, 1, 1, 1, 1],
    })


def test_bpr_fits_and_recommends(small_train_df):
    bpr = BPRRecommender(factors=4, regularization=0.01, iterations=10, random_state=42)
    bpr.fit(small_train_df, num_users=3, num_items=5)

    assert bpr.is_fitted
    assert bpr.model is not None

    recs = bpr.recommend(user_idx=0, k=2, exclude_seen=True)
    assert len(recs) <= 2
    # User 0 saw items 0 and 1 positively
    assert 0 not in recs
    assert 1 not in recs


def test_bpr_recommend_with_scores(small_train_df):
    bpr = BPRRecommender(factors=4, regularization=0.01, iterations=10, random_state=42)
    bpr.fit(small_train_df, num_users=3, num_items=5)

    scored = bpr.recommend_with_scores(user_idx=1, k=2, exclude_seen=True)
    assert len(scored) <= 2
    for item_id, score in scored:
        assert isinstance(item_id, int)
        assert isinstance(score, float)
        assert item_id not in {1, 2}


def test_bpr_deterministic_behavior(small_train_df):
    bpr1 = BPRRecommender(factors=4, regularization=0.01, iterations=10, random_state=42)
    bpr1.fit(small_train_df, num_users=3, num_items=5)

    bpr2 = BPRRecommender(factors=4, regularization=0.01, iterations=10, random_state=42)
    bpr2.fit(small_train_df, num_users=3, num_items=5)

    assert bpr1.recommend(0, k=3) == bpr2.recommend(0, k=3)
    assert bpr1.recommend(1, k=3) == bpr2.recommend(1, k=3)


def test_bpr_unfitted_raises():
    bpr = BPRRecommender()
    with pytest.raises(RuntimeError):
        bpr.recommend(0)
