"""Unit tests for Implicit ALS candidate recommendation model."""

import pandas as pd
import pytest
from src.candidate_gen.als import ALSRecommender


@pytest.fixture
def small_train_df():
    # 3 users, 5 items
    # User 0 likes items 0, 1 (positive=1) and rated item 2 low (positive=0)
    # User 1 likes items 1, 2
    # User 2 likes items 0, 1, 3
    return pd.DataFrame({
        "user_idx": [0, 0, 0, 1, 1, 2, 2, 2],
        "movie_idx": [0, 1, 2, 1, 2, 0, 1, 3],
        "rating": [5.0, 4.0, 2.0, 4.5, 4.0, 5.0, 4.0, 4.0],
        "positive": [1, 1, 0, 1, 1, 1, 1, 1],
    })


def test_als_fits_and_recommends(small_train_df):
    als = ALSRecommender(factors=4, regularization=0.01, iterations=5, random_state=42)
    als.fit(small_train_df, num_users=3, num_items=5)

    assert als.is_fitted
    assert als.model is not None
    assert als.num_users == 3
    assert als.num_items == 5

    # Recommend for user 0
    recs = als.recommend(user_idx=0, k=2, exclude_seen=True)
    assert len(recs) <= 2
    # Seen positive items for user 0 are {0, 1}. Item 2 had positive=0, so it was not seen positive.
    assert 0 not in recs
    assert 1 not in recs


def test_als_recommend_with_scores(small_train_df):
    als = ALSRecommender(factors=4, regularization=0.01, iterations=5, random_state=42)
    als.fit(small_train_df, num_users=3, num_items=5)

    scored_recs = als.recommend_with_scores(user_idx=1, k=2, exclude_seen=True)
    assert len(scored_recs) <= 2
    for item_id, score in scored_recs:
        assert isinstance(item_id, int)
        assert isinstance(score, float)
        assert item_id not in {1, 2}  # User 1's seen positive items


def test_als_recommend_all_consistency(small_train_df):
    als = ALSRecommender(factors=4, regularization=0.01, iterations=5, random_state=42)
    als.fit(small_train_df, num_users=3, num_items=5)

    batch_recs = als.recommend_all([0, 1, 2], k=2, exclude_seen=True)
    for u in [0, 1, 2]:
        single = als.recommend(u, k=2, exclude_seen=True)
        assert batch_recs[u] == single


def test_als_unfitted_raises():
    als = ALSRecommender()
    with pytest.raises(RuntimeError):
        als.recommend(0)
