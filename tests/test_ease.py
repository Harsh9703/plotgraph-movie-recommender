"""Unit tests for EASE candidate recommendation model."""

import numpy as np
import pandas as pd
import pytest
from src.candidate_gen.ease import EASERecommender


@pytest.fixture
def synthetic_cluster_df():
    # 2 movie clusters:
    # Cluster A: items 0, 1 (liked by users 0, 1, 2)
    # Cluster B: items 2, 3 (liked by users 3, 4, 5)
    # User 6 likes item 0
    # User 7 likes item 2
    rows = []
    for u in [0, 1, 2]:
        rows.append({"user_idx": u, "movie_idx": 0, "rating": 5.0, "positive": 1})
        rows.append({"user_idx": u, "movie_idx": 1, "rating": 5.0, "positive": 1})
    for u in [3, 4, 5]:
        rows.append({"user_idx": u, "movie_idx": 2, "rating": 5.0, "positive": 1})
        rows.append({"user_idx": u, "movie_idx": 3, "rating": 5.0, "positive": 1})

    # Test users
    rows.append({"user_idx": 6, "movie_idx": 0, "rating": 5.0, "positive": 1})
    rows.append({"user_idx": 7, "movie_idx": 2, "rating": 5.0, "positive": 1})
    return pd.DataFrame(rows)


def test_ease_fit_properties(synthetic_cluster_df):
    ease = EASERecommender(reg=10.0)
    ease.fit(synthetic_cluster_df, num_users=8, num_items=4)

    assert ease.is_fitted
    assert ease.B is not None
    assert ease.B.shape == (4, 4)
    # Exact zero diagonal
    assert np.allclose(np.diag(ease.B), 0.0)


def test_ease_personalization_and_exclusion(synthetic_cluster_df):
    ease = EASERecommender(reg=1.0)
    ease.fit(synthetic_cluster_df, num_users=8, num_items=4)

    # User 6 liked item 0 -> should recommend item 1
    recs_u6 = ease.recommend(user_idx=6, k=2, exclude_seen=True)
    assert recs_u6[0] == 1
    assert 0 not in recs_u6

    # User 7 liked item 2 -> should recommend item 3
    recs_u7 = ease.recommend(user_idx=7, k=2, exclude_seen=True)
    assert recs_u7[0] == 3
    assert 2 not in recs_u7


def test_ease_recommend_with_scores(synthetic_cluster_df):
    ease = EASERecommender(reg=1.0)
    ease.fit(synthetic_cluster_df, num_users=8, num_items=4)

    scored = ease.recommend_with_scores(user_idx=6, k=2, exclude_seen=True)
    assert len(scored) == 2
    item_id, score = scored[0]
    assert item_id == 1
    assert isinstance(score, float)
    assert score > scored[1][1]  # Highest score first


def test_ease_recommend_all_consistency(synthetic_cluster_df):
    ease = EASERecommender(reg=1.0)
    ease.fit(synthetic_cluster_df, num_users=8, num_items=4)

    batched = ease.recommend_all([6, 7], k=2, exclude_seen=True)
    assert batched[6] == ease.recommend(6, k=2, exclude_seen=True)
    assert batched[7] == ease.recommend(7, k=2, exclude_seen=True)


def test_ease_deterministic_output(synthetic_cluster_df):
    ease1 = EASERecommender(reg=10.0).fit(synthetic_cluster_df)
    ease2 = EASERecommender(reg=10.0).fit(synthetic_cluster_df)

    assert np.allclose(ease1.B, ease2.B)
    assert ease1.recommend(6, k=3) == ease2.recommend(6, k=3)


def test_ease_unfitted_raises():
    ease = EASERecommender()
    with pytest.raises(RuntimeError):
        ease.recommend(0)
