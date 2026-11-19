"""Unit tests for ItemKNN candidate recommendation model."""

import pandas as pd
import pytest
from src.candidate_gen.item_knn import ItemKNNRecommender


@pytest.fixture
def cluster_interactions_df():
    # Construct 2 distinct movie clusters:
    # Cluster A: items 0 and 1 (liked by users 0, 1, 2)
    # Cluster B: items 2 and 3 (liked by users 3, 4, 5)
    # Test users:
    # User 6 likes item 0
    # User 7 likes item 2
    rows = []
    # Cluster A
    for u in [0, 1, 2]:
        rows.append({"user_idx": u, "movie_idx": 0, "rating": 5.0, "positive": 1})
        rows.append({"user_idx": u, "movie_idx": 1, "rating": 4.5, "positive": 1})

    # Cluster B
    for u in [3, 4, 5]:
        rows.append({"user_idx": u, "movie_idx": 2, "rating": 5.0, "positive": 1})
        rows.append({"user_idx": u, "movie_idx": 3, "rating": 4.0, "positive": 1})

    # Non-positive distraction rating (should be ignored!)
    rows.append({"user_idx": 0, "movie_idx": 2, "rating": 2.0, "positive": 0})

    # Test users
    rows.append({"user_idx": 6, "movie_idx": 0, "rating": 5.0, "positive": 1})
    rows.append({"user_idx": 7, "movie_idx": 2, "rating": 5.0, "positive": 1})

    return pd.DataFrame(rows)


def test_item_knn_personalization_and_clustering(cluster_interactions_df):
    knn = ItemKNNRecommender(k_neighbors=2)
    knn.fit(cluster_interactions_df, num_users=8, num_items=4)

    # User 6 liked item 0 -> should recommend item 1 (cluster A neighbor)
    recs_u6 = knn.recommend(user_idx=6, k=2, exclude_seen=True)
    assert recs_u6[0] == 1  # Item 1 must be top-1 recommended
    assert 0 not in recs_u6  # Item 0 was seen, must be excluded

    # User 7 liked item 2 -> should recommend item 3 (cluster B neighbor)
    recs_u7 = knn.recommend(user_idx=7, k=2, exclude_seen=True)
    assert recs_u7[0] == 3  # Item 3 must be top-1 recommended
    assert 2 not in recs_u7  # Item 2 was seen, must be excluded


def test_item_knn_ignores_non_positive_ratings(cluster_interactions_df):
    knn = ItemKNNRecommender(k_neighbors=2)
    knn.fit(cluster_interactions_df, num_users=8, num_items=4)

    # In cluster_interactions_df, user 0 had rating 2.0 (positive=0) for item 2.
    # If non-positives were used, item 0 and item 2 would co-occur.
    # Check similarity between item 0 and item 2 is 0.0
    W = knn.W_sparse.toarray()
    # Item 0 row in W should have 0 similarity to item 2
    assert W[0, 2] == 0.0
    assert W[2, 0] == 0.0


def test_item_knn_recommend_all_consistency(cluster_interactions_df):
    knn = ItemKNNRecommender(k_neighbors=2)
    knn.fit(cluster_interactions_df, num_users=8, num_items=4)

    batched = knn.recommend_all([6, 7], k=2, exclude_seen=True)
    single_u6 = knn.recommend(6, k=2, exclude_seen=True)
    single_u7 = knn.recommend(7, k=2, exclude_seen=True)

    assert batched[6] == single_u6
    assert batched[7] == single_u7


def test_item_knn_deterministic_output(cluster_interactions_df):
    knn1 = ItemKNNRecommender(k_neighbors=2).fit(cluster_interactions_df)
    knn2 = ItemKNNRecommender(k_neighbors=2).fit(cluster_interactions_df)

    assert knn1.recommend(6, k=3) == knn2.recommend(6, k=3)
    assert knn1.recommend(7, k=3) == knn2.recommend(7, k=3)


def test_item_knn_raises_if_unfitted():
    knn = ItemKNNRecommender()
    with pytest.raises(RuntimeError):
        knn.recommend(0)
