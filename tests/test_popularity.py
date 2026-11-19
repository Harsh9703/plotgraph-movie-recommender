"""Unit tests for Popularity candidate recommendation baseline."""

import pandas as pd
import pytest
from src.candidate_gen.popularity import PopularityRecommender


@pytest.fixture
def sample_train_df():
    # 3 users, 5 items (0, 1, 2, 3, 4)
    # Item 0 has 2 positive ratings
    # Item 1 has 3 positive ratings
    # Item 2 has 1 positive rating, 2 non-positive ratings
    # Item 3 has 0 positive ratings, 3 non-positive ratings
    # Item 4 has 1 positive rating
    return pd.DataFrame({
        "user_idx": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
        "movie_idx": [1, 2, 3, 1, 0, 3, 1, 0, 4, 3],
        "rating": [5.0, 3.0, 2.0, 4.0, 5.0, 1.0, 4.5, 4.0, 4.0, 2.5],
        "positive": [1, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    })


def test_popularity_ranks_by_positive_interactions_only(sample_train_df):
    pop = PopularityRecommender()
    pop.fit(sample_train_df, num_users=3, num_items=5)

    # Positive counts:
    # Item 1: 3 positives (users 0, 1, 2)
    # Item 0: 2 positives (users 1, 2)
    # Item 4: 1 positive  (user 2)
    # Item 2: 0 positives (the rating was 3.0, positive=0)
    # Item 3: 0 positives (all ratings < 4.0, positive=0)
    # Deterministic tie breaking by movie_idx ascending:
    # Items with 1 positive: [4]
    # Items with 0 positives: [2, 3]
    # Expected global ranking: [1, 0, 4, 2, 3]
    expected_ranking = [1, 0, 4, 2, 3]
    assert list(pop.ranked_items) == expected_ranking
    assert pop.item_counts[1] == 3
    assert pop.item_counts[0] == 2
    assert pop.item_counts[4] == 1
    assert pop.item_counts[2] == 0
    assert pop.item_counts[3] == 0


def test_popularity_excludes_seen_positive_items(sample_train_df):
    pop = PopularityRecommender().fit(sample_train_df, num_users=3, num_items=5)

    # User 0 saw item 1 positively.
    # Exclude seen -> [0, 4, 2]
    recs_u0 = pop.recommend(user_idx=0, k=3, exclude_seen=True)
    assert recs_u0 == [0, 4, 2]
    assert 1 not in recs_u0

    # User 1 saw item 1 and item 0 positively.
    # Exclude seen -> [4, 2, 3]
    recs_u1 = pop.recommend(user_idx=1, k=3, exclude_seen=True)
    assert recs_u1 == [4, 2, 3]
    assert 1 not in recs_u1
    assert 0 not in recs_u1

    # Include seen -> should return global ranking
    recs_u0_all = pop.recommend(user_idx=0, k=3, exclude_seen=False)
    assert recs_u0_all == [1, 0, 4]


def test_popularity_deterministic_tie_breaking():
    # 4 items all with exactly 0 positive ratings
    df = pd.DataFrame({
        "user_idx": [0, 1],
        "movie_idx": [3, 1],
        "rating": [2.0, 3.0],
        "positive": [0, 0],
    })
    pop = PopularityRecommender().fit(df, num_users=2, num_items=4)
    # Tied at 0 counts: tie breaking should sort by movie_idx ascending [0, 1, 2, 3]
    assert list(pop.ranked_items) == [0, 1, 2, 3]


def test_popularity_recommend_all(sample_train_df):
    pop = PopularityRecommender().fit(sample_train_df, num_users=3, num_items=5)
    all_recs = pop.recommend_all([0, 1], k=2, exclude_seen=True)

    assert 0 in all_recs
    assert 1 in all_recs
    assert all_recs[0] == pop.recommend(0, k=2, exclude_seen=True)
    assert all_recs[1] == pop.recommend(1, k=2, exclude_seen=True)


def test_popularity_raises_if_unfitted():
    pop = PopularityRecommender()
    with pytest.raises(RuntimeError):
        pop.recommend(0)
