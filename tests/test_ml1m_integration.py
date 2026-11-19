"""Integration test for MovieLens-1M real dataset when downloaded."""

from pathlib import Path
import pytest
import pandas as pd
from src.data.loader import load_ml1m_ratings, load_ml1m_movies
from src.data.id_mapper import IDMapper
from src.data.splitter import split_chronological_per_user, verify_no_temporal_leakage


ML1M_DIR = Path("data/raw/ml-1m")
RATINGS_PATH = ML1M_DIR / "ratings.dat"
MOVIES_PATH = ML1M_DIR / "movies.dat"


@pytest.mark.skipif(not RATINGS_PATH.is_file(), reason="MovieLens-1M ratings.dat not present locally")
def test_real_ml1m_ratings_integrity():
    ratings = load_ml1m_ratings(RATINGS_PATH, positive_threshold=4.0)

    # MovieLens-1M benchmark constants
    assert len(ratings) == 1000209
    assert ratings["user_id"].nunique() == 6040
    assert ratings["movie_id"].nunique() == 3706
    assert set(ratings["rating"].unique()) == {1.0, 2.0, 3.0, 4.0, 5.0}
    assert ratings["positive"].sum() == 575281  # All ratings >= 4 in ML-1M


@pytest.mark.skipif(not MOVIES_PATH.is_file(), reason="MovieLens-1M movies.dat not present locally")
def test_real_ml1m_movies_catalog():
    movies = load_ml1m_movies(MOVIES_PATH)

    assert len(movies) == 3883
    assert movies["movie_id"].nunique() == 3883
    assert "genre_list" in movies.columns


@pytest.mark.skipif(not RATINGS_PATH.is_file(), reason="MovieLens-1M ratings.dat not present locally")
def test_real_ml1m_splits_leakage():
    # Test chronological splitting on a subset of 50 users from real data
    ratings = load_ml1m_ratings(RATINGS_PATH, positive_threshold=4.0)
    subset_users = ratings["user_id"].unique()[:50]
    subset_ratings = ratings[ratings["user_id"].isin(subset_users)].copy()

    mapper = IDMapper().fit(subset_ratings)
    indexed = mapper.transform(subset_ratings)

    splits = split_chronological_per_user(indexed)
    assert verify_no_temporal_leakage(splits) is True

    # Check that each user has all three splits non-empty (since all users have >= 20 ratings in ML-1M)
    for uid in subset_users:
        assert (splits["train"]["user_id"] == uid).sum() >= 16
        assert (splits["val"]["user_id"] == uid).sum() >= 2
        assert (splits["test"]["user_id"] == uid).sum() >= 2
