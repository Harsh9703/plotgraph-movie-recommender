"""Unit tests for IDMapper."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from src.data.id_mapper import IDMapper


@pytest.fixture
def sample_ratings_df():
    return pd.DataFrame({
        "user_id": [500, 12, 500, 3, 12],
        "movie_id": [9999, 42, 101, 42, 9999],
        "rating": [5.0, 4.0, 3.0, 4.5, 2.0],
        "timestamp": [100, 101, 102, 103, 104],
    })


def test_consecutive_zero_indexed_ids(sample_ratings_df):
    mapper = IDMapper()
    mapper.fit(sample_ratings_df)

    # 3 unique users: 3, 12, 500
    assert mapper.num_users == 3
    assert set(mapper.user2idx.values()) == {0, 1, 2}
    assert mapper.user2idx[3] == 0
    assert mapper.user2idx[12] == 1
    assert mapper.user2idx[500] == 2

    # 3 unique movies: 42, 101, 9999
    assert mapper.num_movies == 3
    assert set(mapper.movie2idx.values()) == {0, 1, 2}
    assert mapper.movie2idx[42] == 0
    assert mapper.movie2idx[101] == 1
    assert mapper.movie2idx[9999] == 2


def test_transform_preserves_originals(sample_ratings_df):
    mapper = IDMapper().fit(sample_ratings_df)
    transformed = mapper.transform(sample_ratings_df)

    assert "user_id" in transformed.columns
    assert "movie_id" in transformed.columns
    assert "user_idx" in transformed.columns
    assert "movie_idx" in transformed.columns

    # Verify matching
    for _, row in transformed.iterrows():
        assert mapper.user2idx[row["user_id"]] == row["user_idx"]
        assert mapper.movie2idx[row["movie_id"]] == row["movie_idx"]
        assert mapper.idx2user[row["user_idx"]] == row["user_id"]
        assert mapper.idx2movie[row["movie_idx"]] == row["movie_id"]


def test_determinism():
    df1 = pd.DataFrame({"user_id": [10, 20, 30], "movie_id": [1, 2, 3]})
    df2 = pd.DataFrame({"user_id": [30, 10, 20], "movie_id": [3, 1, 2]})

    mapper1 = IDMapper().fit(df1)
    mapper2 = IDMapper().fit(df2)

    assert mapper1.user2idx == mapper2.user2idx
    assert mapper1.movie2idx == mapper2.movie2idx


def test_unknown_id_raises(sample_ratings_df):
    mapper = IDMapper().fit(sample_ratings_df)
    unknown_df = pd.DataFrame({"user_id": [99999], "movie_id": [42]})
    with pytest.raises(KeyError):
        mapper.transform(unknown_df)


def test_save_and_load(sample_ratings_df):
    mapper = IDMapper().fit(sample_ratings_df)
    with tempfile.TemporaryDirectory() as tmpdir:
        mapper.save(tmpdir)
        loaded = IDMapper.load(tmpdir)

        assert loaded.user2idx == mapper.user2idx
        assert loaded.movie2idx == mapper.movie2idx
        assert loaded.idx2user == mapper.idx2user
        assert loaded.idx2movie == mapper.idx2movie
