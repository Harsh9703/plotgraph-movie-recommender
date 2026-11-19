"""Unit tests for MovieLens-1M file loaders."""

import tempfile
from pathlib import Path
import pytest
from src.data.loader import load_ml1m_ratings, load_ml1m_movies


def test_load_ml1m_ratings():
    content = (
        "1::1193::5::978300760\n"
        "1::661::3::978302109\n"
        "1::914::3::978301968\n"
        "2::3408::4::978298413\n"
    )
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = Path(tmp.name)

    try:
        df = load_ml1m_ratings(tmp_path, positive_threshold=4.0)

        assert len(df) == 4
        assert list(df.columns) == ["user_id", "movie_id", "rating", "timestamp", "positive"]

        # Check positive binarization
        # Ratings: 5 (pos=1), 3 (pos=0), 3 (pos=0), 4 (pos=1)
        assert df["positive"].tolist() == [1, 0, 0, 1]
        assert df["user_id"].dtype == "int64"
        assert df["movie_id"].dtype == "int64"
        assert df["timestamp"].dtype == "int64"
    finally:
        if tmp_path.is_file():
            tmp_path.unlink()


def test_load_ml1m_movies_with_encoding():
    content = (
        "1::Toy Story (1995)::Animation|Children's|Comedy\n"
        "2::Jumanji (1995)::Adventure|Children's|Fantasy\n"
        "3::Amélie (2001)::Comedy|Romance\n"
    )
    with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
        tmp.write(content.encode("latin-1"))
        tmp.flush()
        tmp_path = Path(tmp.name)

    try:
        df = load_ml1m_movies(tmp_path, encoding="latin-1")

        assert len(df) == 3
        assert list(df["movie_id"]) == [1, 2, 3]
        assert "Amélie (2001)" in list(df["title"])
        assert df.loc[df["movie_id"] == 1, "genre_list"].iloc[0] == ["Animation", "Children's", "Comedy"]
    finally:
        if tmp_path.is_file():
            tmp_path.unlink()


def test_loader_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_ml1m_ratings("non_existent_file.dat")

    with pytest.raises(FileNotFoundError):
        load_ml1m_movies("non_existent_file.dat")
