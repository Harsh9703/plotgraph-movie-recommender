"""MovieLens-1M dataset loader and parser."""

from pathlib import Path
from typing import Union
import pandas as pd


def load_ml1m_ratings(
    filepath: Union[str, Path],
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """Parse MovieLens-1M ratings.dat file.

    Format: UserID::MovieID::Rating::Timestamp

    Args:
        filepath: Path to ratings.dat.
        positive_threshold: Rating threshold (inclusive) for positive interaction. Default is 4.0.

    Returns:
        DataFrame with columns: ['user_id', 'movie_id', 'rating', 'timestamp', 'positive'].
        user_id and movie_id preserve the original MovieLens identifiers.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Ratings file not found at {path.resolve()}")

    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "float32",
            "timestamp": "int64",
        },
    )

    # Add binary positive interaction indicator (r >= threshold)
    df["positive"] = (df["rating"] >= positive_threshold).astype("int8")

    return df


def load_ml1m_movies(
    filepath: Union[str, Path],
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """Parse MovieLens-1M movies.dat file.

    Format: MovieID::Title::Genres

    Args:
        filepath: Path to movies.dat.
        encoding: Text encoding (defaults to 'latin-1' as required by ML-1M).

    Returns:
        DataFrame with columns: ['movie_id', 'title', 'genres', 'genre_list'].
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Movies file not found at {path.resolve()}")

    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        dtype={
            "movie_id": "int64",
            "title": "string",
            "genres": "string",
        },
        encoding=encoding,
    )

    df["genre_list"] = df["genres"].str.split("|")

    return df
