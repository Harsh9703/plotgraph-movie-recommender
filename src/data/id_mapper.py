"""Bidirectional mapping between original MovieLens IDs and consecutive 0-indexed integers."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


class IDMapper:
    """Manages bijective mappings between external (original) IDs and consecutive internal indices [0, N-1]."""

    def __init__(self) -> None:
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.movie2idx: Dict[int, int] = {}
        self.idx2movie: Dict[int, int] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: Optional[pd.DataFrame] = None,
    ) -> "IDMapper":
        """Build mappings deterministically from interaction data and optional catalog.

        IDs are sorted ascending before assigning indexes to guarantee 100% reproducibility.

        Args:
            ratings_df: DataFrame containing 'user_id' and 'movie_id'.
            movies_df: Optional DataFrame containing all catalog 'movie_id's.

        Returns:
            self
        """
        # Unique users
        unique_users = np.sort(ratings_df["user_id"].unique())
        self.user2idx = {int(uid): idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}

        # Unique movies (union of ratings and catalog if provided)
        if movies_df is not None:
            movie_ids = np.union1d(ratings_df["movie_id"].unique(), movies_df["movie_id"].unique())
        else:
            movie_ids = ratings_df["movie_id"].unique()
        unique_movies = np.sort(movie_ids)
        self.movie2idx = {int(mid): idx for idx, mid in enumerate(unique_movies)}
        self.idx2movie = {idx: mid for mid, idx in self.movie2idx.items()}

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'user_idx' and 'movie_idx' columns to the dataframe while preserving original IDs.

        Args:
            df: DataFrame containing 'user_id' and/or 'movie_id'.

        Returns:
            New DataFrame with internal index columns added.
        """
        if not self.is_fitted:
            raise RuntimeError("IDMapper must be fitted before calling transform.")

        df_out = df.copy()

        if "user_id" in df_out.columns:
            df_out["user_idx"] = df_out["user_id"].map(self.user2idx)
            if df_out["user_idx"].isnull().any():
                missing_users = df_out.loc[df_out["user_idx"].isnull(), "user_id"].unique()
                raise KeyError(f"Encountered unknown user IDs: {missing_users[:5]}")
            df_out["user_idx"] = df_out["user_idx"].astype("int64")

        if "movie_id" in df_out.columns:
            df_out["movie_idx"] = df_out["movie_id"].map(self.movie2idx)
            if df_out["movie_idx"].isnull().any():
                missing_movies = df_out.loc[df_out["movie_idx"].isnull(), "movie_id"].unique()
                raise KeyError(f"Encountered unknown movie IDs: {missing_movies[:5]}")
            df_out["movie_idx"] = df_out["movie_idx"].astype("int64")

        return df_out

    @property
    def num_users(self) -> int:
        return len(self.user2idx)

    @property
    def num_movies(self) -> int:
        return len(self.movie2idx)

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Export mappings as DataFrames."""
        if not self.is_fitted:
            raise RuntimeError("IDMapper is not fitted.")

        user_df = pd.DataFrame(
            [{"user_id": uid, "user_idx": idx} for uid, idx in self.user2idx.items()]
        ).sort_values("user_idx").reset_index(drop=True)

        movie_df = pd.DataFrame(
            [{"movie_id": mid, "movie_idx": idx} for mid, idx in self.movie2idx.items()]
        ).sort_values("movie_idx").reset_index(drop=True)

        return {"users": user_df, "movies": movie_df}

    def save(self, dir_path: Union[str, Path]) -> None:
        """Save ID maps to Parquet files."""
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        dfs = self.to_dataframes()
        dfs["users"].to_parquet(p / "user_id_map.parquet", index=False)
        dfs["movies"].to_parquet(p / "movie_id_map.parquet", index=False)

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "IDMapper":
        """Load ID maps from saved Parquet files."""
        p = Path(dir_path)
        user_path = p / "user_id_map.parquet"
        movie_path = p / "movie_id_map.parquet"
        if not user_path.is_file() or not movie_path.is_file():
            raise FileNotFoundError(f"Missing ID mapping files in {p.resolve()}")

        user_df = pd.read_parquet(user_path)
        movie_df = pd.read_parquet(movie_path)

        mapper = cls()
        mapper.user2idx = dict(zip(user_df["user_id"], user_df["user_idx"]))
        mapper.idx2user = dict(zip(user_df["user_idx"], user_df["user_id"]))
        mapper.movie2idx = dict(zip(movie_df["movie_id"], movie_df["movie_idx"]))
        mapper.idx2movie = dict(zip(movie_df["movie_idx"], movie_df["movie_id"]))
        mapper.is_fitted = True
        return mapper
