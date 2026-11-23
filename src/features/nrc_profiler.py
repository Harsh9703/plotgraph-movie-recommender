"""Leakage-Safe Recency-Weighted NRC Emotion User Profiler (Phase 4C).

Constructs 8-dimensional NRC emotion user profiles strictly from positive TRAIN interactions:
- Enforces rating >= positive_threshold (positive == 1).
- Strictly rejects any VAL or TEST interactions or columns.
- Uses identical exponential half-life recency decay: 0.5 ** (age_days / half_life_days).
- Computes normalized recency-weighted average across positive movies with valid NRC vectors.
- Produces normalized 8D distributions summing to 1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import DEFAULT_HALF_LIFE_DAYS, DEFAULT_POSITIVE_THRESHOLD, PLUTCHIK_EMOTIONS
from .user_profiler import compute_age_days, compute_recency_weight

logger = logging.getLogger(__name__)


@dataclass
class NRCUserProfile:
    """NRC emotion profile for a single user."""

    user_idx: int
    nrc_profile: Optional[np.ndarray] = None  # shape (8,), sums to 1.0
    nrc_available: bool = False
    positive_train_count: int = 0
    nrc_movie_count: int = 0


class NRCUserProfileIndex:
    """Container for NRC user profiles across the catalog."""

    def __init__(
        self,
        profiles: Dict[int, NRCUserProfile],
        total_users: int = 6040,
    ):
        self.profiles = profiles
        self.total_users = total_users

        # Pre-allocate dense matrix (total_users, 8) for fast vectorized lookups
        self.user_matrix = np.full((total_users, 8), np.nan, dtype=np.float32)
        self.user_available_mask = np.zeros(total_users, dtype=bool)

        for u_idx, p in profiles.items():
            if p.nrc_available and p.nrc_profile is not None:
                self.user_matrix[u_idx] = p.nrc_profile
                self.user_available_mask[u_idx] = True

    def get(self, user_idx: int) -> Optional[NRCUserProfile]:
        return self.profiles.get(user_idx)

    def __len__(self) -> int:
        return len(self.profiles)


def build_nrc_user_profiles(
    train_df: pd.DataFrame,
    movie_nrc_df: pd.DataFrame,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    total_users: int = 6040,
) -> NRCUserProfileIndex:
    """Construct leakage-safe NRC emotion user profiles strictly from TRAIN split.

    Args:
        train_df: TRAIN interactions DataFrame.
        movie_nrc_df: DataFrame containing movie NRC vectors and availability flags.
        half_life_days: Exponential decay half-life in days (default: 365.0).
        positive_threshold: Minimum rating for positive interaction (default: 4.0).
        total_users: Total number of catalog users (default: 6040).

    Returns:
        NRCUserProfileIndex containing profiles for all catalog users.
    """
    # -------------------------------------------------------------------------
    # Leakage Guard Assertions
    # -------------------------------------------------------------------------
    col_names_lower = [str(c).lower() for c in train_df.columns]
    for forbidden in ["val", "test", "is_val", "is_test", "val_positive", "test_positive"]:
        if forbidden in col_names_lower:
            raise ValueError(
                f"Data leakage detected: train_df contains forbidden column '{forbidden}'. "
                "NRC user profiles MUST be constructed strictly from the TRAIN split."
            )

    # Filter strictly to positive interactions
    if "positive" in train_df.columns:
        pos_df = train_df[train_df["positive"] == 1].copy()
    else:
        pos_df = train_df[train_df["rating"] >= positive_threshold].copy()

    logger.info(
        f"Building NRC user profiles from {len(pos_df):,} positive TRAIN interactions "
        f"across {pos_df['user_idx'].nunique():,} active users..."
    )

    # Pre-extract movie NRC vectors into indexed lookup table: movie_idx -> (8D vec or None)
    nrc_cols = [f"candidate_nrc_{e}" for e in PLUTCHIK_EMOTIONS]
    movie_nrc_map: Dict[int, np.ndarray] = {}
    for _, row in movie_nrc_df.iterrows():
        m_idx = int(row["movie_idx"])
        is_avail = bool(row["candidate_nrc_available"])
        if is_avail:
            vec = row[nrc_cols].to_numpy(dtype=np.float32)
            if not np.isnan(vec).any():
                movie_nrc_map[m_idx] = vec

    # Group positive history by user_idx: list of (movie_idx, timestamp)
    user_histories: Dict[int, List[Tuple[int, int]]] = {u: [] for u in range(total_users)}
    for row in pos_df[["user_idx", "movie_idx", "timestamp"]].itertuples(index=False):
        u_idx = int(row.user_idx)
        m_idx = int(row.movie_idx)
        t_val = int(row.timestamp)
        user_histories[u_idx].append((m_idx, t_val))

    profiles: Dict[int, NRCUserProfile] = {}

    for u_idx in range(total_users):
        history = user_histories.get(u_idx, [])
        pos_count = len(history)

        if pos_count == 0:
            profiles[u_idx] = NRCUserProfile(
                user_idx=u_idx,
                nrc_profile=None,
                nrc_available=False,
                positive_train_count=0,
                nrc_movie_count=0,
            )
            continue

        max_ts = max(t for _, t in history)

        # 1. Compute recency weights
        raw_weights = [
            compute_recency_weight(compute_age_days(t_val, max_ts), half_life_days=half_life_days)
            for _, t_val in history
        ]

        # 2. Collect valid NRC movie vectors
        nrc_vectors: List[np.ndarray] = []
        nrc_weights: List[float] = []

        for (m_idx, _), w_val in zip(history, raw_weights):
            vec = movie_nrc_map.get(m_idx)
            if vec is not None:
                nrc_vectors.append(vec)
                nrc_weights.append(w_val)

        if len(nrc_vectors) > 0 and sum(nrc_weights) > 0:
            sum_w = sum(nrc_weights)
            weighted_sum = np.zeros(8, dtype=np.float32)
            for w, v in zip(nrc_weights, nrc_vectors):
                weighted_sum += (w * v).astype(np.float32)

            avg_vec = weighted_sum / float(sum_w)
            # Re-normalize to ensure exact distribution sum = 1.0
            v_sum = float(avg_vec.sum())
            if v_sum > 0:
                avg_vec = avg_vec / v_sum

            profiles[u_idx] = NRCUserProfile(
                user_idx=u_idx,
                nrc_profile=avg_vec,
                nrc_available=True,
                positive_train_count=pos_count,
                nrc_movie_count=len(nrc_vectors),
            )
        else:
            profiles[u_idx] = NRCUserProfile(
                user_idx=u_idx,
                nrc_profile=None,
                nrc_available=False,
                positive_train_count=pos_count,
                nrc_movie_count=0,
            )

    profile_index = NRCUserProfileIndex(profiles, total_users=total_users)
    active_profiles = sum(1 for p in profiles.values() if p.nrc_available)
    logger.info(
        f"Built NRC user profiles for {total_users:,} users: "
        f"{active_profiles:,} ({active_profiles / total_users * 100:.2f}%) have valid NRC profiles."
    )
    return profile_index
