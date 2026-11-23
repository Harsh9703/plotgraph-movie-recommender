"""Leakage-Safe Recency-Weighted Text User Profiler (Phase 4B).

Constructs TF-IDF and SBERT user profiles strictly from positive TRAIN history:
- Restricts to positive interactions (rating >= positive_threshold / positive == 1).
- Uses the identical Phase 4A exponential recency decay weighting: 0.5 ** (age / half_life).
- Aggregates unit L2-normalized movie representations into normalized user vectors.
- Asserts zero leakage from VAL and TEST splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .constants import DEFAULT_HALF_LIFE_DAYS, DEFAULT_POSITIVE_THRESHOLD
from .text_encoder import SBERTMovieEncoder, TFIDFMovieEncoder
from .user_profiler import compute_age_days, compute_recency_weight

logger = logging.getLogger(__name__)


@dataclass
class TextUserProfile:
    """Text-semantic profiles for a single user."""

    user_idx: int
    tfidf_profile: Optional[sparse.csr_matrix] = None  # shape (1, vocab_size), unit L2-norm
    tfidf_available: bool = False
    sbert_profile: Optional[np.ndarray] = None  # shape (dim,), unit L2-norm
    sbert_available: bool = False


class TextUserProfileIndex:
    """Container for TextUserProfile instances across the user catalog."""

    def __init__(
        self,
        profiles: Dict[int, TextUserProfile],
        sbert_dimension: int = 384,
        vocab_size: int = 0,
        total_users: int = 6040,
    ):
        self.profiles = profiles
        self.sbert_dimension = sbert_dimension
        self.vocab_size = vocab_size
        self.total_users = total_users

        # Pre-allocate dense matrix for SBERT user profiles for fast vectorized batch lookups
        self.sbert_user_matrix = np.full(
            (total_users, sbert_dimension), np.nan, dtype=np.float32
        )
        self.sbert_user_available_mask = np.zeros(total_users, dtype=bool)

        for u_idx, p in profiles.items():
            if p.sbert_available and p.sbert_profile is not None:
                self.sbert_user_matrix[u_idx] = p.sbert_profile
                self.sbert_user_available_mask[u_idx] = True

    def get(self, user_idx: int) -> Optional[TextUserProfile]:
        return self.profiles.get(user_idx)

    def __len__(self) -> int:
        return len(self.profiles)


def build_text_user_profiles(
    train_df: pd.DataFrame,
    tfidf_encoder: TFIDFMovieEncoder,
    sbert_encoder: SBERTMovieEncoder,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    total_users: int = 6040,
) -> TextUserProfileIndex:
    """Construct leakage-safe text user profiles strictly from TRAIN split.

    Args:
        train_df: TRAIN interactions DataFrame.
        tfidf_encoder: Fitted TFIDFMovieEncoder.
        sbert_encoder: SBERTMovieEncoder with precomputed embeddings.
        half_life_days: Exponential decay half-life in days.
        positive_threshold: Minimum rating for positive interactions (default: 4.0).
        total_users: Total number of catalog users (default: 6040).

    Returns:
        TextUserProfileIndex containing TF-IDF and SBERT profiles for all users.
    """
    # -------------------------------------------------------------------------
    # Leakage Guard Assertions
    # -------------------------------------------------------------------------
    col_names_lower = [str(c).lower() for c in train_df.columns]
    for forbidden in ["val", "test", "is_val", "is_test", "val_positive", "test_positive"]:
        if forbidden in col_names_lower:
            raise ValueError(
                f"Data leakage detected: train_df contains forbidden column '{forbidden}'. "
                "Text user profiles MUST be constructed strictly from the TRAIN split."
            )

    if "positive" in train_df.columns:
        pos_df = train_df[train_df["positive"] == 1].copy()
    else:
        pos_df = train_df[train_df["rating"] >= positive_threshold].copy()

    logger.info(
        f"Building text user profiles from {len(pos_df):,} positive TRAIN interactions "
        f"across {pos_df['user_idx'].nunique():,} active users..."
    )

    # Group positive history by user_idx: list of (movie_idx, timestamp)
    user_histories: Dict[int, List[Tuple[int, int]]] = {u: [] for u in range(total_users)}
    for row in pos_df[["user_idx", "movie_idx", "timestamp"]].itertuples(index=False):
        u_idx = int(row.user_idx)
        m_idx = int(row.movie_idx)
        t_val = int(row.timestamp)
        user_histories[u_idx].append((m_idx, t_val))

    profiles: Dict[int, TextUserProfile] = {}
    sbert_dim = sbert_encoder.embedding_dimension
    vocab_size = tfidf_encoder.vocab_size

    for u_idx in range(total_users):
        history = user_histories.get(u_idx, [])
        if len(history) == 0:
            profiles[u_idx] = TextUserProfile(
                user_idx=u_idx,
                tfidf_profile=None,
                tfidf_available=False,
                sbert_profile=None,
                sbert_available=False,
            )
            continue

        max_ts = max(t for _, t in history)

        # 1. Compute recency weights
        raw_weights = [
            compute_recency_weight(compute_age_days(t_val, max_ts), half_life_days=half_life_days)
            for _, t_val in history
        ]

        # ---------------------------------------------------------------------
        # 2. TF-IDF Profile
        # ---------------------------------------------------------------------
        tfidf_vectors: List[sparse.csr_matrix] = []
        tfidf_weights: List[float] = []

        for (m_idx, _), w_val in zip(history, raw_weights):
            vec = tfidf_encoder.get_movie_vector(m_idx)
            if vec is not None:
                tfidf_vectors.append(vec)
                tfidf_weights.append(w_val)

        tfidf_prof = None
        tfidf_avail = False
        if len(tfidf_vectors) > 0 and sum(tfidf_weights) > 0:
            sum_w = sum(tfidf_weights)
            # Weighted average of sparse row vectors
            stacked = sparse.vstack(tfidf_vectors)  # shape (k, vocab_size)
            weights_arr = np.array(tfidf_weights, dtype=np.float32).reshape(-1, 1)
            # weighted sum
            avg_sparse = sparse.csr_matrix(stacked.multiply(weights_arr).sum(axis=0) / sum_w)
            # Unit L2-normalize
            norm = sparse.linalg.norm(avg_sparse)
            if norm > 0:
                avg_sparse = avg_sparse / norm
            tfidf_prof = avg_sparse
            tfidf_avail = True

        # ---------------------------------------------------------------------
        # 3. SBERT Profile
        # ---------------------------------------------------------------------
        sbert_vectors: List[np.ndarray] = []
        sbert_weights: List[float] = []

        for (m_idx, _), w_val in zip(history, raw_weights):
            vec = sbert_encoder.get_movie_embedding(m_idx)
            if vec is not None:
                sbert_vectors.append(vec)
                sbert_weights.append(w_val)

        sbert_prof = None
        sbert_avail = False
        if len(sbert_vectors) > 0 and sum(sbert_weights) > 0:
            sum_w = sum(sbert_weights)
            weighted_sum = np.zeros(sbert_dim, dtype=np.float32)
            for w, v in zip(sbert_weights, sbert_vectors):
                weighted_sum += (w * v).astype(np.float32)
            avg_vec = weighted_sum / float(sum_w)
            norm = float(np.linalg.norm(avg_vec))
            if norm > 0:
                avg_vec = avg_vec / norm
            sbert_prof = avg_vec
            sbert_avail = True

        profiles[u_idx] = TextUserProfile(
            user_idx=u_idx,
            tfidf_profile=tfidf_prof,
            tfidf_available=tfidf_avail,
            sbert_profile=sbert_prof,
            sbert_available=sbert_avail,
        )

    return TextUserProfileIndex(
        profiles=profiles,
        sbert_dimension=sbert_dim,
        vocab_size=vocab_size,
        total_users=total_users,
    )
