"""Candidate-Level NRC Emotion Feature Builder and Validation (Phase 4C).

Appends 23 leakage-safe NRC emotion features to the Phase 4B candidate feature table:
- 4 compatibility features:
  * nrc_emotion_cosine_similarity
  * nrc_emotion_dot_product
  * nrc_emotion_l1_distance
  * nrc_emotion_l2_distance
- 8 candidate NRC dimensions:
  * candidate_nrc_joy ... candidate_nrc_anticipation
- 8 user NRC dimensions:
  * user_nrc_joy ... user_nrc_anticipation
- 3 coverage / support features:
  * candidate_nrc_available
  * user_nrc_profile_available
  * candidate_nrc_matched_token_count

Preserves all 68 Phase 4B columns, exact row count (604,000), alignment, and ordering.
Produces the full candidate feature matrix (91 columns).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import (
    CANDIDATE_NRC_EMOTION_FEATURES,
    FULL_COLUMNS,
    NRC_COVERAGE_FEATURES,
    NRC_EMOTION_COMPATIBILITY_FEATURES,
    NRC_EMOTION_FEATURES,
    PHASE4B_COLUMNS,
    PLUTCHIK_EMOTIONS,
    USER_NRC_EMOTION_FEATURES,
)
from .nrc_profiler import NRCUserProfileIndex

logger = logging.getLogger(__name__)


def build_nrc_candidate_features(
    phase4b_df: pd.DataFrame,
    nrc_user_profiles: NRCUserProfileIndex,
    movie_nrc_df: pd.DataFrame,
    validate: bool = True,
) -> pd.DataFrame:
    """Compute and append 23 NRC emotion features to candidate DataFrame.

    Args:
        phase4b_df: Existing Phase 4B candidate feature DataFrame (604,000 rows).
        nrc_user_profiles: Leakage-safe NRCUserProfileIndex.
        movie_nrc_df: Catalog movie NRC vectors DataFrame (from compute_catalog_vectors).
        validate: Whether to run verification assertions.

    Returns:
        DataFrame with exactly len(phase4b_df) rows and all 91 columns (Phase 4B + 4C).
    """
    n_rows = len(phase4b_df)
    logger.info(f"Appending NRC emotion baseline features to {n_rows:,} candidate rows...")

    user_indices = phase4b_df["user_idx"].to_numpy(dtype=np.int64)
    movie_indices = phase4b_df["movie_idx"].to_numpy(dtype=np.int64)

    # -------------------------------------------------------------------------
    # 1. Pre-Extract Candidate NRC Lookup Tables (indexed by movie_idx)
    # -------------------------------------------------------------------------
    catalog_size = int(movie_nrc_df["movie_idx"].max()) + 1 if not movie_nrc_df.empty else 3883
    cand_matrix = np.full((catalog_size, 8), np.nan, dtype=np.float32)
    cand_avail_arr = np.zeros(catalog_size, dtype=np.int32)
    cand_matched_arr = np.zeros(catalog_size, dtype=np.int32)

    nrc_cols = [f"candidate_nrc_{e}" for e in PLUTCHIK_EMOTIONS]

    for _, row in movie_nrc_df.iterrows():
        m_idx = int(row["movie_idx"])
        avail = int(row["candidate_nrc_available"])
        cand_avail_arr[m_idx] = avail
        cand_matched_arr[m_idx] = int(row["candidate_nrc_matched_token_count"])
        if avail:
            cand_matrix[m_idx] = row[nrc_cols].to_numpy(dtype=np.float32)

    # -------------------------------------------------------------------------
    # 2. Pre-Allocate Feature Columns
    # -------------------------------------------------------------------------
    nrc_dot = np.full(n_rows, np.nan, dtype=np.float32)
    nrc_cosine = np.full(n_rows, np.nan, dtype=np.float32)
    nrc_l1 = np.full(n_rows, np.nan, dtype=np.float32)
    nrc_l2 = np.full(n_rows, np.nan, dtype=np.float32)

    # Candidate 8D vectors
    cand_nrc_dims: Dict[str, np.ndarray] = {
        col: np.full(n_rows, np.nan, dtype=np.float32) for col in CANDIDATE_NRC_EMOTION_FEATURES
    }

    # User 8D vectors
    user_nrc_dims: Dict[str, np.ndarray] = {
        col: np.full(n_rows, np.nan, dtype=np.float32) for col in USER_NRC_EMOTION_FEATURES
    }

    # Coverage flags
    cand_nrc_avail = np.zeros(n_rows, dtype=np.int32)
    user_nrc_avail = np.zeros(n_rows, dtype=np.int32)
    cand_matched_tokens = np.zeros(n_rows, dtype=np.int32)

    # Fast row-level candidate vector population
    for i in range(n_rows):
        m_idx = movie_indices[i]
        if m_idx < catalog_size:
            is_avail = cand_avail_arr[m_idx]
            cand_nrc_avail[i] = is_avail
            cand_matched_tokens[i] = cand_matched_arr[m_idx]
            if is_avail:
                for idx, emo in enumerate(PLUTCHIK_EMOTIONS):
                    cand_nrc_dims[f"candidate_nrc_{emo}"][i] = cand_matrix[m_idx, idx]

    # -------------------------------------------------------------------------
    # 3. Fast Vectorized User-Block Feature Computation
    # -------------------------------------------------------------------------
    if n_rows > 0:
        change_points = np.where(user_indices[:-1] != user_indices[1:])[0] + 1
        starts = np.r_[0, change_points]
        ends = np.r_[change_points, n_rows]

        for start, end in zip(starts, ends):
            u_idx = int(user_indices[start])
            prof = nrc_user_profiles.get(u_idx)

            if prof is None or not prof.nrc_available or prof.nrc_profile is None:
                continue

            u_vec = prof.nrc_profile.astype(np.float32)
            u_norm = float(np.linalg.norm(u_vec))
            user_nrc_avail[start:end] = 1

            # Populate user dimensions for this block
            for idx, emo in enumerate(PLUTCHIK_EMOTIONS):
                user_nrc_dims[f"user_nrc_{emo}"][start:end] = u_vec[idx]

            # Vectorized compatibility metrics against candidate movies in block
            cand_m = movie_indices[start:end]
            cand_sub = cand_matrix[cand_m]  # shape (k, 8)
            block_c_avail = cand_nrc_avail[start:end]

            # Matrix-vector dot product: (k, 8) @ (8,) -> (k,)
            dots = np.dot(cand_sub, u_vec)

            # Norms for cosine similarity
            cand_norms = np.linalg.norm(cand_sub, axis=1)
            denom = cand_norms * u_norm
            cosines = np.where(denom > 0, dots / denom, 0.0)
            cosines = np.clip(cosines, -1.0, 1.0)

            # Distances
            diffs = cand_sub - u_vec  # shape (k, 8)
            l1_dists = np.sum(np.abs(diffs), axis=1)
            l2_dists = np.linalg.norm(diffs, axis=1)

            # Mask out candidates with no valid NRC vector
            dots = np.where(block_c_avail == 1, dots, np.nan)
            cosines = np.where(block_c_avail == 1, cosines, np.nan)
            l1_dists = np.where(block_c_avail == 1, l1_dists, np.nan)
            l2_dists = np.where(block_c_avail == 1, l2_dists, np.nan)

            nrc_dot[start:end] = dots
            nrc_cosine[start:end] = cosines
            nrc_l1[start:end] = l1_dists
            nrc_l2[start:end] = l2_dists

    # -------------------------------------------------------------------------
    # 4. Assemble Output DataFrame
    # -------------------------------------------------------------------------
    df_out = phase4b_df.copy()

    # Compatibility features
    df_out["nrc_emotion_cosine_similarity"] = nrc_cosine
    df_out["nrc_emotion_dot_product"] = nrc_dot
    df_out["nrc_emotion_l1_distance"] = nrc_l1
    df_out["nrc_emotion_l2_distance"] = nrc_l2

    # Candidate dimensions
    for col in CANDIDATE_NRC_EMOTION_FEATURES:
        df_out[col] = cand_nrc_dims[col]

    # User dimensions
    for col in USER_NRC_EMOTION_FEATURES:
        df_out[col] = user_nrc_dims[col]

    # Coverage flags
    df_out["candidate_nrc_available"] = cand_nrc_avail
    df_out["user_nrc_profile_available"] = user_nrc_avail
    df_out["candidate_nrc_matched_token_count"] = cand_matched_tokens

    if validate:
        validate_nrc_candidate_features(df_out, phase4b_df)

    logger.info(
        f"Phase 4C NRC candidate features appended successfully: {df_out.shape} "
        f"({len(df_out.columns)} total columns)."
    )
    return df_out


def validate_nrc_candidate_features(
    full_df: pd.DataFrame,
    phase4b_df: pd.DataFrame,
) -> None:
    """Run strict integrity, alignment, distribution, and leakage assertions."""
    # 1. Row count preservation
    assert len(full_df) == len(phase4b_df), (
        f"Row count mismatch: expected {len(phase4b_df)}, got {len(full_df)}"
    )

    # 2. Candidate alignment preservation
    np.testing.assert_array_equal(
        full_df["user_idx"].to_numpy(),
        phase4b_df["user_idx"].to_numpy(),
        err_msg="user_idx column mismatch after NRC feature merge.",
    )
    np.testing.assert_array_equal(
        full_df["movie_idx"].to_numpy(),
        phase4b_df["movie_idx"].to_numpy(),
        err_msg="movie_idx column mismatch after NRC feature merge.",
    )
    np.testing.assert_array_equal(
        full_df["candidate_rank"].to_numpy(),
        phase4b_df["candidate_rank"].to_numpy(),
        err_msg="candidate_rank mismatch after NRC feature merge.",
    )
    np.testing.assert_allclose(
        full_df["ease_score"].to_numpy(),
        phase4b_df["ease_score"].to_numpy(),
        rtol=1e-5,
        err_msg="ease_score mismatch after NRC feature merge.",
    )

    # 3. All Phase 4B columns preserved exactly
    for col in PHASE4B_COLUMNS:
        assert col in full_df.columns, f"Phase 4B column '{col}' missing from Phase 4C output."

    # 4. All Phase 4C columns present
    for col in NRC_EMOTION_FEATURES:
        assert col in full_df.columns, f"Phase 4C column '{col}' missing from Phase 4C output."

    # 5. Full column schema check
    missing_cols = set(FULL_COLUMNS) - set(full_df.columns)
    assert not missing_cols, f"Features DataFrame is missing expected columns: {missing_cols}"
    assert len(full_df.columns) == len(FULL_COLUMNS), (
        f"Expected {len(FULL_COLUMNS)} columns, got {len(full_df.columns)}"
    )

    # 6. Leakage assertion
    assert int(full_df["candidate_seen_in_positive_train"].sum()) == 0, (
        "CRITICAL LEAKAGE DETECTED: positive train items leaked into candidate pool!"
    )

    # 7. Distribution sum checks for valid NRC vectors
    valid_cand_mask = full_df["candidate_nrc_available"] == 1
    if valid_cand_mask.any():
        cand_cols = [f"candidate_nrc_{e}" for e in PLUTCHIK_EMOTIONS]
        cand_sums = full_df.loc[valid_cand_mask, cand_cols].sum(axis=1)
        np.testing.assert_allclose(
            cand_sums.to_numpy(),
            1.0,
            atol=1e-4,
            err_msg="Candidate NRC vectors do not sum to 1.0 where available.",
        )

    valid_user_mask = full_df["user_nrc_profile_available"] == 1
    if valid_user_mask.any():
        user_cols = [f"user_nrc_{e}" for e in PLUTCHIK_EMOTIONS]
        user_sums = full_df.loc[valid_user_mask, user_cols].sum(axis=1)
        np.testing.assert_allclose(
            user_sums.to_numpy(),
            1.0,
            atol=1e-4,
            err_msg="User NRC profiles do not sum to 1.0 where available.",
        )

    # 8. Cosine similarity bounds
    valid_sims = full_df["nrc_emotion_cosine_similarity"].dropna()
    if len(valid_sims) > 0:
        assert (valid_sims >= -1.0001).all() and (valid_sims <= 1.0001).all(), (
            "NRC cosine similarity outside valid bounds [-1.0, 1.0]"
        )
