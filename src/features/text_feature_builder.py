"""Candidate-Level Text Semantic Feature Builder (Phase 4B).

Appends leakage-safe TF-IDF and SBERT text-semantic compatibility features
to the Phase 4A candidate feature table:
- tfidf_cosine_similarity
- tfidf_dot_product
- tfidf_candidate_available
- tfidf_user_profile_available
- sbert_cosine_similarity
- sbert_dot_product
- sbert_candidate_available
- sbert_user_profile_available

Preserves exact row count (604,000), ordering, and all Phase 4A columns.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .constants import PHASE4A_COLUMNS, PHASE4B_COLUMNS, TEXT_FEATURES
from .text_encoder import SBERTMovieEncoder, TFIDFMovieEncoder
from .text_profiler import TextUserProfileIndex

logger = logging.getLogger(__name__)


def build_text_candidate_features(
    phase4a_df: pd.DataFrame,
    text_user_profiles: TextUserProfileIndex,
    tfidf_encoder: TFIDFMovieEncoder,
    sbert_encoder: SBERTMovieEncoder,
    validate: bool = True,
) -> pd.DataFrame:
    """Compute and append TF-IDF and SBERT text features to candidate DataFrame.

    Args:
        phase4a_df: Existing Phase 4A candidate features DataFrame (604,000 rows).
        text_user_profiles: Leakage-safe TextUserProfileIndex.
        tfidf_encoder: Fitted TFIDFMovieEncoder.
        sbert_encoder: SBERTMovieEncoder with precomputed embeddings.
        validate: Whether to run verification assertions.

    Returns:
        DataFrame with exactly len(phase4a_df) rows and all Phase 4A + 4B columns.
    """
    n_rows = len(phase4a_df)
    logger.info(f"Computing text-semantic features for {n_rows:,} candidate rows...")

    user_indices = phase4a_df["user_idx"].to_numpy(dtype=np.int64)
    movie_indices = phase4a_df["movie_idx"].to_numpy(dtype=np.int64)

    # Pre-allocate feature arrays
    tfidf_sim = np.full(n_rows, np.nan, dtype=np.float32)
    tfidf_dot = np.full(n_rows, np.nan, dtype=np.float32)
    tfidf_cand_avail = np.zeros(n_rows, dtype=np.int32)
    tfidf_user_avail = np.zeros(n_rows, dtype=np.int32)

    sbert_sim = np.full(n_rows, np.nan, dtype=np.float32)
    sbert_dot = np.full(n_rows, np.nan, dtype=np.float32)
    sbert_cand_avail = np.zeros(n_rows, dtype=np.int32)
    sbert_user_avail = np.zeros(n_rows, dtype=np.int32)

    # Reference masks
    tfidf_has_text = (
        tfidf_encoder.has_text_mask
        if tfidf_encoder.has_text_mask is not None
        else np.zeros(tfidf_encoder.catalog_size, dtype=bool)
    )
    sbert_has_emb = (
        sbert_encoder.has_embedding_mask
        if sbert_encoder.has_embedding_mask is not None
        else np.zeros(sbert_encoder.embeddings_matrix.shape[0] if sbert_encoder.embeddings_matrix is not None else 3883, dtype=bool)
    )
    sbert_matrix = sbert_encoder.embeddings_matrix

    # Pre-populate candidate availability flags
    for i in range(n_rows):
        m_idx = movie_indices[i]
        if m_idx < len(tfidf_has_text) and tfidf_has_text[m_idx]:
            tfidf_cand_avail[i] = 1
        if m_idx < len(sbert_has_emb) and sbert_has_emb[m_idx]:
            sbert_cand_avail[i] = 1

    # -------------------------------------------------------------------------
    # Fast User-Block Vectorized Feature Computation
    # -------------------------------------------------------------------------
    if n_rows > 0:
        change_points = np.where(user_indices[:-1] != user_indices[1:])[0] + 1
        starts = np.r_[0, change_points]
        ends = np.r_[change_points, n_rows]

        tfidf_cat_matrix = tfidf_encoder.movie_tfidf_matrix

        for start, end in zip(starts, ends):
            u_idx = int(user_indices[start])
            cand_m = movie_indices[start:end]
            curr_text_prof = text_user_profiles.get(u_idx)

            if curr_text_prof is None:
                continue

            # 1. TF-IDF block computation
            if curr_text_prof.tfidf_available and curr_text_prof.tfidf_profile is not None and tfidf_cat_matrix is not None:
                tfidf_user_avail[start:end] = 1
                cand_sparse = tfidf_cat_matrix[cand_m]
                # Dot product between (k, vocab_size) and (vocab_size, 1) -> (k, 1)
                t_sims = cand_sparse.dot(curr_text_prof.tfidf_profile.T).toarray().flatten()
                t_sims = np.clip(t_sims, -1.0, 1.0)
                # Mask out candidates with no overview
                c_avail = tfidf_cand_avail[start:end]
                t_sims[c_avail == 0] = np.nan
                tfidf_sim[start:end] = t_sims
                tfidf_dot[start:end] = t_sims

            # 2. SBERT block computation
            if curr_text_prof.sbert_available and curr_text_prof.sbert_profile is not None and sbert_matrix is not None:
                sbert_user_avail[start:end] = 1
                cand_dense = sbert_matrix[cand_m]
                # Matrix-vector product: (k, dim) @ (dim,) -> (k,)
                s_sims = np.dot(cand_dense, curr_text_prof.sbert_profile)
                s_sims = np.clip(s_sims, -1.0, 1.0)
                # Mask out candidates with no overview
                c_avail = sbert_cand_avail[start:end]
                s_sims[c_avail == 0] = np.nan
                sbert_sim[start:end] = s_sims
                sbert_dot[start:end] = s_sims

    # Append new columns to copy of Phase 4A DataFrame
    df_out = phase4a_df.copy()
    df_out["tfidf_cosine_similarity"] = tfidf_sim
    df_out["tfidf_dot_product"] = tfidf_dot
    df_out["tfidf_candidate_available"] = tfidf_cand_avail
    df_out["tfidf_user_profile_available"] = tfidf_user_avail

    df_out["sbert_cosine_similarity"] = sbert_sim
    df_out["sbert_dot_product"] = sbert_dot
    df_out["sbert_candidate_available"] = sbert_cand_avail
    df_out["sbert_user_profile_available"] = sbert_user_avail

    if validate:
        validate_text_candidate_features(df_out, phase4a_df)

    logger.info(
        f"Phase 4B text candidate features appended successfully: {df_out.shape} "
        f"({len(df_out.columns)} total columns)."
    )
    return df_out


def validate_text_candidate_features(
    features_df: pd.DataFrame,
    phase4a_df: pd.DataFrame,
) -> None:
    """Run strict integrity, alignment, and leakage assertions."""
    # 1. Row count preservation
    assert len(features_df) == len(phase4a_df), (
        f"Row count mismatch: expected {len(phase4a_df)}, got {len(features_df)}"
    )

    # 2. Candidate alignment preservation
    np.testing.assert_array_equal(
        features_df["user_idx"].to_numpy(),
        phase4a_df["user_idx"].to_numpy(),
        err_msg="user_idx column mismatch after text feature merge.",
    )
    np.testing.assert_array_equal(
        features_df["movie_idx"].to_numpy(),
        phase4a_df["movie_idx"].to_numpy(),
        err_msg="movie_idx column mismatch after text feature merge.",
    )
    np.testing.assert_array_equal(
        features_df["candidate_rank"].to_numpy(),
        phase4a_df["candidate_rank"].to_numpy(),
        err_msg="candidate_rank mismatch after text feature merge.",
    )
    np.testing.assert_allclose(
        features_df["ease_score"].to_numpy(),
        phase4a_df["ease_score"].to_numpy(),
        rtol=1e-5,
        err_msg="ease_score mismatch after text feature merge.",
    )

    # 3. All Phase 4A columns preserved exactly
    for col in PHASE4A_COLUMNS:
        assert col in features_df.columns, f"Phase 4A column '{col}' missing from Phase 4B output."

    # 4. All Phase 4B columns present
    for col in TEXT_FEATURES:
        assert col in features_df.columns, f"Phase 4B column '{col}' missing from Phase 4B output."

    # 5. Full Phase 4B column set
    missing_cols = set(PHASE4B_COLUMNS) - set(features_df.columns)
    assert not missing_cols, f"Features DataFrame is missing expected columns: {missing_cols}"

    # 6. Leakage assertion
    assert int(features_df["candidate_seen_in_positive_train"].sum()) == 0, (
        "CRITICAL LEAKAGE DETECTED: positive train items leaked into candidate pool!"
    )
