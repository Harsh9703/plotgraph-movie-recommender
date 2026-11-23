"""Candidate-Level Feature Table Construction (Phase 4A).

Constructs the comprehensive feature table for all 604,000 EASE candidate rows:
1. Base candidate/history features (rank, EASE score, train count, coverage flags)
2. KG-M affinity & overlap features (genre, director, actor, keyword, collection, decade)
3. KG-L semantic affinity & overlap features (theme, setting, narrative_device, protagonist_type, source_material)
4. Continuous emotion compatibility (cosine, dot product, L1, L2) and raw dimension values

Enforces strict leakage assertions and preserves exact candidate ordering.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import (
    ALL_COLUMNS,
    CANDIDATE_EMOTION_FEATURES,
    KGL_CATEGORIES,
    KGM_CATEGORIES,
    PLUTCHIK_EMOTIONS,
    USER_EMOTION_FEATURES,
)
from .movie_indexer import MovieEntityIndex, MovieMetadataItem
from .user_profiler import UserProfile, UserProfileIndex

logger = logging.getLogger(__name__)


def build_candidate_features(
    candidate_df: pd.DataFrame,
    user_profiles: UserProfileIndex,
    movie_index: MovieEntityIndex,
    validate: bool = True,
) -> pd.DataFrame:
    """Build the deterministic candidate feature matrix for all candidate rows.

    Args:
        candidate_df: DataFrame containing at least ['user_idx', 'movie_idx', 'rank', 'score'].
        user_profiles: Leakage-safe UserProfileIndex built strictly from TRAIN split.
        movie_index: Fast in-memory MovieEntityIndex.
        validate: Whether to run strict leakage and integrity assertions.

    Returns:
        DataFrame containing exactly len(candidate_df) rows and all feature columns.
    """
    n_rows = len(candidate_df)
    logger.info(f"Extracting features for {n_rows:,} candidate rows...")

    user_indices = candidate_df["user_idx"].to_numpy(dtype=np.int64)
    movie_indices = candidate_df["movie_idx"].to_numpy(dtype=np.int64)

    # 1. Base Candidate Arrays
    ranks = candidate_df["rank"].to_numpy(dtype=np.int32)
    scores = candidate_df["score"].to_numpy(dtype=np.float32)

    user_pos_train_counts = np.zeros(n_rows, dtype=np.int32)
    seen_in_train = np.zeros(n_rows, dtype=np.int32)
    has_tmdb = np.zeros(n_rows, dtype=np.int32)
    has_overview = np.zeros(n_rows, dtype=np.int32)
    has_kgm = np.zeros(n_rows, dtype=np.int32)
    has_kgl = np.zeros(n_rows, dtype=np.int32)
    cand_emo_avail = np.zeros(n_rows, dtype=np.int32)
    user_emo_avail = np.zeros(n_rows, dtype=np.int32)

    # 2. KG-M Arrays
    kgm_affinities: Dict[str, np.ndarray] = {
        cat: np.zeros(n_rows, dtype=np.float32) for cat in KGM_CATEGORIES
    }
    kgm_overlaps: Dict[str, np.ndarray] = {
        cat: np.zeros(n_rows, dtype=np.int32) for cat in KGM_CATEGORIES
    }
    kgm_mean = np.zeros(n_rows, dtype=np.float32)
    kgm_max = np.zeros(n_rows, dtype=np.float32)
    kgm_matched = np.zeros(n_rows, dtype=np.int32)

    # 3. KG-L Arrays
    kgl_affinities: Dict[str, np.ndarray] = {
        cat: np.zeros(n_rows, dtype=np.float32) for cat in KGL_CATEGORIES
    }
    kgl_overlaps: Dict[str, np.ndarray] = {
        cat: np.zeros(n_rows, dtype=np.int32) for cat in KGL_CATEGORIES
    }
    kgl_mean = np.zeros(n_rows, dtype=np.float32)
    kgl_max = np.zeros(n_rows, dtype=np.float32)
    kgl_matched = np.zeros(n_rows, dtype=np.int32)

    # 4. Emotion Arrays
    emo_cosine = np.full(n_rows, np.nan, dtype=np.float32)
    emo_dot = np.full(n_rows, np.nan, dtype=np.float32)
    emo_l1 = np.full(n_rows, np.nan, dtype=np.float32)
    emo_l2 = np.full(n_rows, np.nan, dtype=np.float32)

    cand_emotions: Dict[str, np.ndarray] = {
        emo: np.full(n_rows, np.nan, dtype=np.float32) for emo in PLUTCHIK_EMOTIONS
    }
    user_emotions: Dict[str, np.ndarray] = {
        emo: np.full(n_rows, np.nan, dtype=np.float32) for emo in PLUTCHIK_EMOTIONS
    }

    # -------------------------------------------------------------------------
    # Fast In-Memory Iteration Grouped by User
    # -------------------------------------------------------------------------
    # Cache user profile references
    cached_u_idx = -1
    curr_prof: Optional[UserProfile] = None
    curr_u_emo_vec: Optional[np.ndarray] = None
    curr_u_emo_norm: float = 0.0

    for i in range(n_rows):
        u_idx = int(user_indices[i])
        m_idx = int(movie_indices[i])

        # Check if user changed
        if u_idx != cached_u_idx:
            cached_u_idx = u_idx
            curr_prof = user_profiles.get(u_idx)
            if curr_prof is not None and curr_prof.emotion_profile is not None:
                curr_u_emo_vec = curr_prof.emotion_profile
                curr_u_emo_norm = float(np.linalg.norm(curr_u_emo_vec))
            else:
                curr_u_emo_vec = None
                curr_u_emo_norm = 0.0

        m_item = movie_index.get(m_idx)

        # Baseline features
        if curr_prof is not None:
            user_pos_train_counts[i] = curr_prof.positive_train_count
            if m_idx in curr_prof.positive_train_movies:
                seen_in_train[i] = 1

            if curr_prof.emotion_profile_available:
                user_emo_avail[i] = 1
                for idx, emo in enumerate(PLUTCHIK_EMOTIONS):
                    user_emotions[emo][i] = curr_prof.emotion_profile[idx]

        if m_item is not None:
            has_tmdb[i] = 1 if m_item.has_tmdb_link else 0
            has_overview[i] = 1 if m_item.has_overview else 0
            has_kgm[i] = 1 if m_item.has_kgm_metadata else 0
            has_kgl[i] = 1 if m_item.has_kgl_semantics else 0
            if m_item.has_emotion and m_item.emotion_vector is not None:
                cand_emo_avail[i] = 1
                for idx, emo in enumerate(PLUTCHIK_EMOTIONS):
                    cand_emotions[emo][i] = m_item.emotion_vector[idx]

        # KG-M Features
        kgm_aff_vals = []
        if curr_prof is not None and m_item is not None:
            for cat in KGM_CATEGORIES:
                cand_ents = m_item.kgm_entities.get(cat, [])
                u_cat_prof = curr_prof.kgm_profiles.get(cat, {})
                u_cat_set = curr_prof.kgm_entity_sets.get(cat, set())

                # Sum user-profile weights for entities attached to candidate
                aff = sum(u_cat_prof.get(e, 0.0) for e in cand_ents)
                kgm_affinities[cat][i] = aff
                kgm_aff_vals.append(aff)

                # Overlap count
                overlap = sum(1 for e in cand_ents if e in u_cat_set)
                kgm_overlaps[cat][i] = overlap
        else:
            kgm_aff_vals = [0.0] * len(KGM_CATEGORIES)

        kgm_mean[i] = float(np.mean(kgm_aff_vals))
        kgm_max[i] = float(np.max(kgm_aff_vals))
        kgm_matched[i] = int(sum(1 for a in kgm_aff_vals if a > 0.0))

        # KG-L Features
        kgl_aff_vals = []
        if curr_prof is not None and m_item is not None:
            for cat in KGL_CATEGORIES:
                cand_ents = m_item.kgl_entities.get(cat, [])
                u_cat_prof = curr_prof.kgl_profiles.get(cat, {})
                u_cat_set = curr_prof.kgl_entity_sets.get(cat, set())

                aff = sum(u_cat_prof.get(e, 0.0) for e in cand_ents)
                kgl_affinities[cat][i] = aff
                kgl_aff_vals.append(aff)

                overlap = sum(1 for e in cand_ents if e in u_cat_set)
                kgl_overlaps[cat][i] = overlap
        else:
            kgl_aff_vals = [0.0] * len(KGL_CATEGORIES)

        kgl_mean[i] = float(np.mean(kgl_aff_vals))
        kgl_max[i] = float(np.max(kgl_aff_vals))
        kgl_matched[i] = int(sum(1 for a in kgl_aff_vals if a > 0.0))

        # Continuous Emotion Compatibility Metrics
        if (
            curr_u_emo_vec is not None
            and m_item is not None
            and m_item.emotion_vector is not None
        ):
            v_cand = m_item.emotion_vector
            dot_prod = float(np.dot(curr_u_emo_vec, v_cand))
            v_norm = float(np.linalg.norm(v_cand))

            cos_sim = 0.0
            if curr_u_emo_norm > 0.0 and v_norm > 0.0:
                cos_sim = dot_prod / (curr_u_emo_norm * v_norm)
                cos_sim = float(np.clip(cos_sim, -1.0, 1.0))

            diff = curr_u_emo_vec - v_cand
            l1_dist = float(np.sum(np.abs(diff)))
            l2_dist = float(np.linalg.norm(diff))

            emo_dot[i] = dot_prod
            emo_cosine[i] = cos_sim
            emo_l1[i] = l1_dist
            emo_l2[i] = l2_dist

    # -------------------------------------------------------------------------
    # Assemble DataFrame
    # -------------------------------------------------------------------------
    data: Dict[str, Any] = {
        "user_idx": user_indices,
        "movie_idx": movie_indices,
        # Base
        "candidate_rank": ranks,
        "ease_score": scores,
        "user_positive_train_count": user_pos_train_counts,
        "candidate_seen_in_positive_train": seen_in_train,
        "candidate_has_tmdb_link": has_tmdb,
        "candidate_has_overview": has_overview,
        "candidate_has_kgm_metadata": has_kgm,
        "candidate_has_kgl_semantics": has_kgl,
        "candidate_emotion_available": cand_emo_avail,
        "user_emotion_profile_available": user_emo_avail,
        # KG-M Affinities
        "kgm_genre_affinity": kgm_affinities["genre"],
        "kgm_director_affinity": kgm_affinities["director"],
        "kgm_actor_affinity": kgm_affinities["actor"],
        "kgm_keyword_affinity": kgm_affinities["keyword"],
        "kgm_collection_affinity": kgm_affinities["collection"],
        "kgm_decade_affinity": kgm_affinities["decade"],
        "kgm_affinity_mean": kgm_mean,
        "kgm_affinity_max": kgm_max,
        "kgm_categories_matched": kgm_matched,
        # KG-M Overlaps
        "shared_genre_count": kgm_overlaps["genre"],
        "shared_director_count": kgm_overlaps["director"],
        "shared_actor_count": kgm_overlaps["actor"],
        "shared_keyword_count": kgm_overlaps["keyword"],
        "shared_collection_count": kgm_overlaps["collection"],
        "shared_decade_count": kgm_overlaps["decade"],
        # KG-L Affinities
        "kgl_theme_affinity": kgl_affinities["theme"],
        "kgl_setting_affinity": kgl_affinities["setting"],
        "kgl_narrative_device_affinity": kgl_affinities["narrative_device"],
        "kgl_protagonist_type_affinity": kgl_affinities["protagonist_type"],
        "kgl_source_material_affinity": kgl_affinities["source_material"],
        "kgl_semantic_affinity_mean": kgl_mean,
        "kgl_semantic_affinity_max": kgl_max,
        "kgl_semantic_categories_matched": kgl_matched,
        # KG-L Overlaps
        "shared_theme_count": kgl_overlaps["theme"],
        "shared_setting_count": kgl_overlaps["setting"],
        "shared_narrative_device_count": kgl_overlaps["narrative_device"],
        "shared_protagonist_type_count": kgl_overlaps["protagonist_type"],
        "shared_source_material_count": kgl_overlaps["source_material"],
        # Emotion Compatibility
        "emotion_cosine_similarity": emo_cosine,
        "emotion_dot_product": emo_dot,
        "emotion_l1_distance": emo_l1,
        "emotion_l2_distance": emo_l2,
    }

    # Candidate Raw Emotions
    for emo in PLUTCHIK_EMOTIONS:
        data[f"candidate_emotion_{emo}"] = cand_emotions[emo]

    # User Raw Emotions
    for emo in PLUTCHIK_EMOTIONS:
        data[f"user_emotion_{emo}"] = user_emotions[emo]

    df_out = pd.DataFrame(data)

    if validate:
        validate_candidate_features(df_out, candidate_df)

    logger.info(f"Feature table built successfully: {df_out.shape} ({len(df_out.columns)} columns).")
    return df_out


def validate_candidate_features(
    features_df: pd.DataFrame,
    original_candidate_df: pd.DataFrame,
) -> None:
    """Run strict verification assertions on the generated candidate features table."""
    # 1. Row count preservation
    assert len(features_df) == len(original_candidate_df), (
        f"Row count mismatch: expected {len(original_candidate_df)}, got {len(features_df)}"
    )

    # 2. Candidate pool alignment
    np.testing.assert_array_equal(
        features_df["user_idx"].to_numpy(),
        original_candidate_df["user_idx"].to_numpy(),
        err_msg="user_idx column does not match candidate pool exactly.",
    )
    np.testing.assert_array_equal(
        features_df["movie_idx"].to_numpy(),
        original_candidate_df["movie_idx"].to_numpy(),
        err_msg="movie_idx column does not match candidate pool exactly.",
    )
    np.testing.assert_array_equal(
        features_df["candidate_rank"].to_numpy(),
        original_candidate_df["rank"].to_numpy(),
        err_msg="candidate_rank does not match candidate pool rank exactly.",
    )
    np.testing.assert_allclose(
        features_df["ease_score"].to_numpy(),
        original_candidate_df["score"].to_numpy(),
        rtol=1e-5,
        atol=1e-6,
        err_msg="ease_score does not match candidate pool score exactly.",
    )

    # 3. Leakage Guard Assertion: Zero positive train items in candidates
    leaked_count = int(features_df["candidate_seen_in_positive_train"].sum())
    assert leaked_count == 0, (
        f"CRITICAL LEAKAGE DETECTED: {leaked_count} candidate items were seen in user's positive TRAIN history!"
    )

    # 4. Column coverage
    missing_cols = set(ALL_COLUMNS) - set(features_df.columns)
    assert not missing_cols, f"Features DataFrame is missing expected columns: {missing_cols}"

    # 5. Non-null assertions on identifiers and base features
    for col in ["user_idx", "movie_idx", "candidate_rank", "ease_score", "user_positive_train_count"]:
        assert features_df[col].notna().all(), f"Column {col} contains unexpected nulls."
