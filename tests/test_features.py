"""Unit and Integration Tests for PlotGraph Feature Engineering (Phase 4A).

Verifies:
1. Exponential recency decay formulation and age calculations.
2. Positive-only TRAIN interaction extraction (rating >= 4.0).
3. Leakage guards rejecting VAL and TEST interactions.
4. User profile normalization (distributions sum to 1.0).
5. KG-M affinity summation and shared entity overlap counts.
6. KG-L semantic affinity and overlap counts.
7. Continuous 8D emotion profile aggregation (weighted average) and compatibility metrics (cosine, dot, L1, L2).
8. Missing emotion handling (availability flags and NaNs for LightGBM).
9. Candidate pool row count, rank, and EASE score preservation.
10. Feature group definitions and ablation completeness.
11. Real artifact end-to-end verification (if present on disk).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pytest

from src.features.constants import (
    ALL_COLUMNS,
    ALL_FEATURES,
    ALL_KG_FEATURES,
    BASE_FEATURES,
    CANDIDATE_EMOTION_FEATURES,
    DEFAULT_HALF_LIFE_DAYS,
    EMOTION_COMPATIBILITY_FEATURES,
    EMOTION_FEATURES,
    IDENTIFIER_COLUMNS,
    KGL_CATEGORIES,
    KGL_SEMANTIC_AFFINITY_FEATURES,
    KGL_SEMANTIC_FEATURES,
    KGL_SEMANTIC_OVERLAP_FEATURES,
    KGM_AFFINITY_FEATURES,
    KGM_CATEGORIES,
    KGM_FEATURES,
    KGM_OVERLAP_FEATURES,
    OVERLAP_COUNT_FEATURES,
    PLUTCHIK_EMOTIONS,
    USER_EMOTION_FEATURES,
)
from src.features.feature_builder import (
    build_candidate_features,
    validate_candidate_features,
)
from src.features.movie_indexer import MovieEntityIndex, MovieMetadataItem
from src.features.summary import generate_feature_summary
from src.features.user_profiler import (
    UserProfile,
    UserProfileIndex,
    build_user_profiles,
    compute_age_days,
    compute_recency_weight,
)


# ==============================================================================
# 1. RECENCY WEIGHTING TESTS
# ==============================================================================

def test_recency_weighting_exponential_decay():
    """Verify exponential half-life recency weighting and age calculation."""
    # Age 0 -> weight 1.0
    assert compute_recency_weight(0.0, half_life_days=365.0) == 1.0

    # Age == half_life -> weight 0.5
    assert pytest.approx(compute_recency_weight(365.0, half_life_days=365.0), rel=1e-5) == 0.5

    # Age == 2 * half_life -> weight 0.25
    assert pytest.approx(compute_recency_weight(730.0, half_life_days=365.0), rel=1e-5) == 0.25

    # Strict Monotonicity: older -> lower weight
    w1 = compute_recency_weight(10.0, half_life_days=180.0)
    w2 = compute_recency_weight(50.0, half_life_days=180.0)
    w3 = compute_recency_weight(100.0, half_life_days=180.0)
    assert w1 > w2 > w3

    # Half life <= 0 -> unweighted 1.0
    assert compute_recency_weight(100.0, half_life_days=0.0) == 1.0

    # Age calculation: 86,400 seconds = 1.0 day
    assert compute_age_days(1000, 1000 + 86400) == 1.0

    # Temporal leakage detection in age calculation
    with pytest.raises(ValueError, match="Temporal leakage detected"):
        compute_age_days(timestamp=2000, max_timestamp=1000)


# ==============================================================================
# 2. POSITIVE-ONLY TRAIN INTERACTION EXTRACTION TESTS
# ==============================================================================

def test_train_only_positive_history_extraction():
    """Verify that only interactions with rating >= 4.0 (positive == 1) enter profiles."""
    synthetic_train = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 10, "rating": 5.0, "timestamp": 1000, "positive": 1},
        {"user_idx": 0, "movie_idx": 11, "rating": 4.0, "timestamp": 2000, "positive": 1},
        {"user_idx": 0, "movie_idx": 12, "rating": 3.0, "timestamp": 3000, "positive": 0},
        {"user_idx": 0, "movie_idx": 13, "rating": 1.0, "timestamp": 4000, "positive": 0},
    ])

    meta = pd.DataFrame([
        {"movie_idx": i, "movie_id": i + 1, "ml_title": f"M {i}", "tmdb_genres": ["Drama"]}
        for i in range(15)
    ])
    index = MovieEntityIndex.from_dataframes(meta)

    user_profiles = build_user_profiles(
        train_df=synthetic_train,
        movie_index=index,
        positive_threshold=4.0,
        total_users=1,
    )

    u0 = user_profiles.get(0)
    assert u0 is not None
    # Only 2 positive movies (10 and 11), movies 12 and 13 excluded!
    assert u0.positive_train_count == 2
    assert u0.positive_train_movies == {10, 11}
    assert u0.max_train_timestamp == 2000  # from movie 11 (latest positive)


# ==============================================================================
# 3. LEAKAGE GUARDS REJECTING VAL AND TEST
# ==============================================================================

def test_leakage_guards_reject_val_and_test():
    """Verify that passing VAL or TEST splits/columns raises immediate ValueError."""
    meta = pd.DataFrame([{"movie_idx": 0, "movie_id": 1, "ml_title": "M 0"}])
    index = MovieEntityIndex.from_dataframes(meta)

    # Forbidden 'val' column
    df_val = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 1000, "val": 1}
    ])
    with pytest.raises(ValueError, match="Data leakage detected"):
        build_user_profiles(df_val, index, total_users=1)

    # Forbidden 'test' column
    df_test = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 1000, "is_test": True}
    ])
    with pytest.raises(ValueError, match="Data leakage detected"):
        build_user_profiles(df_test, index, total_users=1)


# ==============================================================================
# 4. USER PROFILE NORMALIZATION TESTS
# ==============================================================================

def test_user_profile_normalization():
    """Verify that category preference distributions sum to 1.0."""
    meta = pd.DataFrame([
        {
            "movie_idx": 0,
            "movie_id": 1,
            "ml_title": "Toy Story",
            "tmdb_genres": ["Animation", "Comedy"],
            "director": "John Lasseter",
        },
        {
            "movie_idx": 1,
            "movie_id": 2,
            "ml_title": "A Bug's Life",
            "tmdb_genres": ["Animation", "Family"],
            "director": "John Lasseter",
        },
    ])
    sem = pd.DataFrame([
        {
            "movie_idx": 0,
            "movie_id": 1,
            "themes": ["Friendship"],
            "settings": ["Bedroom"],
            "narrative_devices": [],
            "protagonist_types": ["Toy"],
            "based_on": [],
        },
        {
            "movie_idx": 1,
            "movie_id": 2,
            "themes": ["Friendship", "Courage"],
            "settings": ["Colony"],
            "narrative_devices": [],
            "protagonist_types": ["Insect"],
            "based_on": [],
        },
    ])
    index = MovieEntityIndex.from_dataframes(meta, sem)

    train = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 1000, "positive": 1},
        {"user_idx": 0, "movie_idx": 1, "rating": 4.0, "timestamp": 1000, "positive": 1},
    ])

    user_profiles = build_user_profiles(train, index, total_users=1)
    u0 = user_profiles.get(0)
    assert u0 is not None

    # Genre profile sums to 1.0
    genre_prof = u0.kgm_profiles["genre"]
    assert pytest.approx(sum(genre_prof.values()), rel=1e-5) == 1.0
    assert "animation" in genre_prof
    assert "comedy" in genre_prof
    assert "family" in genre_prof

    # Director profile has only 1 director with weight 1.0
    dir_prof = u0.kgm_profiles["director"]
    assert pytest.approx(sum(dir_prof.values()), rel=1e-5) == 1.0
    assert dir_prof["john lasseter"] == 1.0

    # Theme profile sums to 1.0
    theme_prof = u0.kgl_profiles["theme"]
    assert pytest.approx(sum(theme_prof.values()), rel=1e-5) == 1.0
    assert "friendship" in theme_prof
    assert "courage" in theme_prof

    # Empty category profile is empty dict
    assert u0.kgl_profiles["narrative_device"] == {}


# ==============================================================================
# 5. KG-M AFFINITY & OVERLAP TESTS
# ==============================================================================

def test_kgm_affinity_and_overlap_counts():
    """Verify category affinity summation and interpretable overlap counts."""
    meta = pd.DataFrame([
        # Train movie 0
        {
            "movie_idx": 0,
            "movie_id": 1,
            "ml_title": "Movie 0",
            "tmdb_genres": ["Action", "Sci-Fi"],
            "director": "James Cameron",
            "top_actors": ["Arnold Schwarzenegger"],
        },
        # Candidate movie 1 (shares director, actor, genre)
        {
            "movie_idx": 1,
            "movie_id": 2,
            "ml_title": "Candidate 1",
            "tmdb_genres": ["Action", "Adventure"],
            "director": "James Cameron",
            "top_actors": ["Arnold Schwarzenegger", "Linda Hamilton"],
        },
        # Candidate movie 2 (shares nothing)
        {
            "movie_idx": 2,
            "movie_id": 3,
            "ml_title": "Candidate 2",
            "tmdb_genres": ["Romance"],
            "director": "Jane Campion",
            "top_actors": ["Holly Hunter"],
        },
    ])
    index = MovieEntityIndex.from_dataframes(meta)
    train = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 1000, "positive": 1}
    ])
    user_profiles = build_user_profiles(train, index, total_users=1)

    cands = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 1, "rank": 1, "score": 0.8},
        {"user_idx": 0, "movie_idx": 2, "rank": 2, "score": 0.4},
    ])

    features = build_candidate_features(cands, user_profiles, index, validate=True)
    assert len(features) == 2

    # Candidate 1:
    row1 = features.iloc[0]
    # Genre affinity: Action is 0.5 of user's genre profile (Action, Sci-Fi)
    assert pytest.approx(row1["kgm_genre_affinity"], rel=1e-5) == 0.5
    assert row1["shared_genre_count"] == 1  # Action
    # Director affinity: James Cameron is 1.0 of user's director profile
    assert pytest.approx(row1["kgm_director_affinity"], rel=1e-5) == 1.0
    assert row1["shared_director_count"] == 1
    # Actor affinity: Arnold is 1.0 of user's actor profile
    assert pytest.approx(row1["kgm_actor_affinity"], rel=1e-5) == 1.0
    assert row1["shared_actor_count"] == 1

    # Candidate 2:
    row2 = features.iloc[1]
    assert row2["kgm_genre_affinity"] == 0.0
    assert row2["shared_genre_count"] == 0
    assert row2["kgm_director_affinity"] == 0.0
    assert row2["shared_director_count"] == 0
    assert row2["kgm_actor_affinity"] == 0.0
    assert row2["shared_actor_count"] == 0


# ==============================================================================
# 6. KG-L SEMANTIC AFFINITY & OVERLAP TESTS
# ==============================================================================

def test_kgl_semantic_affinity_and_overlap_counts():
    """Verify KG-L semantic affinities and overlap counts."""
    meta = pd.DataFrame([
        {"movie_idx": 0, "movie_id": 1, "ml_title": "M 0"},
        {"movie_idx": 1, "movie_id": 2, "ml_title": "M 1"},
    ])
    sem = pd.DataFrame([
        {
            "movie_idx": 0,
            "movie_id": 1,
            "themes": ["Survival", "Dystopia"],
            "settings": ["Desert"],
            "narrative_devices": ["Nonlinear"],
            "protagonist_types": ["Outlaw"],
            "based_on": ["Novel"],
        },
        {
            "movie_idx": 1,
            "movie_id": 2,
            "themes": ["Survival"],
            "settings": ["Desert", "Wasteland"],
            "narrative_devices": [],
            "protagonist_types": ["Outlaw"],
            "based_on": [],
        },
    ])
    index = MovieEntityIndex.from_dataframes(meta, sem)
    train = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 1000, "positive": 1}
    ])
    user_profiles = build_user_profiles(train, index, total_users=1)

    cand = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 1, "rank": 1, "score": 0.9}
    ])
    features = build_candidate_features(cand, user_profiles, index, validate=True)
    row = features.iloc[0]

    # Theme: Survival is 0.5 of user's theme profile
    assert pytest.approx(row["kgl_theme_affinity"], rel=1e-5) == 0.5
    assert row["shared_theme_count"] == 1
    # Setting: Desert is 1.0 of user's setting profile
    assert pytest.approx(row["kgl_setting_affinity"], rel=1e-5) == 1.0
    assert row["shared_setting_count"] == 1
    # Protagonist: Outlaw is 1.0
    assert pytest.approx(row["kgl_protagonist_type_affinity"], rel=1e-5) == 1.0
    assert row["shared_protagonist_type_count"] == 1
    # Narrative device on candidate is empty
    assert row["kgl_narrative_device_affinity"] == 0.0
    assert row["shared_narrative_device_count"] == 0


# ==============================================================================
# 7. CONTINUOUS EMOTION COMPATIBILITY METRICS TESTS
# ==============================================================================

def test_continuous_emotion_profile_and_compatibility_metrics():
    """Verify recency-weighted continuous emotion profile, cosine similarity, dot product, L1, and L2."""
    meta = pd.DataFrame([
        {"movie_idx": 0, "movie_id": 1, "ml_title": "M 0"},
        {"movie_idx": 1, "movie_id": 2, "ml_title": "M 1"},
    ])
    # Movie 0 vector: Joy 0.8, Anticipation 0.6, others 0.0
    # Movie 1 vector: Joy 0.4, Anticipation 0.3, others 0.0
    emo = pd.DataFrame([
        {
            "movie_idx": 0,
            "movie_id": 1,
            "joy": 0.8,
            "trust": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.6,
        },
        {
            "movie_idx": 1,
            "movie_id": 2,
            "joy": 0.4,
            "trust": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.3,
        },
    ])
    index = MovieEntityIndex.from_dataframes(meta, emotion_df=emo)

    train = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 1000, "positive": 1}
    ])
    user_profiles = build_user_profiles(train, index, total_users=1)

    cand = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 1, "rank": 1, "score": 0.75}
    ])
    features = build_candidate_features(cand, user_profiles, index, validate=True)
    row = features.iloc[0]

    assert row["user_emotion_profile_available"] == 1
    assert row["candidate_emotion_available"] == 1

    # u = [0.8, 0, 0, 0, 0, 0, 0, 0.6]
    # v = [0.4, 0, 0, 0, 0, 0, 0, 0.3]
    # Notice v is a positive scalar multiple of u (v = 0.5 * u)!
    # Therefore, cosine similarity must be EXACTLY 1.0!
    assert pytest.approx(row["emotion_cosine_similarity"], rel=1e-5) == 1.0

    # Dot product: 0.8*0.4 + 0.6*0.3 = 0.32 + 0.18 = 0.50
    assert pytest.approx(row["emotion_dot_product"], rel=1e-5) == 0.50

    # L1 distance: |0.8 - 0.4| + |0.6 - 0.3| = 0.4 + 0.3 = 0.70
    assert pytest.approx(row["emotion_l1_distance"], rel=1e-5) == 0.70

    # L2 distance: sqrt((0.4)^2 + (0.3)^2) = sqrt(0.16 + 0.09) = sqrt(0.25) = 0.50
    assert pytest.approx(row["emotion_l2_distance"], rel=1e-5) == 0.50


# ==============================================================================
# 8. MISSING EMOTION HANDLING & EDGE CASES
# ==============================================================================

def test_missing_emotion_and_edge_cases():
    """Verify that missing emotion profiles/vectors produce NaNs and availability=0."""
    meta = pd.DataFrame([
        {"movie_idx": 0, "movie_id": 1, "ml_title": "M 0"},
        {"movie_idx": 1, "movie_id": 2, "ml_title": "M 1 (no emo)"},
    ])
    # Only Movie 0 has an emotion vector
    emo = pd.DataFrame([
        {
            "movie_idx": 0,
            "movie_id": 1,
            "joy": 0.5,
            "trust": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.5,
        }
    ])
    index = MovieEntityIndex.from_dataframes(meta, emotion_df=emo)

    train = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 1000, "positive": 1},
        # User 1 has no positive train interactions
    ])
    user_profiles = build_user_profiles(train, index, total_users=2)

    # Candidate 1 (no emotion) for User 0 (has emotion profile)
    # Candidate 0 (has emotion) for User 1 (no emotion profile)
    cands = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 1, "rank": 1, "score": 0.5},
        {"user_idx": 1, "movie_idx": 0, "rank": 1, "score": 0.5},
    ])

    features = build_candidate_features(cands, user_profiles, index, validate=True)

    # Row 0: User 0 has emotion, Candidate 1 does NOT
    row0 = features.iloc[0]
    assert row0["user_emotion_profile_available"] == 1
    assert row0["candidate_emotion_available"] == 0
    assert np.isnan(row0["emotion_cosine_similarity"])
    assert np.isnan(row0["emotion_dot_product"])
    assert np.isnan(row0["emotion_l1_distance"])
    assert np.isnan(row0["emotion_l2_distance"])
    assert np.isnan(row0["candidate_emotion_joy"])
    assert not np.isnan(row0["user_emotion_joy"])

    # Row 1: User 1 has NO emotion, Candidate 0 DOES
    row1 = features.iloc[1]
    assert row1["user_emotion_profile_available"] == 0
    assert row1["candidate_emotion_available"] == 1
    assert np.isnan(row1["emotion_cosine_similarity"])
    assert np.isnan(row1["user_emotion_joy"])
    assert not np.isnan(row1["candidate_emotion_joy"])


# ==============================================================================
# 9. CANDIDATE INTEGRITY & LEAKAGE ASSERTIONS
# ==============================================================================

def test_candidate_row_preservation_and_ease_score_integrity():
    """Verify that rank, EASE score, and candidate rows are preserved exactly."""
    meta = pd.DataFrame([{"movie_idx": i, "movie_id": i + 1, "ml_title": f"M {i}"} for i in range(5)])
    index = MovieEntityIndex.from_dataframes(meta)
    train = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 1000, "positive": 1}
    ])
    user_profiles = build_user_profiles(train, index, total_users=1)

    cands = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 1, "rank": 1, "score": 0.987654},
        {"user_idx": 0, "movie_idx": 2, "rank": 2, "score": 0.876543},
    ])
    features = build_candidate_features(cands, user_profiles, index, validate=True)

    assert len(features) == 2
    assert features["candidate_rank"].tolist() == [1, 2]
    np.testing.assert_allclose(features["ease_score"].to_numpy(), [0.987654, 0.876543])

    # Trigger leakage assertion: candidate movie 0 was seen in positive train!
    leaky_cands = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rank": 1, "score": 0.999}
    ])
    with pytest.raises(AssertionError, match="CRITICAL LEAKAGE DETECTED"):
        build_candidate_features(leaky_cands, user_profiles, index, validate=True)


# ==============================================================================
# 10. FEATURE GROUP DEFINITIONS AND ABLATION COMPLETENESS
# ==============================================================================

def test_feature_group_definitions_and_ablation_completeness():
    """Verify feature group definitions, disjoint groupings, and complete coverage."""
    # All features in ALL_FEATURES must be unique
    assert len(ALL_FEATURES) == len(set(ALL_FEATURES))

    # All columns == IDENTIFIER_COLUMNS + ALL_FEATURES
    assert ALL_COLUMNS == IDENTIFIER_COLUMNS + ALL_FEATURES

    # Check that each sub-group is a subset of ALL_FEATURES
    for group in [
        BASE_FEATURES,
        KGM_FEATURES,
        KGM_AFFINITY_FEATURES,
        KGM_OVERLAP_FEATURES,
        KGL_SEMANTIC_FEATURES,
        KGL_SEMANTIC_AFFINITY_FEATURES,
        KGL_SEMANTIC_OVERLAP_FEATURES,
        EMOTION_FEATURES,
        EMOTION_COMPATIBILITY_FEATURES,
        CANDIDATE_EMOTION_FEATURES,
        USER_EMOTION_FEATURES,
        OVERLAP_COUNT_FEATURES,
        ALL_KG_FEATURES,
    ]:
        assert set(group).issubset(set(ALL_FEATURES))

    # Check ablation groups:
    # EASE + KG-M
    ease_plus_kgm = IDENTIFIER_COLUMNS + BASE_FEATURES + KGM_FEATURES
    assert len(ease_plus_kgm) == len(set(ease_plus_kgm))

    # EASE + KG-L
    ease_plus_kgl = IDENTIFIER_COLUMNS + BASE_FEATURES + KGL_SEMANTIC_FEATURES
    assert len(ease_plus_kgl) == len(set(ease_plus_kgl))

    # EASE + Emotion
    ease_plus_emo = IDENTIFIER_COLUMNS + BASE_FEATURES + EMOTION_FEATURES
    assert len(ease_plus_emo) == len(set(ease_plus_emo))


# ==============================================================================
# 11. REAL ARTIFACT VERIFICATION TEST (IF PRESENT ON DISK)
# ==============================================================================

def test_real_artifacts_phase4a_end_to_end():
    """Verify production feature table and summary report if already generated."""
    feat_path = Path("data/processed/ml-1m/features/kg_candidate_features.parquet")
    summary_path = Path("data/processed/ml-1m/features/kg_feature_summary.json")

    if not feat_path.exists() or not summary_path.exists():
        pytest.skip("Phase 4A production feature artifacts not yet generated on disk.")

    df_feats = pd.read_parquet(feat_path)
    assert len(df_feats) == 604_000
    assert df_feats["user_idx"].nunique() == 6_040

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["total_rows"] == 604_000
    assert summary["total_users"] == 6_040
    assert summary["leakage_verification"]["zero_positive_train_candidate_leakage"] is True
    assert summary["leakage_verification"]["val_split_used_for_user_profiles"] is False
    assert summary["leakage_verification"]["test_split_accessed"] is False
    assert summary["leakage_verification"]["candidates_per_user_exact_100"] is True
