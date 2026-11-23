"""Unit and Integration Tests for NRC Lexicon Emotion Baseline (Phase 4C).

Verifies:
1. Deterministic text preprocessing and tokenization (Unicode normalization, lowercasing, punctuation stripping).
2. NRC word-to-emotion lookup restricted strictly to the 8 Plutchik emotions (excluding sentiment polarities).
3. 8-dimensional movie NRC vector construction, continuous distribution normalization (sum=1.0), and zero/missing overview handling.
4. Catalog NRC vector computation, NaN representation, and parquet persistence.
5. Strict leakage guards rejecting VAL and TEST interactions/columns in NRC user profiles.
6. Positive-only TRAIN filtering and exponential half-life recency decay.
7. User NRC profile normalization (sum=1.0) and handling of inactive or zero-match users.
8. Candidate-level NRC feature builder:
   - Full 91-column preservation (all 68 Phase 4B columns + 23 Phase 4C columns).
   - Exact row count (604,000) and alignment preservation.
   - Compatibility metrics (cosine, dot, L1, L2) and NaN propagation on missing text/profiles.
9. Clear feature group separation between Qwen and NRC emotion dimensions.
10. Qwen-vs-NRC descriptive comparative analytics.
11. Real Phase 4C artifact verification (if present on disk).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.features.constants import (
    ABLATION_FEATURE_GROUPS,
    CANDIDATE_EMOTION_FEATURES,
    CANDIDATE_NRC_EMOTION_FEATURES,
    FULL_COLUMNS,
    FULL_FEATURES,
    IDENTIFIER_COLUMNS,
    NRC_COVERAGE_FEATURES,
    NRC_EMOTION_COMPATIBILITY_FEATURES,
    NRC_EMOTION_FEATURES,
    PHASE4A_COLUMNS,
    PHASE4B_COLUMNS,
    PHASE4B_FEATURES,
    PLUTCHIK_EMOTIONS,
    QWEN_EMOTION_FEATURES,
    USER_EMOTION_FEATURES,
    USER_NRC_EMOTION_FEATURES,
)
from src.features.nrc_feature_builder import (
    build_nrc_candidate_features,
    validate_nrc_candidate_features,
)
from src.features.nrc_lexicon import NRCLexicon
from src.features.nrc_profiler import (
    NRCUserProfile,
    NRCUserProfileIndex,
    build_nrc_user_profiles,
)
from src.features.nrc_summary import (
    compute_qwen_vs_nrc_comparison,
    generate_nrc_feature_summary,
)


# ==============================================================================
# 1. TEXT PREPROCESSING & TOKENIZATION TESTS
# ==============================================================================

def test_nrc_preprocessing_and_tokenization():
    """Verify Unicode normalization, lowercasing, and regex word extraction."""
    # Accents, diacritics, punctuation, uppercase
    raw_text = "Café's naïve élite hero confronted a fearful, angry monster—123 times!"
    tokens = NRCLexicon.preprocess_text(raw_text)

    # Must be lowercase alphabetic tokens
    assert "cafe" in tokens
    assert "naive" in tokens
    assert "elite" in tokens
    assert "hero" in tokens
    assert "fearful" in tokens
    assert "angry" in tokens
    assert "monster" in tokens
    # Punctuation and numbers must be stripped
    assert "123" not in tokens
    assert "—" not in tokens

    # Edge cases
    assert NRCLexicon.preprocess_text("") == []
    assert NRCLexicon.preprocess_text(None) == []  # type: ignore
    assert NRCLexicon.preprocess_text("   \n\t  ") == []


# ==============================================================================
# 2. NRC LEXICON LOOKUP & EMOTION FILTERING TESTS
# ==============================================================================

def test_nrc_lexicon_lookup_and_emotions():
    """Verify that NRC words map to Plutchik emotions and sentiments are excluded."""
    lexicon = NRCLexicon()

    # Words with known Plutchik associations in NRC EmoLex
    # 'abandon' has 'fear', 'sadness', plus 'negative'
    abandon_emos = lexicon.lexicon.get("abandon", set())
    assert "fear" in abandon_emos
    assert "sadness" in abandon_emos
    # Ensure binary sentiments ('negative', 'positive') are NOT in the Plutchik emotion sets
    assert "negative" not in abandon_emos
    assert "positive" not in abandon_emos

    # All emotions in the cleaned lexicon must belong strictly to PLUTCHIK_EMOTIONS
    plutchik_set = set(PLUTCHIK_EMOTIONS)
    for w, emos in list(lexicon.lexicon.items())[:500]:
        assert emos.issubset(plutchik_set), f"Word '{w}' contains non-Plutchik category: {emos - plutchik_set}"


# ==============================================================================
# 3. MOVIE NRC VECTOR SCORING & NORMALIZATION TESTS
# ==============================================================================

def test_nrc_movie_vector_scoring_and_normalization():
    """Verify 8D count vector scoring, sum-to-1 normalization, and empty text handling."""
    lexicon = NRCLexicon()

    # Text containing clear emotion words
    text = "A joyful and celebratory reunion filled with happiness and trust."
    vec, matched_tokens, total_matches, is_avail = lexicon.score_overview(text)

    assert is_avail
    assert vec is not None
    assert len(vec) == 8
    assert matched_tokens > 0
    assert total_matches > 0
    # Continuous distribution must sum to 1.0
    assert pytest.approx(float(vec.sum()), rel=1e-5) == 1.0
    # Values must be non-negative
    assert (vec >= 0.0).all()

    # Joy and trust should be prominent
    joy_idx = PLUTCHIK_EMOTIONS.index("joy")
    assert vec[joy_idx] > 0.0

    # Text with zero emotion words (stop words only)
    zero_text = "The a this that which and of to in is"
    z_vec, z_matched, z_total, z_avail = lexicon.score_overview(zero_text)
    assert not z_avail
    assert z_vec is None
    assert z_matched == 0
    assert z_total == 0

    # None / Empty overview
    n_vec, n_matched, n_total, n_avail = lexicon.score_overview(None)
    assert not n_avail
    assert n_vec is None
    assert n_matched == 0


# ==============================================================================
# 4. CATALOG NRC VECTOR COMPUTATION & PERSISTENCE TESTS
# ==============================================================================

def test_catalog_nrc_vector_computation_and_persistence(tmp_path: Path):
    """Verify catalog DataFrame construction, NaN representations, and serialization."""
    lexicon = NRCLexicon()

    metadata = pd.DataFrame([
        {"movie_idx": 0, "overview": "A heroic warrior fights an evil monster in a terrifying war."},
        {"movie_idx": 1, "overview": "The of and that which."},  # zero NRC matches
        {"movie_idx": 2, "overview": None},  # missing overview
    ])

    catalog_df = lexicon.compute_catalog_vectors(metadata, catalog_size=3)

    assert len(catalog_df) == 3
    assert set(catalog_df["movie_idx"]) == {0, 1, 2}

    # Movie 0: valid overview + valid NRC
    r0 = catalog_df.iloc[0]
    assert r0["candidate_has_overview"] == 1
    assert r0["candidate_nrc_available"] == 1
    assert r0["candidate_nrc_matched_token_count"] > 0
    assert not pd.isna(r0["candidate_nrc_fear"])
    # Sum of candidate 8D emotions should equal 1.0
    emo_sum = sum(r0[f"candidate_nrc_{e}"] for e in PLUTCHIK_EMOTIONS)
    assert pytest.approx(emo_sum, rel=1e-5) == 1.0

    # Movie 1: has overview, but 0 NRC matches
    r1 = catalog_df.iloc[1]
    assert r1["candidate_has_overview"] == 1
    assert r1["candidate_nrc_available"] == 0
    assert r1["candidate_nrc_matched_token_count"] == 0
    assert pd.isna(r1["candidate_nrc_fear"])

    # Movie 2: missing overview
    r2 = catalog_df.iloc[2]
    assert r2["candidate_has_overview"] == 0
    assert r2["candidate_nrc_available"] == 0
    assert pd.isna(r2["candidate_nrc_fear"])

    # Save and Load roundtrip
    saved_file = NRCLexicon.save_vectors(catalog_df, tmp_path / "nrc")
    assert saved_file.exists()

    loaded_df = NRCLexicon.load_vectors(tmp_path / "nrc")
    assert len(loaded_df) == 3
    assert list(loaded_df.columns) == list(catalog_df.columns)
    np.testing.assert_array_equal(loaded_df["candidate_nrc_available"], catalog_df["candidate_nrc_available"])


# ==============================================================================
# 5. LEAKAGE-SAFE NRC USER PROFILER TESTS
# ==============================================================================

def test_nrc_user_profiler_leakage_guards():
    """Verify that train_df containing VAL or TEST columns is strictly rejected."""
    # Synthetic movie NRC df
    movie_df = pd.DataFrame([{"movie_idx": 0, "candidate_nrc_available": 1, **{f"candidate_nrc_{e}": 0.125 for e in PLUTCHIK_EMOTIONS}}])

    # Forbidden 'val' column
    df_val = pd.DataFrame([{"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 100, "val": 1}])
    with pytest.raises(ValueError, match="Data leakage detected"):
        build_nrc_user_profiles(df_val, movie_df, total_users=1)

    # Forbidden 'test' column
    df_test = pd.DataFrame([{"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 100, "is_test": True}])
    with pytest.raises(ValueError, match="Data leakage detected"):
        build_nrc_user_profiles(df_test, movie_df, total_users=1)


def test_nrc_user_profiler_recency_and_normalization():
    """Verify positive-only filtering, recency decay weighting, and distribution sum = 1.0."""
    # 3 movies with distinct NRC vectors
    movie_data = []
    # Movie 0: pure joy (1.0 joy, 0 others)
    v0 = {f"candidate_nrc_{e}": (1.0 if e == "joy" else 0.0) for e in PLUTCHIK_EMOTIONS}
    movie_data.append({"movie_idx": 0, "candidate_nrc_available": 1, **v0})

    # Movie 1: pure fear (1.0 fear, 0 others)
    v1 = {f"candidate_nrc_{e}": (1.0 if e == "fear" else 0.0) for e in PLUTCHIK_EMOTIONS}
    movie_data.append({"movie_idx": 1, "candidate_nrc_available": 1, **v1})

    # Movie 2: missing NRC vector (unavailable)
    v2 = {f"candidate_nrc_{e}": np.nan for e in PLUTCHIK_EMOTIONS}
    movie_data.append({"movie_idx": 2, "candidate_nrc_available": 0, **v2})

    movie_df = pd.DataFrame(movie_data)

    # User 0: positive on movie 0 (old) and movie 1 (recent), negative on movie 0 (recent rating 2.0)
    # User 1: positive ONLY on movie 2 (which has no NRC vector)
    # User 2: inactive (no interactions)
    train_df = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 1000},  # positive, old
        {"user_idx": 0, "movie_idx": 1, "rating": 4.0, "timestamp": 2000},  # positive, recent (max_ts)
        {"user_idx": 0, "movie_idx": 2, "rating": 2.0, "timestamp": 2000},  # negative -> excluded
        {"user_idx": 1, "movie_idx": 2, "rating": 5.0, "timestamp": 1000},  # positive on missing NRC movie
    ])

    user_index = build_nrc_user_profiles(
        train_df=train_df,
        movie_nrc_df=movie_df,
        half_life_days=365.0,
        positive_threshold=4.0,
        total_users=3,
    )

    assert len(user_index) == 3

    # User 0
    p0 = user_index.get(0)
    assert p0.nrc_available
    assert p0.nrc_profile is not None
    assert pytest.approx(float(p0.nrc_profile.sum()), rel=1e-5) == 1.0
    joy_idx = PLUTCHIK_EMOTIONS.index("joy")
    fear_idx = PLUTCHIK_EMOTIONS.index("fear")
    # Because movie 1 is more recent (timestamp 2000 vs 1000), fear weight > joy weight
    assert p0.nrc_profile[fear_idx] > p0.nrc_profile[joy_idx]

    # User 1: positive movie has no NRC vector -> unavailable
    p1 = user_index.get(1)
    assert not p1.nrc_available
    assert p1.nrc_profile is None

    # User 2: inactive -> unavailable
    p2 = user_index.get(2)
    assert not p2.nrc_available
    assert p2.nrc_profile is None


# ==============================================================================
# 6. NRC CANDIDATE FEATURE BUILDER TESTS
# ==============================================================================

def test_build_nrc_candidate_features_and_preservation():
    """Verify appending 23 NRC features, dot/cosine equivalence, and Phase 4B preservation."""
    # Synthetic Phase 4B candidate DataFrame (2 users, 2 candidates each = 4 rows, 68 columns)
    rows = []
    for u in [0, 1]:
        for rank, m in enumerate([10, 20]):
            r = {
                "user_idx": u,
                "movie_idx": m,
                "candidate_rank": rank + 1,
                "ease_score": 1.2 - rank * 0.4,
                "candidate_seen_in_positive_train": 0,
            }
            for col in PHASE4B_COLUMNS:
                if col not in r:
                    r[col] = 0.5
            rows.append(r)

    p4b_df = pd.DataFrame(rows)

    # Movie NRC DataFrame
    # Movie 10: valid NRC vector (joy 0.5, trust 0.5)
    v10 = {f"candidate_nrc_{e}": (0.5 if e in ["joy", "trust"] else 0.0) for e in PLUTCHIK_EMOTIONS}
    # Movie 20: missing NRC vector
    v20 = {f"candidate_nrc_{e}": np.nan for e in PLUTCHIK_EMOTIONS}

    movie_nrc_df = pd.DataFrame([
        {"movie_idx": 10, "candidate_nrc_available": 1, "candidate_nrc_matched_token_count": 5, **v10},
        {"movie_idx": 20, "candidate_nrc_available": 0, "candidate_nrc_matched_token_count": 0, **v20},
    ])

    # User Profiles:
    # User 0 has valid NRC profile (joy 0.5, trust 0.5)
    u0_vec = np.array([0.5 if e in ["joy", "trust"] else 0.0 for e in PLUTCHIK_EMOTIONS], dtype=np.float32)
    u0_prof = NRCUserProfile(user_idx=0, nrc_profile=u0_vec, nrc_available=True, positive_train_count=2, nrc_movie_count=1)
    # User 1 has no profile
    u1_prof = NRCUserProfile(user_idx=1, nrc_profile=None, nrc_available=False, positive_train_count=0, nrc_movie_count=0)

    user_index = NRCUserProfileIndex({0: u0_prof, 1: u1_prof}, total_users=2)

    # Build features
    full_df = build_nrc_candidate_features(
        phase4b_df=p4b_df,
        nrc_user_profiles=user_index,
        movie_nrc_df=movie_nrc_df,
        validate=False,  # custom row count, not 604,000
    )

    # 1. Total row count preserved
    assert len(full_df) == 4
    # 2. Total columns = 68 + 23 = 91
    assert len(full_df.columns) == len(FULL_COLUMNS)
    for col in FULL_COLUMNS:
        assert col in full_df.columns

    # 3. Check Row 0 (User 0, Movie 10: both available, identical vectors)
    r0 = full_df.iloc[0]
    assert r0["candidate_nrc_available"] == 1
    assert r0["user_nrc_profile_available"] == 1
    assert r0["candidate_nrc_matched_token_count"] == 5
    assert pytest.approx(r0["nrc_emotion_cosine_similarity"], abs=1e-5) == 1.0
    assert pytest.approx(r0["nrc_emotion_l1_distance"], abs=1e-5) == 0.0
    assert pytest.approx(r0["nrc_emotion_l2_distance"], abs=1e-5) == 0.0

    # 4. Check Row 1 (User 0, Movie 20: movie has no NRC vector)
    r1 = full_df.iloc[1]
    assert r1["candidate_nrc_available"] == 0
    assert r1["user_nrc_profile_available"] == 1
    assert pd.isna(r1["candidate_nrc_joy"])
    assert pd.isna(r1["nrc_emotion_cosine_similarity"])
    assert pd.isna(r1["nrc_emotion_l1_distance"])

    # 5. Check Row 2 (User 1, Movie 10: user has no profile)
    r2 = full_df.iloc[2]
    assert r2["candidate_nrc_available"] == 1
    assert r2["user_nrc_profile_available"] == 0
    assert not pd.isna(r2["candidate_nrc_joy"])  # candidate has NRC
    assert pd.isna(r2["user_nrc_joy"])  # user has no profile
    assert pd.isna(r2["nrc_emotion_cosine_similarity"])


# ==============================================================================
# 7. FEATURE GROUP SEPARATION & CONSTANTS TESTS
# ==============================================================================

def test_feature_group_separation_and_ablation_constants():
    """Verify strict separation between Qwen and NRC feature groups and ablation completeness."""
    # Non-overlapping assertion between Qwen and NRC emotion feature groups
    qwen_set = set(QWEN_EMOTION_FEATURES)
    nrc_set = set(NRC_EMOTION_FEATURES)
    assert len(qwen_set.intersection(nrc_set)) == 0, "Qwen and NRC features must NOT overlap!"

    # Candidate emotion dimensions separation
    qwen_cand = set(CANDIDATE_EMOTION_FEATURES)
    nrc_cand = set(CANDIDATE_NRC_EMOTION_FEATURES)
    assert len(qwen_cand.intersection(nrc_cand)) == 0

    # Column counts
    assert len(PHASE4B_COLUMNS) == 68
    assert len(FULL_COLUMNS) == 91
    assert len(NRC_EMOTION_FEATURES) == 23

    # Ablation groups dictionary check
    expected_groups = {"BASE", "KGM", "KGL_SEMANTIC", "QWEN_EMOTION", "NRC_EMOTION", "TEXT", "ALL"}
    assert set(ABLATION_FEATURE_GROUPS.keys()) == expected_groups
    assert ABLATION_FEATURE_GROUPS["NRC_EMOTION"] == NRC_EMOTION_FEATURES
    assert ABLATION_FEATURE_GROUPS["QWEN_EMOTION"] == QWEN_EMOTION_FEATURES
    assert ABLATION_FEATURE_GROUPS["ALL"] == FULL_FEATURES


# ==============================================================================
# 8. QWEN VS NRC COMPARATIVE ANALYTICS TESTS
# ==============================================================================

def test_qwen_vs_nrc_comparison_computation(tmp_path: Path):
    """Verify descriptive statistics computation between movie-level Qwen and NRC vectors."""
    # Synthetic Qwen vectors parquet
    qwen_df = pd.DataFrame([
        {"movie_idx": 0, "joy": 0.8, "trust": 0.4, "fear": 0.0, "surprise": 0.1, "sadness": 0.0, "disgust": 0.0, "anger": 0.0, "anticipation": 0.2},
        {"movie_idx": 1, "joy": 0.0, "trust": 0.0, "fear": 0.7, "surprise": 0.3, "sadness": 0.6, "disgust": 0.2, "anger": 0.5, "anticipation": 0.1},
    ])
    qwen_path = tmp_path / "qwen_emotions.parquet"
    qwen_df.to_parquet(qwen_path, index=False)

    # Synthetic Movie NRC DataFrame
    # Movie 0 has matching high joy
    nrc_df = pd.DataFrame([
        {"movie_idx": 0, "candidate_nrc_available": 1, "candidate_nrc_joy": 0.7, "candidate_nrc_trust": 0.3, "candidate_nrc_fear": 0.0, "candidate_nrc_surprise": 0.0, "candidate_nrc_sadness": 0.0, "candidate_nrc_disgust": 0.0, "candidate_nrc_anger": 0.0, "candidate_nrc_anticipation": 0.0},
        {"movie_idx": 1, "candidate_nrc_available": 1, "candidate_nrc_joy": 0.0, "candidate_nrc_trust": 0.0, "candidate_nrc_fear": 0.5, "candidate_nrc_surprise": 0.1, "candidate_nrc_sadness": 0.4, "candidate_nrc_disgust": 0.0, "candidate_nrc_anger": 0.0, "candidate_nrc_anticipation": 0.0},
    ])

    comp = compute_qwen_vs_nrc_comparison(nrc_df, qwen_vectors_path=qwen_path)
    assert comp["common_movies_count"] == 2
    assert "average_cosine_similarity" in comp
    assert comp["average_cosine_similarity"] > 0.8  # high similarity on these synthetic cases
    assert "joy" in comp["per_emotion_comparison"]
    assert "nrc_mean" in comp["per_emotion_comparison"]["joy"]
    assert "qwen_mean" in comp["per_emotion_comparison"]["joy"]


# ==============================================================================
# 9. REAL PHASE 4C ARTIFACT VERIFICATION (IF ON DISK)
# ==============================================================================

def test_real_phase4c_artifacts_if_present():
    """Verify integrity of real Phase 4C feature artifacts if generated."""
    feature_path = Path("data/processed/ml-1m/features/full_candidate_features.parquet")
    summary_path = Path("data/processed/ml-1m/features/nrc_feature_summary.json")

    if not feature_path.exists() or not summary_path.exists():
        pytest.skip("Phase 4C production feature artifacts not yet generated on disk.")

    # 1. Summary verification
    with open(summary_path, "r") as f:
        summary = json.load(f)

    assert summary["total_rows"] == 604_000
    assert summary["total_users"] == 6_040
    assert summary["leakage_verification"]["zero_positive_train_candidate_leakage"]
    assert not summary["leakage_verification"]["val_split_used_for_user_profiles"]
    assert not summary["leakage_verification"]["test_split_accessed"]

    # 2. DataFrame schema and row verification
    df = pd.read_parquet(feature_path)
    assert len(df) == 604_000
    assert len(df.columns) == 91
    assert list(df.columns) == FULL_COLUMNS

    # Zero leakage check
    assert int(df["candidate_seen_in_positive_train"].sum()) == 0

    # Sum check for candidate NRC vectors where available
    valid_cands = df[df["candidate_nrc_available"] == 1]
    cand_cols = [f"candidate_nrc_{e}" for e in PLUTCHIK_EMOTIONS]
    cand_sums = valid_cands[cand_cols].sum(axis=1)
    np.testing.assert_allclose(cand_sums.to_numpy(), 1.0, atol=1e-4)

    # Sum check for user NRC profiles where available
    valid_users = df[df["user_nrc_profile_available"] == 1]
    user_cols = [f"user_nrc_{e}" for e in PLUTCHIK_EMOTIONS]
    user_sums = valid_users[user_cols].sum(axis=1)
    np.testing.assert_allclose(user_sums.to_numpy(), 1.0, atol=1e-4)
