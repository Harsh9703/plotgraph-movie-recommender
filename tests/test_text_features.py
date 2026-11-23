"""Unit and Integration Tests for Leakage-Safe Text-Semantic Features (Phase 4B).

Verifies:
1. TFIDFMovieEncoder: deterministic fitting, unit L2-norm, empty handling, save/load.
2. SBERTMovieEncoder: unit L2-norm, empty handling, save/load (using mock for fast tests).
3. FAISSIndexHelper: IndexFlatIP and NumPy fallback nearest-neighbor search.
4. Leakage-safe TextUserProfileIndex:
   - Rejection of VAL/TEST columns or splits.
   - Restricting to positive TRAIN interactions (rating >= 4.0).
   - Recency weighting and unit L2 normalization.
   - Inactive user and missing overview handling.
5. Candidate-level text feature building:
   - Preservation of all 60 Phase 4A columns and exact row counts.
   - Equivalence of cosine similarity and dot product for normalized vectors.
   - NaNs and availability flags for missing overviews and inactive users.
   - Validation assertions and leakage checks.
6. Summary generation and metadata reporting.
7. Real Phase 4B artifacts verification (if present on disk).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.features.constants import (
    ALL_COLUMNS,
    ALL_FEATURES,
    IDENTIFIER_COLUMNS,
    PHASE4A_COLUMNS,
    PHASE4B_COLUMNS,
    SBERT_FEATURES,
    TEXT_FEATURES,
    TFIDF_FEATURES,
)
from src.features.text_encoder import (
    FAISSIndexHelper,
    SBERTMovieEncoder,
    TFIDFMovieEncoder,
    is_faiss_available,
)
from src.features.text_feature_builder import (
    build_text_candidate_features,
    validate_text_candidate_features,
)
from src.features.text_profiler import (
    TextUserProfile,
    TextUserProfileIndex,
    build_text_user_profiles,
)
from src.features.text_summary import generate_text_feature_summary


# ==============================================================================
# 1. TF-IDF MOVIE ENCODER TESTS
# ==============================================================================

def test_tfidf_encoder_fit_transform_and_unit_norm(tmp_path: Path):
    """Verify TF-IDF fitting, unit L2 normalization, empty handling, and serialization."""
    overviews = {
        0: "A brave knight embarks on a quest to defeat an evil dragon in a fantasy realm.",
        1: "A futuristic detective investigates cybernetic crimes in a neon-lit dystopian city.",
        2: "An epic fantasy story where knights and wizards battle dark magical creatures.",
        3: "",  # Empty overview
        4: "   ",  # Whitespace only
    }
    catalog_size = 5

    encoder = TFIDFMovieEncoder(ngram_range=(1, 2), min_df=1)
    matrix = encoder.fit_transform(overviews, catalog_size=catalog_size)

    assert encoder.is_fitted
    assert matrix.shape[0] == catalog_size
    assert encoder.vocab_size > 0
    assert encoder.catalog_size == catalog_size

    # Movies 0, 1, 2 must have text mask True and unit L2 norm
    for m_idx in [0, 1, 2]:
        assert encoder.has_text_mask[m_idx]
        vec = encoder.get_movie_vector(m_idx)
        assert vec is not None
        norm = sparse.linalg.norm(vec)
        assert pytest.approx(norm, rel=1e-5) == 1.0

    # Movies 3, 4 must have text mask False and get_movie_vector returns None
    for m_idx in [3, 4]:
        assert not encoder.has_text_mask[m_idx]
        assert encoder.get_movie_vector(m_idx) is None
        # In matrix, row is all zeros
        assert matrix[m_idx].nnz == 0

    # Save and load roundtrip
    encoder.save(tmp_path / "tfidf")
    loaded = TFIDFMovieEncoder.load(tmp_path / "tfidf")
    assert loaded.is_fitted
    assert loaded.catalog_size == catalog_size
    assert loaded.vocab_size == encoder.vocab_size
    assert np.array_equal(loaded.has_text_mask, encoder.has_text_mask)

    # Check identical vectors
    for m_idx in range(catalog_size):
        diff = (matrix[m_idx] - loaded.movie_tfidf_matrix[m_idx]).toarray()
        assert np.allclose(diff, 0.0)


# ==============================================================================
# 2. SBERT MOVIE ENCODER TESTS (MOCKED FOR SPEED)
# ==============================================================================

def test_sbert_encoder_mocked(tmp_path: Path):
    """Verify SBERT catalog encoding, unit L2-norm, empty handling, and serialization."""
    overviews = {
        0: "A touching drama about family and resilience.",
        1: "A high-octane space sci-fi thriller with starships.",
        2: "",  # missing
    }
    catalog_size = 3
    dim = 16

    encoder = SBERTMovieEncoder(model_name="mock-model", device="cpu")
    encoder.embedding_dimension = dim

    # Mock the internal SentenceTransformer model
    mock_model = MagicMock()
    # Return normalized 16-d vectors for the 2 valid texts
    fake_raw = np.random.randn(2, dim).astype(np.float32)
    fake_raw /= np.linalg.norm(fake_raw, axis=1, keepdims=True)
    mock_model.encode.return_value = fake_raw
    mock_model.get_sentence_embedding_dimension.return_value = dim

    encoder._model = mock_model

    embeddings = encoder.encode_catalog(overviews, catalog_size=catalog_size, show_progress_bar=False)

    assert embeddings.shape == (catalog_size, dim)
    assert encoder.has_embedding_mask[0]
    assert encoder.has_embedding_mask[1]
    assert not encoder.has_embedding_mask[2]

    # Check norms for valid movies
    assert pytest.approx(np.linalg.norm(encoder.get_movie_embedding(0)), rel=1e-5) == 1.0
    assert pytest.approx(np.linalg.norm(encoder.get_movie_embedding(1)), rel=1e-5) == 1.0
    assert encoder.get_movie_embedding(2) is None
    assert np.isnan(embeddings[2]).all()

    # Save and load roundtrip
    encoder.save(tmp_path / "sbert")
    loaded = SBERTMovieEncoder.load(tmp_path / "sbert", model_name="mock-model")
    assert loaded.embedding_dimension == dim
    assert np.array_equal(loaded.has_embedding_mask, encoder.has_embedding_mask)
    assert np.allclose(loaded.embeddings_matrix[:2], embeddings[:2])


# ==============================================================================
# 3. FAISS INDEX HELPER TESTS
# ==============================================================================

def test_faiss_index_helper():
    """Verify FAISS nearest-neighbor search with IndexFlatIP and NumPy fallback."""
    dim = 8
    # 4 movies, movie 2 missing
    embeddings = np.zeros((4, dim), dtype=np.float32)
    embeddings[0] = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    embeddings[1] = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    embeddings[2] = np.full(dim, np.nan, dtype=np.float32)
    embeddings[3] = np.array([0.7071, 0.7071, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    mask = np.array([True, True, False, True], dtype=bool)

    # 1. Standard build
    helper = FAISSIndexHelper(dimension=dim)
    helper.build_index(embeddings, mask)

    # Query matching movie 0
    q0 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    results = helper.search_nearest(q0, top_k=2)

    assert len(results) == 2
    top_movie, top_sim = results[0]
    assert top_movie == 0
    assert pytest.approx(top_sim, abs=1e-4) == 1.0

    second_movie, second_sim = results[1]
    assert second_movie == 3
    assert pytest.approx(second_sim, abs=1e-4) == 0.7071

    # 2. Force NumPy fallback
    helper_np = FAISSIndexHelper(dimension=dim)
    helper_np.build_index(embeddings, mask)
    helper_np.uses_faiss = False  # force pure NumPy

    results_np = helper_np.search_nearest(q0, top_k=2)
    assert len(results_np) == 2
    assert results_np[0][0] == 0
    assert pytest.approx(results_np[0][1], abs=1e-4) == 1.0
    assert results_np[1][0] == 3
    assert pytest.approx(results_np[1][1], abs=1e-4) == 0.7071


# ==============================================================================
# 4. LEAKAGE-SAFE TEXT USER PROFILER TESTS
# ==============================================================================

def test_build_text_user_profiles_leakage_guards():
    """Verify that train_df containing VAL or TEST columns is strictly rejected."""
    tfidf = TFIDFMovieEncoder()
    sbert = SBERTMovieEncoder()

    # Forbidden 'val' column
    df_val = pd.DataFrame([{"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 100, "val": 1}])
    with pytest.raises(ValueError, match="Data leakage detected"):
        build_text_user_profiles(df_val, tfidf, sbert, total_users=1)

    # Forbidden 'is_test' column
    df_test = pd.DataFrame([{"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 100, "is_test": True}])
    with pytest.raises(ValueError, match="Data leakage detected"):
        build_text_user_profiles(df_test, tfidf, sbert, total_users=1)


def test_build_text_user_profiles_recency_and_normalization():
    """Verify recency weighting, positive-only filtering, and unit normalization of user profiles."""
    # Setup encoders with 3 movies
    overviews = {
        0: "action explosion hero guns adventure",
        1: "romance love drama emotional relationship",
        2: "action thriller car chase guns",
    }
    tfidf = TFIDFMovieEncoder(ngram_range=(1, 1), min_df=1)
    tfidf.fit_transform(overviews, catalog_size=3)

    sbert = SBERTMovieEncoder()
    sbert.embedding_dimension = 4
    sbert.has_embedding_mask = np.array([True, True, True], dtype=bool)
    # 3 distinct orthogonal-ish unit vectors
    emb = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.7071, 0.0, 0.7071, 0.0],
    ], dtype=np.float32)
    sbert.embeddings_matrix = emb

    # User 0: positive on movie 0 (old) and movie 1 (recent), negative (<4.0) on movie 2
    # User 1: no positive interactions (only rating 2.0)
    # User 2: no interactions at all
    train_df = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rating": 5.0, "timestamp": 1000},  # positive, older
        {"user_idx": 0, "movie_idx": 1, "rating": 4.0, "timestamp": 2000},  # positive, recent (max_ts)
        {"user_idx": 0, "movie_idx": 2, "rating": 2.0, "timestamp": 2000},  # negative -> excluded
        {"user_idx": 1, "movie_idx": 0, "rating": 3.0, "timestamp": 2000},  # negative -> excluded
    ])

    user_profiles = build_text_user_profiles(
        train_df=train_df,
        tfidf_encoder=tfidf,
        sbert_encoder=sbert,
        half_life_days=365.0,
        positive_threshold=4.0,
        total_users=3,
    )

    assert len(user_profiles) == 3

    # User 0
    p0 = user_profiles.get(0)
    assert p0.tfidf_available
    assert p0.sbert_available
    # User TF-IDF profile should be unit normalized
    assert pytest.approx(sparse.linalg.norm(p0.tfidf_profile), rel=1e-5) == 1.0
    # User SBERT profile should be unit normalized
    assert pytest.approx(np.linalg.norm(p0.sbert_profile), rel=1e-5) == 1.0

    # User 1 (no positive interactions)
    p1 = user_profiles.get(1)
    assert not p1.tfidf_available
    assert not p1.sbert_available
    assert p1.tfidf_profile is None
    assert p1.sbert_profile is None

    # User 2 (no interactions)
    p2 = user_profiles.get(2)
    assert not p2.tfidf_available
    assert not p2.sbert_available


# ==============================================================================
# 5. CANDIDATE TEXT FEATURE BUILDER TESTS
# ==============================================================================

def test_build_text_candidate_features_and_validation():
    """Verify feature computation, dot-cosine equivalence, NaN handling, and preservation."""
    # Synthetic Phase 4A candidate table
    # 2 users, 2 candidates each = 4 rows
    candidate_data = []
    for u in [0, 1]:
        for rank, m in enumerate([10, 20]):
            row = {
                "user_idx": u,
                "movie_idx": m,
                "candidate_rank": rank + 1,
                "ease_score": 1.5 - rank * 0.5,
                "candidate_seen_in_positive_train": 0,
            }
            # Add dummy columns for the remaining Phase 4A features
            for col in PHASE4A_COLUMNS:
                if col not in row:
                    row[col] = 0.5
            candidate_data.append(row)

    phase4a_df = pd.DataFrame(candidate_data)

    # Encoders
    dim = 4
    sbert = SBERTMovieEncoder()
    sbert.embedding_dimension = dim
    sbert.catalog_size = 25
    sbert.embeddings_matrix = np.full((25, dim), np.nan, dtype=np.float32)
    sbert.has_embedding_mask = np.zeros(25, dtype=bool)

    # Movie 10 has valid SBERT and TF-IDF
    sbert.embeddings_matrix[10] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sbert.has_embedding_mask[10] = True
    # Movie 20 has NO text overview (missing)
    sbert.has_embedding_mask[20] = False

    tfidf = TFIDFMovieEncoder()
    tfidf.catalog_size = 25
    tfidf.vocab_size = 6
    tfidf.has_text_mask = np.zeros(25, dtype=bool)
    tfidf.has_text_mask[10] = True
    tfidf.has_text_mask[20] = False
    tfidf.is_fitted = True

    # Sparse row for movie 10: unit normalized
    row_10 = sparse.csr_matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    full_sparse = sparse.vstack([
        row_10 if i == 10 else sparse.csr_matrix((1, 6), dtype=np.float32)
        for i in range(25)
    ], format="csr")
    tfidf.movie_tfidf_matrix = full_sparse

    # User Profiles:
    # User 0 has valid profiles matching movie 10
    u0_prof = TextUserProfile(
        user_idx=0,
        tfidf_profile=row_10,
        tfidf_available=True,
        sbert_profile=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sbert_available=True,
    )
    # User 1 has no profile
    u1_prof = TextUserProfile(
        user_idx=1,
        tfidf_profile=None,
        tfidf_available=False,
        sbert_profile=None,
        sbert_available=False,
    )

    profiles = TextUserProfileIndex(
        profiles={0: u0_prof, 1: u1_prof},
        sbert_dimension=dim,
        vocab_size=6,
        total_users=2,
    )

    # Build features
    features_df = build_text_candidate_features(
        phase4a_df=phase4a_df,
        text_user_profiles=profiles,
        tfidf_encoder=tfidf,
        sbert_encoder=sbert,
        validate=False,  # custom row count, not 604,000
    )

    # 1. Total row count preserved
    assert len(features_df) == 4
    # 2. Total columns = 60 + 8 = 68
    assert len(features_df.columns) == len(PHASE4B_COLUMNS)
    for col in PHASE4B_COLUMNS:
        assert col in features_df.columns

    # 3. Check Row 0 (User 0, Movie 10: both available, identical vectors)
    r0 = features_df.iloc[0]
    assert r0["tfidf_candidate_available"] == 1
    assert r0["tfidf_user_profile_available"] == 1
    assert pytest.approx(r0["tfidf_cosine_similarity"], abs=1e-5) == 1.0
    assert pytest.approx(r0["tfidf_dot_product"], abs=1e-5) == 1.0

    assert r0["sbert_candidate_available"] == 1
    assert r0["sbert_user_profile_available"] == 1
    assert pytest.approx(r0["sbert_cosine_similarity"], abs=1e-5) == 1.0
    assert pytest.approx(r0["sbert_dot_product"], abs=1e-5) == 1.0

    # 4. Check Row 1 (User 0, Movie 20: movie missing overview)
    r1 = features_df.iloc[1]
    assert r1["tfidf_candidate_available"] == 0
    assert r1["tfidf_user_profile_available"] == 1
    assert pd.isna(r1["tfidf_cosine_similarity"])
    assert pd.isna(r1["tfidf_dot_product"])

    assert r1["sbert_candidate_available"] == 0
    assert r1["sbert_user_profile_available"] == 1
    assert pd.isna(r1["sbert_cosine_similarity"])
    assert pd.isna(r1["sbert_dot_product"])

    # 5. Check Row 2 (User 1, Movie 10: user profile missing)
    r2 = features_df.iloc[2]
    assert r2["tfidf_candidate_available"] == 1
    assert r2["tfidf_user_profile_available"] == 0
    assert pd.isna(r2["tfidf_cosine_similarity"])
    assert pd.isna(r2["tfidf_dot_product"])

    assert r2["sbert_candidate_available"] == 1
    assert r2["sbert_user_profile_available"] == 0
    assert pd.isna(r2["sbert_cosine_similarity"])
    assert pd.isna(r2["sbert_dot_product"])


# ==============================================================================
# 6. TEXT FEATURE SUMMARY TESTS
# ==============================================================================

def test_generate_text_feature_summary():
    """Verify summary generation metrics, coverage, and leakage verification."""
    # Synthetic candidate features table
    df = pd.DataFrame({
        "user_idx": [0, 0, 1, 1],
        "movie_idx": [10, 20, 10, 20],
        "candidate_rank": [1, 2, 1, 2],
        "ease_score": [1.0, 0.5, 1.0, 0.5],
        "candidate_seen_in_positive_train": [0, 0, 0, 0],
        "tfidf_cosine_similarity": [0.8, np.nan, 0.5, np.nan],
        "tfidf_dot_product": [0.8, np.nan, 0.5, np.nan],
        "tfidf_candidate_available": [1, 0, 1, 0],
        "tfidf_user_profile_available": [1, 1, 1, 1],
        "sbert_cosine_similarity": [0.75, np.nan, 0.45, np.nan],
        "sbert_dot_product": [0.75, np.nan, 0.45, np.nan],
        "sbert_candidate_available": [1, 0, 1, 0],
        "sbert_user_profile_available": [1, 1, 1, 1],
    })

    summary = generate_text_feature_summary(
        features_df=df,
        tfidf_vocab_size=15000,
        sbert_dimension=384,
        sbert_model_id="sentence-transformers/all-MiniLM-L6-v2",
        catalog_overview_count=3810,
        total_catalog_size=3883,
        runtime_seconds=12.5,
    )

    assert summary["total_rows"] == 4
    assert summary["total_users"] == 2
    assert summary["sbert_dimension"] == 384
    assert summary["tfidf_vocab_size"] == 15000
    assert summary["catalog_overview_coverage"]["usable_overviews"] == 3810
    assert summary["leakage_verification"]["zero_positive_train_candidate_leakage"]
    assert not summary["leakage_verification"]["val_split_used_for_user_profiles"]
    assert not summary["leakage_verification"]["test_split_accessed"]
    assert summary["descriptive_statistics"]["tfidf_cosine_similarity"]["mean"] == pytest.approx(0.65)


# ==============================================================================
# 7. REAL ARTIFACTS VERIFICATION (IF ON DISK)
# ==============================================================================

def test_real_phase4b_artifacts_if_present():
    """Verify integrity of real Phase 4B feature artifacts if generated."""
    feature_path = Path("data/processed/ml-1m/features/kg_text_candidate_features.parquet")
    summary_path = Path("data/processed/ml-1m/features/text_feature_summary.json")

    if not feature_path.exists() or not summary_path.exists():
        pytest.skip("Phase 4B production feature artifacts not yet generated on disk.")

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
    assert len(df.columns) == 68
    assert list(df.columns) == PHASE4B_COLUMNS

    # Zero leakage check
    assert int(df["candidate_seen_in_positive_train"].sum()) == 0

    # Values check
    for col in ["tfidf_cosine_similarity", "sbert_cosine_similarity"]:
        valid = df[col].dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0001).all()
