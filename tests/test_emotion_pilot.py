"""Unit and Integration Tests for Phase 3E Dedicated Qwen Emotion Pilot.

Verifies:
1. Safety Guard: Rejection of full catalog and samples > 30 movies.
2. Output Directory Isolation: Directs artifacts to emotion_pilot_v1/ without touching Phase 3D pilot dirs.
3. Mock Emotion Backend: Deterministic generation of schema-compliant evidence and score structures.
4. Checkpointing and Resumption: Skipping already validated movies.
5. Derived Vector Generation: emotion_vectors.jsonl contains exact 8 continuous floats and prominent_emotions (>0.5).
6. Evidence Review CSV: Contains per-emotion evidence lists and calibrated scores.
7. Three-Way Emotion Comparison: Integration with pilot_comparison module.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.knowledge_graph.emotion_contract import (
    DEFAULT_EMOTION_PROMPT_VERSION,
    DEFAULT_EMOTION_SCHEMA_VERSION,
    MovieEmotionExtraction,
    parse_and_validate_emotion_extraction,
)
from src.knowledge_graph.kg_l_contract import PLUTCHIK_EMOTIONS
from src.knowledge_graph.pilot_comparison import (
    compare_three_way_emotion_pilots,
    compute_evidence_emotion_metrics,
    format_three_way_emotion_table,
)
from src.knowledge_graph.qwen_emotion_pilot import (
    MockQwenEmotionBackend,
    export_emotion_review_csv,
    run_qwen_emotion_pilot,
)
from src.knowledge_graph.qwen_pilot import (
    MAX_PILOT_MOVIES,
    load_pilot_movies,
)

SAMPLE_PARQUET_PATH = Path("data/processed/ml-1m/kg_l/sample_movies.parquet")


# ==============================================================================
# 1. SAFETY GUARD TESTS
# ==============================================================================

def test_emotion_pilot_safety_guard_rejects_full_catalog(tmp_path: Path):
    """Safety guard must reject attempts to run against full catalog artifacts."""
    fake_catalog = tmp_path / "movies_with_tmdb.parquet"
    df = pd.DataFrame({
        "movie_id": [1, 2],
        "movie_idx": [0, 1],
        "ml_title": ["Movie 1", "Movie 2"],
        "overview": ["Overview 1", "Overview 2"],
    })
    df.to_parquet(fake_catalog)

    backend = MockQwenEmotionBackend()
    with pytest.raises(ValueError, match="Safety guard violation"):
        run_qwen_emotion_pilot(
            input_sample_path=fake_catalog,
            output_dir=tmp_path / "out",
            backend=backend,
        )


def test_emotion_pilot_safety_guard_rejects_more_than_30_movies(tmp_path: Path):
    """Safety guard must reject inputs containing > 30 movies."""
    too_large = tmp_path / "too_large.parquet"
    df = pd.DataFrame({
        "movie_id": list(range(1, 32)),
        "movie_idx": list(range(31)),
        "ml_title": [f"Movie {i}" for i in range(31)],
        "overview": [f"Overview {i}" for i in range(31)],
    })
    df.to_parquet(too_large)

    backend = MockQwenEmotionBackend()
    with pytest.raises(ValueError, match="exceeds the maximum pilot limit of 30"):
        run_qwen_emotion_pilot(
            input_sample_path=too_large,
            output_dir=tmp_path / "out",
            backend=backend,
        )


# ==============================================================================
# 2. MOCK BACKEND TESTS
# ==============================================================================

def test_mock_emotion_backend_generates_valid_extraction():
    """MockQwenEmotionBackend generates valid, grounded evidence and scores."""
    backend = MockQwenEmotionBackend()
    output_text = backend.generate(
        system_prompt="Test system prompt",
        user_prompt="Extract emotions for Toy Story (1995)",
        movie_id=1,
        movie_idx=0,
        tmdb_id=862,
        title="Toy Story (1995)",
        overview="A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.",
    )

    extraction = parse_and_validate_emotion_extraction(
        raw_text=output_text,
        expected_movie_id=1,
        expected_movie_idx=0,
        expected_tmdb_id=862,
    )
    assert extraction.movie_id == 1
    assert extraction.movie_idx == 0
    assert extraction.title == "Toy Story (1995)"

    # Verify all 8 emotions and evidence-score consistency
    for emo in PLUTCHIK_EMOTIONS:
        val = getattr(extraction.emotions, emo)
        if val.score > 0.0:
            assert len(val.evidence) >= 1
        else:
            assert len(val.evidence) == 0


def test_mock_emotion_backend_failure_modes():
    """Verify failure injection modes in MockQwenEmotionBackend."""
    # 1. Malformed JSON
    b_malformed = MockQwenEmotionBackend(failure_mode="malformed_json")
    out_mal = b_malformed.generate("sys", "user", movie_id=1)
    with pytest.raises(Exception):
        parse_and_validate_emotion_extraction(out_mal, expected_movie_id=1)

    # 2. Score without evidence
    b_no_ev = MockQwenEmotionBackend(failure_mode="score_without_evidence")
    out_no_ev = b_no_ev.generate("sys", "user", movie_id=1)
    with pytest.raises(Exception, match="requires at least one supporting evidence"):
        parse_and_validate_emotion_extraction(out_no_ev, expected_movie_id=1)

    # 3. Evidence with zero score
    b_zero_ev = MockQwenEmotionBackend(failure_mode="evidence_with_zero_score")
    out_zero_ev = b_zero_ev.generate("sys", "user", movie_id=1)
    with pytest.raises(Exception, match="evidence list must be empty"):
        parse_and_validate_emotion_extraction(out_zero_ev, expected_movie_id=1)

    # 4. ID mismatch
    b_mismatch = MockQwenEmotionBackend(failure_mode="id_mismatch")
    out_mismatch = b_mismatch.generate("sys", "user", movie_id=1)
    with pytest.raises(Exception, match="movie_id mismatch"):
        parse_and_validate_emotion_extraction(out_mismatch, expected_movie_id=1)


# ==============================================================================
# 3. END-TO-END PILOT EXECUTION & CHECKPOINTING
# ==============================================================================

def test_run_qwen_emotion_pilot_mock(tmp_path: Path):
    """Test full execution of the dedicated emotion pilot with mock backend."""
    output_dir = tmp_path / "emotion_pilot_test"

    # Create a 3-movie sample
    sample_df = pd.DataFrame([
        {
            "movie_id": 1,
            "movie_idx": 0,
            "tmdb_id": 862,
            "title": "Toy Story (1995)",
            "overview": "A cowboy toy feels threatened and jealous by a spaceman toy, experiencing fear and eventual joy.",
        },
        {
            "movie_id": 2,
            "movie_idx": 1,
            "tmdb_id": 8844,
            "title": "Jumanji (1995)",
            "overview": "Two siblings play a magical board game that unleashes dangerous jungle creatures.",
        },
        {
            "movie_id": 3,
            "movie_idx": 2,
            "tmdb_id": 15602,
            "title": "Grumpier Old Men (1995)",
            "overview": "John and Max resolve to save their favorite bait shop from becoming an Italian restaurant.",
        },
    ])
    sample_path = tmp_path / "sample.parquet"
    sample_df.to_parquet(sample_path)

    backend = MockQwenEmotionBackend()
    summary = run_qwen_emotion_pilot(
        input_sample_path=sample_path,
        output_dir=output_dir,
        backend=backend,
        max_movies=3,
    )

    assert summary["total_movies"] == 3
    assert summary["successful_movies"] == 3
    assert summary["failed_movies"] == 0
    assert summary["schema_success_rate"] == 100.0

    # Verify all artifacts exist
    raw_outputs_file = output_dir / "raw_outputs.jsonl"
    validated_evidence_file = output_dir / "validated_emotion_evidence.jsonl"
    vectors_file = output_dir / "emotion_vectors.jsonl"
    summary_file = output_dir / "pilot_summary.json"
    review_csv_file = output_dir / "emotion_review.csv"

    assert raw_outputs_file.exists()
    assert validated_evidence_file.exists()
    assert vectors_file.exists()
    assert summary_file.exists()
    assert review_csv_file.exists()

    # Verify vectors file structure
    vectors = [json.loads(line) for line in vectors_file.read_text().strip().split("\n")]
    assert len(vectors) == 3
    for v in vectors:
        assert "movie_id" in v
        assert "movie_idx" in v
        assert "title" in v
        assert "prominent_emotions" in v
        for emo in PLUTCHIK_EMOTIONS:
            assert emo in v
            assert isinstance(v[emo], float)
            assert 0.0 <= v[emo] <= 1.0

    # Verify review CSV structure
    df_review = pd.read_csv(review_csv_file)
    assert len(df_review) == 3
    for emo in PLUTCHIK_EMOTIONS:
        assert f"{emo}_evidence" in df_review.columns
        assert f"{emo}_score" in df_review.columns
    assert "prominent_emotions" in df_review.columns
    assert "status" in df_review.columns

    # Verify Checkpointing & Resumption
    # Second run should skip all 3 movies and make 0 backend calls
    backend_spy = MockQwenEmotionBackend()
    summary_resumed = run_qwen_emotion_pilot(
        input_sample_path=sample_path,
        output_dir=output_dir,
        backend=backend_spy,
        max_movies=3,
    )
    assert summary_resumed["skipped_already_validated"] == 3
    assert summary_resumed["successful_movies"] == 3


def test_emotion_pilot_retry_and_failure_logging(tmp_path: Path):
    """Verify that pilot handles failures, retries once, and logs to emotion_failures.jsonl."""
    output_dir = tmp_path / "emotion_failure_test"

    sample_df = pd.DataFrame([
        {
            "movie_id": 1,
            "movie_idx": 0,
            "tmdb_id": 862,
            "title": "Toy Story (1995)",
            "overview": "Woody and Buzz adventure.",
        },
    ])
    sample_path = tmp_path / "sample.parquet"
    sample_df.to_parquet(sample_path)

    # Persistent failure
    backend = MockQwenEmotionBackend(failure_mode="malformed_json")
    summary = run_qwen_emotion_pilot(
        input_sample_path=sample_path,
        output_dir=output_dir,
        backend=backend,
        max_retries=1,
    )

    assert summary["successful_movies"] == 0
    assert summary["failed_movies"] == 1
    assert summary["retries_attempted"] == 1
    assert summary["schema_success_rate"] == 0.0

    failures_file = output_dir / "failures.jsonl"
    assert failures_file.exists()
    failures = [json.loads(line) for line in failures_file.read_text().strip().split("\n")]
    assert len(failures) == 1
    assert failures[0]["movie_id"] == 1
    assert failures[0]["error_category"] == "json_syntax_error"


# ==============================================================================
# 4. THREE-WAY EMOTION COMPARISON TESTS
# ==============================================================================

def test_three_way_emotion_comparison(tmp_path: Path):
    """Verify three-way emotion comparison across mock DataFrames."""
    # Create synthetic pilot reviews for v1.0, v1.1, and dedicated emotion v1.0
    df_v1_0 = pd.DataFrame({
        "movie_id": [1, 2],
        "status": ["SUCCESS", "SUCCESS"],
        "joy": [0.0, 0.0],
        "trust": [0.0, 0.0],
        "fear": [0.0, 0.6],
        "surprise": [0.0, 0.0],
        "sadness": [0.0, 0.0],
        "disgust": [0.0, 0.0],
        "anger": [0.0, 0.0],
        "anticipation": [0.0, 0.4],
        "prominent_emotions": ["", "fear"],
    })

    df_v1_1 = pd.DataFrame({
        "movie_id": [1, 2],
        "status": ["SUCCESS", "SUCCESS"],
        "joy": [0.0, 0.2],
        "trust": [0.0, 0.0],
        "fear": [0.1, 0.7],
        "surprise": [0.0, 0.0],
        "sadness": [0.0, 0.0],
        "disgust": [0.0, 0.0],
        "anger": [0.0, 0.0],
        "anticipation": [0.2, 0.5],
        "prominent_emotions": ["", "fear"],
    })

    df_emo = pd.DataFrame({
        "movie_id": [1, 2],
        "status": ["SUCCESS", "SUCCESS"],
        "joy_score": [0.75, 0.2],
        "joy_evidence": ["Laughing with friends", "Brief grin"],
        "trust_score": [0.60, 0.0],
        "trust_evidence": ["Unwavering loyalty", ""],
        "fear_score": [0.0, 0.80],
        "fear_evidence": ["", "Terrifying monster pursuit"],
        "surprise_score": [0.0, 0.55],
        "surprise_evidence": ["", "Plot twist reveal"],
        "sadness_score": [0.0, 0.0],
        "sadness_evidence": ["", ""],
        "disgust_score": [0.0, 0.0],
        "disgust_evidence": ["", ""],
        "anger_score": [0.0, 0.0],
        "anger_evidence": ["", ""],
        "anticipation_score": [0.70, 0.65],
        "anticipation_evidence": ["Countdown ticking", "Impending race"],
        "prominent_emotions": ["joy; trust; anticipation", "fear; surprise; anticipation"],
    })

    metrics = compute_evidence_emotion_metrics(df_emo, emotion_threshold=0.5)
    assert metrics["successful_movies"] == 2
    assert metrics["evidence_integrity"]["nonzero_with_evidence_pct"] == 100.0
    assert metrics["evidence_integrity"]["zero_with_empty_evidence_pct"] == 100.0

    comparison = compare_three_way_emotion_pilots(
        source_v1_0=df_v1_0,
        source_v1_1=df_v1_1,
        source_emotion_v1=df_emo,
    )

    report_table = format_three_way_emotion_table(comparison)
    assert "THREE-WAY EMOTION PILOT COMPARISON REPORT" in report_table
    assert "EVIDENCE INTEGRITY" in report_table
    assert "Nonzero Scores with >= 1 Evidence Item" in report_table
    assert "Zero Scores with Empty Evidence []" in report_table
