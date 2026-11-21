"""Unit and Integration Tests for Phase 3D Qwen2.5-7B-Instruct 30-Movie Pilot.

Verifies:
1. Strict Safety Guard: Rejection of full catalog and samples > 30 movies.
2. Mock Inference Backend: Deterministic generation of schema-compliant responses.
3. Checkpointing and Resumption: Skipping already validated movies without redundant calls.
4. Retry Mechanism & Failure Logging: Error categorization, raw output preservation, and graceful continuation.
5. Pilot Review CSV & Summary JSON: Correct schema, continuous emotions, and prominent emotion filtering (score > 0.3).
6. vLLM Backend Lazy Import: Clear diagnostic error message when vLLM is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.knowledge_graph.kg_l_contract import (
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    ExtractionParseError,
    ExtractionValidationError,
    MoviePlotExtraction,
    parse_and_validate_extraction,
)
from src.knowledge_graph.qwen_pilot import (
    MAX_PILOT_MOVIES,
    MockQwenBackend,
    VLLMQwenBackend,
    export_pilot_review_csv,
    load_failure_indices,
    load_pilot_movies,
    load_validated_extractions,
    load_validated_movie_indices,
    run_qwen_pilot,
)

SAMPLE_PARQUET_PATH = Path("data/processed/ml-1m/kg_l/sample_movies.parquet")


# ==============================================================================
# 1. SAFETY GUARD TESTS
# ==============================================================================

def test_pilot_safety_guard_rejects_full_catalog_artifact(tmp_path: Path):
    """Safety guard must immediately reject attempts to pass the full catalog."""
    fake_catalog_path = tmp_path / "movies_with_tmdb.parquet"
    df = pd.DataFrame({
        "movie_id": [1, 2],
        "movie_idx": [0, 1],
        "ml_title": ["Movie 1", "Movie 2"],
        "overview": ["Overview 1", "Overview 2"],
    })
    df.to_parquet(fake_catalog_path)

    with pytest.raises(ValueError, match="Safety guard violation: Detected attempt to run against full catalog"):
        load_pilot_movies(fake_catalog_path, max_allowed=30)


def test_pilot_safety_guard_rejects_more_than_30_movies(tmp_path: Path):
    """Safety guard must reject any dataset containing more than 30 movies."""
    too_large_path = tmp_path / "large_sample.parquet"
    df = pd.DataFrame({
        "movie_id": list(range(1, 32)),
        "movie_idx": list(range(31)),
        "ml_title": [f"Movie {i}" for i in range(31)],
        "overview": [f"Overview for movie {i}" for i in range(31)],
    })
    df.to_parquet(too_large_path)

    with pytest.raises(ValueError, match="exceeds the maximum pilot limit of 30"):
        load_pilot_movies(too_large_path, max_allowed=30)


def test_pilot_safety_guard_rejects_max_allowed_exceeding_30():
    """Safety guard must reject max_allowed parameter exceeding the hard cap of 30."""
    with pytest.raises(ValueError, match="exceeds the hard pilot upper limit of 30"):
        load_pilot_movies(SAMPLE_PARQUET_PATH, max_allowed=35)


def test_pilot_safety_guard_accepts_valid_sample():
    """Safety guard must accept the valid Phase 3C 30-movie sample."""
    if not SAMPLE_PARQUET_PATH.exists():
        pytest.skip(f"Pilot sample not found at {SAMPLE_PARQUET_PATH}")

    df = load_pilot_movies(SAMPLE_PARQUET_PATH, max_allowed=30)
    assert len(df) == 30
    assert "movie_id" in df.columns
    assert "movie_idx" in df.columns
    assert "overview" in df.columns
    assert "ml_title" in df.columns or "title" in df.columns


# ==============================================================================
# 2. MOCK BACKEND TESTS
# ==============================================================================

def test_mock_backend_generates_valid_schema():
    """MockQwenBackend must generate markdown-fenced JSON that passes schema validation."""
    backend = MockQwenBackend()
    user_prompt = (
        "Movie Identifiers:\n"
        "- Title: Toy Story (1995)\n"
        "- MovieLens ID: 1\n"
        "- Movie Index: 0\n"
        "- TMDB ID: 862\n\n"
        "Plot Overview:\n"
        "\"\"\"A little boy named Andy loves to be in his room, playing with his toys.\"\"\""
    )

    response = backend.generate("system prompt", user_prompt)
    assert "```json" in response

    extraction = parse_and_validate_extraction(response, expected_ids={"movie_id": 1, "movie_idx": 0})
    assert isinstance(extraction, MoviePlotExtraction)
    assert extraction.movie_id == 1
    assert extraction.movie_idx == 0
    assert extraction.title == "Toy Story (1995)"
    assert extraction.tmdb_id == 862

    # Check Plutchik emotion constraints
    emotions = extraction.emotions.to_dict()
    assert len(emotions) == 8
    for emo, score in emotions.items():
        assert 0.0 <= score <= 1.0

    # Prominent emotions must be strictly > 0.3 (revised from >0.5)
    prominent = extraction.emotions.get_prominent_emotions(threshold=DEFAULT_EMOTION_EDGE_THRESHOLD)
    for emo, score in prominent.items():
        assert score > DEFAULT_EMOTION_EDGE_THRESHOLD


def test_mock_backend_simulates_failure_modes():
    """MockQwenBackend must accurately simulate specified failure modes for testing."""
    # 1. Malformed JSON
    b_malformed = MockQwenBackend(failure_modes={0: "malformed_json"})
    prompt_0 = "- Movie Index: 0\n- MovieLens ID: 1\n- Title: Test 0\n"
    resp_0 = b_malformed.generate("", prompt_0)
    with pytest.raises(ExtractionParseError):
        parse_and_validate_extraction(resp_0)

    # 2. Schema validation error
    b_schema = MockQwenBackend(failure_modes={1: "schema_validation_error"})
    prompt_1 = "- Movie Index: 1\n- MovieLens ID: 2\n- Title: Test 1\n"
    resp_1 = b_schema.generate("", prompt_1)
    with pytest.raises(ExtractionValidationError):
        parse_and_validate_extraction(resp_1)

    # 3. ID mismatch
    b_mismatch = MockQwenBackend(failure_modes={2: "id_mismatch"})
    prompt_2 = "- Movie Index: 2\n- MovieLens ID: 3\n- Title: Test 2\n"
    resp_2 = b_mismatch.generate("", prompt_2)
    with pytest.raises(ExtractionValidationError, match="mismatch"):
        parse_and_validate_extraction(resp_2, expected_ids={"movie_id": 3, "movie_idx": 2})

    # 4. Backend error
    b_err = MockQwenBackend(failure_modes={3: "backend_error"})
    prompt_3 = "- Movie Index: 3\n- MovieLens ID: 4\n- Title: Test 3\n"
    with pytest.raises(RuntimeError, match="Simulated CUDA out of memory"):
        b_err.generate("", prompt_3)

    # 5. Retry then succeed
    b_retry = MockQwenBackend(failure_modes={4: "retry_then_succeed"})
    prompt_4 = "- Movie Index: 4\n- MovieLens ID: 5\n- Title: Test 4\n"
    call_1 = b_retry.generate("", prompt_4)
    with pytest.raises(ExtractionParseError):
        parse_and_validate_extraction(call_1)

    call_2 = b_retry.generate("", prompt_4)
    valid_ext = parse_and_validate_extraction(call_2, expected_ids={"movie_id": 5, "movie_idx": 4})
    assert valid_ext.movie_id == 5


# ==============================================================================
# 3. END-TO-END PILOT EXECUTION & ARTIFACTS
# ==============================================================================

def test_run_qwen_pilot_end_to_end_mock(tmp_path: Path):
    """Test full 30-movie pilot execution using mock backend in a clean temporary directory."""
    if not SAMPLE_PARQUET_PATH.exists():
        pytest.skip(f"Pilot sample not found at {SAMPLE_PARQUET_PATH}")

    out_dir = tmp_path / "pilot_test_out"

    summary = run_qwen_pilot(
        config_path="configs/kg_l.yaml",
        backend_type="mock",
        input_path=SAMPLE_PARQUET_PATH,
        output_dir=out_dir,
        max_movies=30,
    )

    # Verify summary metrics
    assert summary["backend"] == "mock"
    assert summary["total_movies_in_sample"] == 30
    assert summary["successfully_validated_count"] == 30
    assert summary["failed_count"] == 0
    assert summary["total_retries_used"] == 0
    assert summary["success_rate"] == 100.0
    assert summary["total_pilot_runtime_seconds"] >= 0.0

    # Verify artifacts exist on disk
    raw_outputs_file = out_dir / "raw_outputs.jsonl"
    validated_file = out_dir / "validated_extractions.jsonl"
    failures_file = out_dir / "failures.jsonl"
    summary_file = out_dir / "pilot_summary.json"
    review_file = out_dir / "pilot_review.csv"

    assert raw_outputs_file.exists()
    assert validated_file.exists()
    assert summary_file.exists()
    assert review_file.exists()

    # Validate raw outputs file has exactly 30 entries
    with open(raw_outputs_file, "r", encoding="utf-8") as f:
        raw_lines = [json.loads(line) for line in f if line.strip()]
    assert len(raw_lines) == 30
    assert raw_lines[0]["attempt"] == 1
    assert "raw_response" in raw_lines[0]

    # Validate validated extractions file has exactly 30 valid records
    with open(validated_file, "r", encoding="utf-8") as f:
        val_lines = [json.loads(line) for line in f if line.strip()]
    assert len(val_lines) == 30
    # Every validated line must validate against the Pydantic schema
    for line in val_lines:
        rec = MoviePlotExtraction.model_validate(line)
        assert rec.movie_id > 0

    # Validate human review CSV
    df_review = pd.read_csv(review_file)
    assert len(df_review) == 30
    expected_cols = [
        "movie_idx", "movie_id", "title",
        "themes", "settings", "narrative_devices", "protagonist_types", "based_on",
        "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation",
        "prominent_emotions", "status",
    ]
    for col in expected_cols:
        assert col in df_review.columns

    # Status must all be SUCCESS
    assert (df_review["status"] == "SUCCESS").all()

    # Check emotion column ranges and prominent emotion filtering
    for _, row in df_review.iterrows():
        for emo in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]:
            val = float(row[emo])
            assert 0.0 <= val <= 1.0

            if pd.notna(row["prominent_emotions"]) and row["prominent_emotions"]:
                prom_list = [e.strip() for e in str(row["prominent_emotions"]).split(";")]
                for prom_emo in prom_list:
                    assert float(row[prom_emo]) > DEFAULT_EMOTION_EDGE_THRESHOLD


# ==============================================================================
# 4. CHECKPOINTING & RESUMPTION TESTS
# ==============================================================================

def test_checkpointing_resumption_skips_completed_movies(tmp_path: Path):
    """Re-running the pilot must skip already validated movies without invoking backend."""
    if not SAMPLE_PARQUET_PATH.exists():
        pytest.skip(f"Pilot sample not found at {SAMPLE_PARQUET_PATH}")

    out_dir = tmp_path / "checkpoint_test_out"

    # Run 1: process all 30 movies
    run_qwen_pilot(
        config_path="configs/kg_l.yaml",
        backend_type="mock",
        input_path=SAMPLE_PARQUET_PATH,
        output_dir=out_dir,
        max_movies=30,
    )

    # Inspect raw output count after Run 1
    raw_path = out_dir / "raw_outputs.jsonl"
    with open(raw_path, "r", encoding="utf-8") as f:
        run1_raw_count = len(f.readlines())
    assert run1_raw_count == 30

    # Run 2: execute again in the exact same directory
    summary2 = run_qwen_pilot(
        config_path="configs/kg_l.yaml",
        backend_type="mock",
        input_path=SAMPLE_PARQUET_PATH,
        output_dir=out_dir,
        max_movies=30,
    )

    # Raw outputs count must NOT increase because all movies were skipped via checkpoint
    with open(raw_path, "r", encoding="utf-8") as f:
        run2_raw_count = len(f.readlines())
    assert run2_raw_count == 30, "Raw outputs increased on resume! Checkpointing failed to skip completed movies."

    # Summary must still reflect all 30 validated movies
    assert summary2["successfully_validated_count"] == 30
    assert summary2["total_retries_used"] == 0


# ==============================================================================
# 5. RETRY LOGIC & FAILURE HANDLING TESTS
# ==============================================================================

def test_retry_and_failure_handling(tmp_path: Path):
    """Test that transient errors are retried and persistent errors are logged to failures.jsonl."""
    # Create a small 3-movie sample for targeted failure testing
    sample_df = pd.DataFrame({
        "movie_id": [101, 102, 103],
        "movie_idx": [10, 20, 30],
        "ml_title": ["Movie S1 (1995)", "Movie S2 (1996)", "Movie S3 (1997)"],
        "overview": ["Overview 1", "Overview 2", "Overview 3"],
        "tmdb_id": [11, 22, 33],
    })
    sub_sample_path = tmp_path / "sub_sample.parquet"
    sample_df.to_parquet(sub_sample_path)

    out_dir = tmp_path / "failure_test_out"

    # Configure mock failures:
    # - movie_idx 10: normal success (0 retries)
    # - movie_idx 20: retry_then_succeed (fails on attempt 1, succeeds on attempt 2)
    # - movie_idx 30: malformed_json (persistent failure across attempts 1 and 2)
    mock_failures = {
        20: "retry_then_succeed",
        30: "malformed_json",
    }

    summary = run_qwen_pilot(
        config_path="configs/kg_l.yaml",
        backend_type="mock",
        input_path=sub_sample_path,
        output_dir=out_dir,
        max_movies=3,
        mock_failures=mock_failures,
    )

    # 2 movies succeeded (10 and 20), 1 movie failed (30)
    assert summary["total_movies_in_sample"] == 3
    assert summary["successfully_validated_count"] == 2
    assert summary["failed_count"] == 1
    assert summary["total_retries_used"] == 2  # 1 retry for movie 20 + 1 retry for movie 30

    # Verify failures.jsonl
    failures_file = out_dir / "failures.jsonl"
    assert failures_file.exists()
    with open(failures_file, "r", encoding="utf-8") as f:
        failures = [json.loads(line) for line in f if line.strip()]

    assert len(failures) == 1
    failed_rec = failures[0]
    assert failed_rec["movie_idx"] == 30
    assert failed_rec["movie_id"] == 103
    assert failed_rec["error_type"] == "json_syntax_error"
    assert failed_rec["attempts"] == 2

    # Verify raw outputs has 5 attempts total: 1 (movie 10) + 2 (movie 20) + 2 (movie 30)
    raw_file = out_dir / "raw_outputs.jsonl"
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_entries = [json.loads(line) for line in f if line.strip()]
    assert len(raw_entries) == 5

    # Verify review CSV includes both succeeded and failed rows
    df_review = pd.read_csv(out_dir / "pilot_review.csv")
    assert len(df_review) == 3

    success_rows = df_review[df_review["status"] == "SUCCESS"]
    failed_rows = df_review[df_review["status"] == "FAILED"]

    assert len(success_rows) == 2
    assert set(success_rows["movie_idx"]) == {10, 20}

    assert len(failed_rows) == 1
    assert failed_rows.iloc[0]["movie_idx"] == 30
    assert failed_rows.iloc[0]["themes"] == "" or pd.isna(failed_rows.iloc[0]["themes"])


# ==============================================================================
# 6. VLLM BACKEND IMPORT ERROR HANDLING
# ==============================================================================

def test_vllm_backend_import_error_message():
    """VLLMQwenBackend must provide clear user instructions when vllm is not installed."""
    with patch.dict("sys.modules", {"vllm": None}):
        with pytest.raises(ImportError) as exc_info:
            VLLMQwenBackend()
        err_msg = str(exc_info.value)
        assert "vLLM is not installed" in err_msg
        assert "pip install .[gpu]" in err_msg
        assert "--backend mock" in err_msg


# ==============================================================================
# 7. PILOT V1.1 INDEPENDENT OUTPUT DIRECTORY TEST
# ==============================================================================

def test_v1_1_output_directory_does_not_overwrite_v1_0(tmp_path: Path):
    """Running pilot with prompt v1.1.0 must not overwrite v1.0.0 artifacts."""
    if not SAMPLE_PARQUET_PATH.exists():
        pytest.skip(f"Pilot sample not found at {SAMPLE_PARQUET_PATH}")

    dir_v1_0 = tmp_path / "pilot"
    dir_v1_1 = tmp_path / "pilot_v1_1"

    # Run v1.0.0
    summary_v1_0 = run_qwen_pilot(
        config_path="configs/kg_l.yaml",
        backend_type="mock",
        input_path=SAMPLE_PARQUET_PATH,
        output_dir=dir_v1_0,
        prompt_version="v1.0.0",
        max_movies=5,
    )
    assert summary_v1_0["prompt_version"] == "v1.0.0"
    v1_0_summary_file = dir_v1_0 / "pilot_summary.json"
    assert v1_0_summary_file.exists()
    v1_0_mtime_before = v1_0_summary_file.stat().st_mtime

    # Run v1.1.0 in separate directory
    summary_v1_1 = run_qwen_pilot(
        config_path="configs/kg_l.yaml",
        backend_type="mock",
        input_path=SAMPLE_PARQUET_PATH,
        output_dir=dir_v1_1,
        prompt_version="v1.1.0",
        max_movies=5,
    )
    assert summary_v1_1["prompt_version"] == "v1.1.0"
    v1_1_summary_file = dir_v1_1 / "pilot_summary.json"
    assert v1_1_summary_file.exists()

    # Confirm v1.0 artifacts were not touched or overwritten
    assert v1_0_summary_file.stat().st_mtime == v1_0_mtime_before
    with open(v1_0_summary_file, "r", encoding="utf-8") as f:
        data_v1_0 = json.load(f)
    with open(v1_1_summary_file, "r", encoding="utf-8") as f:
        data_v1_1 = json.load(f)

    assert data_v1_0["prompt_version"] == "v1.0.0"
    assert data_v1_1["prompt_version"] == "v1.1.0"
    assert dir_v1_0 != dir_v1_1
