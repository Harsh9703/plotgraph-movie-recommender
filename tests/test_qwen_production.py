"""Unit and Integration Tests for Phase 3F Production KG-L Extraction.

Verifies:
1. Eligible movie filtering and deterministic movie ordering (by movie_idx).
2. Production runner distinct from pilot runner; pilot <= 30 safety guards remain active.
3. Batch result alignment and ordering preservation.
4. Semantic and emotion independent checkpointing.
5. Resume skips already completed tasks; partial success resumes missing tasks only.
6. Retry behavior (max_retries=1) and graceful error categorization.
7. Permanent failure does not terminate the production run.
8. Duplicate prevention across repeated runs.
9. Chunk and range partitioning (coverage, non-overlap, boundary handling).
10. Production summary metrics and emotion descriptive statistics.
11. Completion validation utility (detects missing, duplicates, ID mismatches, evidence inconsistency).
12. Production output isolation (mock mode cannot pollute production dir).
13. Mock mode does not require vLLM or GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pytest

from src.knowledge_graph.kg_l_contract import (
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SCHEMA_VERSION,
    PLUTCHIK_EMOTIONS,
    format_extraction_prompt,
)
from src.knowledge_graph.emotion_contract import (
    DEFAULT_EMOTION_PROMPT_VERSION,
    DEFAULT_EMOTION_SCHEMA_VERSION,
    format_emotion_prompt,
)
from src.knowledge_graph.qwen_pilot import (
    MAX_PILOT_MOVIES,
    load_pilot_movies as pilot_load_pilot_movies,
)
from src.knowledge_graph.qwen_emotion_pilot import (
    load_pilot_movies as emotion_pilot_load_pilot_movies,
)
from src.knowledge_graph.qwen_production import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PRODUCTION_OUTPUT_DIR,
    MockProductionBackend,
    ProductionCheckpointManager,
    compute_production_emotion_statistics,
    estimate_production_runtime,
    export_production_parquet_artifacts,
    load_eligible_production_movies,
    partition_movies,
    run_production_extraction,
    validate_production_run,
)

METADATA_PARQUET_PATH = Path("data/processed/ml-1m/metadata/movies_with_tmdb.parquet")


# ==============================================================================
# 1. ELIGIBLE MOVIE FILTERING & ORDERING TESTS
# ==============================================================================

def test_eligible_movie_filtering_and_ordering():
    """Verify filtering of movies_with_tmdb.parquet to eligible movies with usable overviews."""
    if not METADATA_PARQUET_PATH.exists():
        pytest.skip(f"Metadata parquet not found at {METADATA_PARQUET_PATH}")

    df_eligible = load_eligible_production_movies(METADATA_PARQUET_PATH)

    # 1. Total eligible count should be approximately 3,810 (computed dynamically)
    assert len(df_eligible) > 3000
    assert len(df_eligible) == 3810

    # 2. Strict filtering constraints
    assert (df_eligible["has_overview"] == True).all()
    assert df_eligible["overview"].notna().all()
    assert (df_eligible["overview"].astype(str).str.strip() != "").all()

    # 3. Required columns present with non-null values
    for col in ["movie_idx", "movie_id", "tmdb_id", "ml_title", "overview"]:
        assert col in df_eligible.columns
        assert df_eligible[col].notna().all()

    # 4. Deterministic ascending sort by movie_idx
    assert df_eligible["movie_idx"].is_monotonic_increasing


def test_synthetic_metadata_filtering(tmp_path: Path):
    """Verify eligible movie filtering against a controlled synthetic metadata DataFrame."""
    synth_df = pd.DataFrame([
        {"movie_idx": 10, "movie_id": 1, "tmdb_id": 100, "ml_title": "Good 1", "overview": "A great plot.", "has_overview": True},
        {"movie_idx": 5, "movie_id": 2, "tmdb_id": 200, "ml_title": "Good 2", "overview": " Another plot. ", "has_overview": True},
        {"movie_idx": 2, "movie_id": 3, "tmdb_id": 300, "ml_title": "No Overview", "overview": "", "has_overview": True},
        {"movie_idx": 3, "movie_id": 4, "tmdb_id": 400, "ml_title": "Flag False", "overview": "Some plot.", "has_overview": False},
        {"movie_idx": 4, "movie_id": 5, "tmdb_id": 500, "ml_title": "Null Overview", "overview": None, "has_overview": True},
    ])
    p = tmp_path / "synth_meta.parquet"
    synth_df.to_parquet(p)

    df_res = load_eligible_production_movies(p)
    assert len(df_res) == 2
    # Must be sorted by movie_idx ascending: 5 then 10
    assert list(df_res["movie_idx"]) == [5, 10]
    assert list(df_res["ml_title"]) == ["Good 2", "Good 1"]


# ==============================================================================
# 2. PILOT SAFETY GUARDS REMAIN ACTIVE & DISTINCT RUNNER
# ==============================================================================

def test_pilot_safety_guards_remain_active():
    """Verify that pilot loaders strictly reject full catalog and samples > 30 movies."""
    if not METADATA_PARQUET_PATH.exists():
        pytest.skip("Metadata parquet not found.")

    # Phase 3D pilot loader must reject full catalog
    with pytest.raises(ValueError) as exc1:
        pilot_load_pilot_movies(METADATA_PARQUET_PATH)
    assert "safety guard" in str(exc1.value).lower() or "limit" in str(exc1.value).lower()

    # Phase 3E pilot loader must reject full catalog
    with pytest.raises(ValueError) as exc2:
        emotion_pilot_load_pilot_movies(METADATA_PARQUET_PATH)
    assert "safety guard" in str(exc2.value).lower() or "limit" in str(exc2.value).lower()

    # Phase 3D hard safety limit <= 30
    assert MAX_PILOT_MOVIES == 30


# ==============================================================================
# 3. CHUNKING & RANGE PARTITIONING TESTS
# ==============================================================================

def test_chunk_partitioning_coverage_and_non_overlap():
    """Verify that K chunks partition N movies with zero overlap and 100% coverage."""
    # Create synthetic pre-sorted DataFrame of 35 movies
    df_sample = pd.DataFrame({
        "movie_idx": list(range(35)),
        "movie_id": [i + 100 for i in range(35)],
        "tmdb_id": [i + 1000 for i in range(35)],
        "ml_title": [f"Movie {i}" for i in range(35)],
        "overview": [f"Overview {i}" for i in range(35)],
    })

    num_chunks = 4
    all_chunk_indices: List[int] = []

    for chunk_id in range(num_chunks):
        chunk_df, info = partition_movies(df_sample, chunk_id=chunk_id, num_chunks=num_chunks)
        assert len(chunk_df) > 0
        assert f"Chunk {chunk_id + 1}/{num_chunks}" in info
        all_chunk_indices.extend(chunk_df["movie_idx"].tolist())

    # 1. Total movies recovered equals 35
    assert len(all_chunk_indices) == 35
    # 2. No duplicates across chunks
    assert len(set(all_chunk_indices)) == 35
    # 3. Full original set covered exactly
    assert set(all_chunk_indices) == set(range(35))


def test_chunk_partitioning_invalid_parameters():
    """Verify that invalid chunk parameters raise ValueError."""
    df = pd.DataFrame({"movie_idx": [0, 1, 2]})
    with pytest.raises(ValueError):
        partition_movies(df, chunk_id=3, num_chunks=3)  # chunk_id out of range
    with pytest.raises(ValueError):
        partition_movies(df, chunk_id=0, num_chunks=0)  # num_chunks < 1
    with pytest.raises(ValueError):
        partition_movies(df, chunk_id=1, num_chunks=None)  # missing num_chunks


def test_range_partitioning():
    """Verify row range slicing behavior and boundary error checks."""
    df = pd.DataFrame({"movie_idx": list(range(20))})

    r1, info1 = partition_movies(df, start_idx=5, end_idx=10)
    assert len(r1) == 5
    assert list(r1["movie_idx"]) == list(range(5, 10))
    assert "rows 5..9" in info1

    # End index beyond total is clamped gracefully
    r2, _ = partition_movies(df, start_idx=15, end_idx=50)
    assert len(r2) == 5
    assert list(r2["movie_idx"]) == list(range(15, 20))

    # Invalid range raises ValueError
    with pytest.raises(ValueError):
        partition_movies(df, start_idx=10, end_idx=5)
    with pytest.raises(ValueError):
        partition_movies(df, start_idx=-1, end_idx=5)


# ==============================================================================
# 4. PRODUCTION OUTPUT ISOLATION
# ==============================================================================

def test_production_output_isolation_guard():
    """Verify that mock mode is blocked from writing to real production dir without override."""
    with pytest.raises(ValueError) as exc_info:
        run_production_extraction(
            backend_type="mock",
            output_dir=DEFAULT_PRODUCTION_OUTPUT_DIR,
            allow_mock_in_production_dir=False,
        )
    assert "production safety violation" in str(exc_info.value).lower()


# ==============================================================================
# 5. BATCH RESULT ALIGNMENT & MOCK INFERENCE
# ==============================================================================

def test_mock_backend_batch_result_alignment():
    """Verify that MockProductionBackend generates valid completions matching prompt order."""
    backend = MockProductionBackend()

    prompts_sem = [
        format_extraction_prompt("Toy Story (1995)", "Toys come to life.", 1, 0, 862, DEFAULT_PROMPT_VERSION),
        format_extraction_prompt("Jumanji (1995)", "Magical board game.", 2, 1, 8844, DEFAULT_PROMPT_VERSION),
    ]
    sem_outs = backend.generate_batch(prompts_sem, task_type="semantic")
    assert len(sem_outs) == 2
    assert '"movie_idx": 0' in sem_outs[0]
    assert '"movie_idx": 1' in sem_outs[1]

    prompts_emo = [
        format_emotion_prompt("Toy Story (1995)", "Toys come to life.", 1, 0, 862, DEFAULT_EMOTION_PROMPT_VERSION),
        format_emotion_prompt("Jumanji (1995)", "Magical board game.", 2, 1, 8844, DEFAULT_EMOTION_PROMPT_VERSION),
    ]
    emo_outs = backend.generate_batch(prompts_emo, task_type="emotion")
    assert len(emo_outs) == 2
    assert '"movie_idx": 0' in emo_outs[0]
    assert '"movie_idx": 1' in emo_outs[1]


# ==============================================================================
# 6. INDEPENDENT CHECKPOINTING, RESUME, AND DUPLICATE PREVENTION
# ==============================================================================

def test_independent_checkpointing_and_resumption(tmp_path: Path):
    """Verify that resuming skips completed tasks without re-executing or creating duplicates."""
    out_dir = tmp_path / "prod_checkpoint_test"

    # Step 1: Run on movies 0..4
    summary1 = run_production_extraction(
        backend_type="mock",
        output_dir=out_dir,
        start_idx=0,
        end_idx=5,
        batch_size=2,
    )
    assert summary1["session_metrics"]["semantic"]["successes"] == 5
    assert summary1["session_metrics"]["emotion"]["successes"] == 5

    # Check files exist
    assert (out_dir / "semantic_validated.jsonl").exists()
    assert (out_dir / "emotion_vectors.jsonl").exists()
    assert (out_dir / "semantic_validated.parquet").exists()
    assert (out_dir / "emotion_vectors.parquet").exists()

    # Step 2: Re-run exact same slice (0..4) without retry_failed
    # All 5 movies should be detected as completed; zero new extractions executed
    backend_spy = MockProductionBackend()
    summary2 = run_production_extraction(
        backend_type="mock",
        output_dir=out_dir,
        start_idx=0,
        end_idx=5,
        backend_instance=backend_spy,
    )
    assert summary2["session_metrics"]["semantic"]["successes"] == 0
    assert summary2["session_metrics"]["emotion"]["successes"] == 0
    assert len(backend_spy.call_counts) == 0  # Zero backend calls made!

    # Step 3: Verify no duplicates exist in validated files
    val_res = validate_production_run(output_dir=out_dir, expected_indices=set(range(5)))
    assert val_res["is_valid"] is True
    assert val_res["semantic_validated_count"] == 5
    assert val_res["emotion_validated_count"] == 5


def test_partial_semantic_success_emotion_failure_resume(tmp_path: Path):
    """Verify that if semantic succeeds and emotion fails, resume executes only emotion."""
    out_dir = tmp_path / "prod_partial_test"

    # In Run 1, movie 1 has simulated emotion failure
    failure_modes: Dict[Tuple[int, str], str] = {
        (1, "emotion"): "schema_validation_error",
    }
    backend1 = MockProductionBackend(failure_modes=failure_modes)

    summary1 = run_production_extraction(
        backend_type="mock",
        output_dir=out_dir,
        start_idx=0,
        end_idx=3,
        backend_instance=backend1,
    )
    # Semantic: all 3 succeed
    assert summary1["session_metrics"]["semantic"]["successes"] == 3
    # Emotion: 2 succeed, 1 fails
    assert summary1["session_metrics"]["emotion"]["successes"] == 2
    assert summary1["session_metrics"]["emotion"]["failures"] == 1

    # In Run 2 with clean backend and retry_failed=True:
    # Semantic must NOT be re-executed for any movie.
    # Emotion should re-attempt ONLY movie 1.
    backend2 = MockProductionBackend()
    summary2 = run_production_extraction(
        backend_type="mock",
        output_dir=out_dir,
        start_idx=0,
        end_idx=3,
        retry_failed=True,
        backend_instance=backend2,
    )

    # Semantic: 0 executed (all were already validated)
    assert summary2["session_metrics"]["semantic"]["successes"] == 0
    # Emotion: movie 1 now succeeds
    assert summary2["session_metrics"]["emotion"]["successes"] == 1
    assert summary2["session_metrics"]["emotion"]["failures"] == 0

    # Backend was called only for movie 1 on emotion
    assert (1, "semantic") not in backend2.call_counts
    assert (1, "emotion") in backend2.call_counts


# ==============================================================================
# 7. RETRY BEHAVIOR & PERMANENT FAILURE RESILIENCE
# ==============================================================================

def test_retry_then_succeed_and_permanent_failure_resilience(tmp_path: Path):
    """Verify retry behavior, raw output auditability, and pipeline continuation on failure."""
    out_dir = tmp_path / "prod_retry_test"

    # movie 0: retry_then_succeed on semantic
    # movie 1: permanent malformed_json on semantic
    # movie 2: normal success
    failure_modes: Dict[Tuple[int, str], str] = {
        (0, "semantic"): "retry_then_succeed",
        (1, "semantic"): "malformed_json",
    }
    backend = MockProductionBackend(failure_modes=failure_modes)

    summary = run_production_extraction(
        backend_type="mock",
        output_dir=out_dir,
        start_idx=0,
        end_idx=3,
        batch_size=3,
        backend_instance=backend,
    )

    # Semantic metrics:
    # movie 0: attempt 1 failed, attempt 2 succeeded -> 1 retry, 1 success
    # movie 1: attempt 1 failed, attempt 2 failed -> 1 retry, 1 failure
    # movie 2: attempt 1 succeeded -> 1 success
    sem_m = summary["session_metrics"]["semantic"]
    assert sem_m["successes"] == 2
    assert sem_m["failures"] == 1
    assert sem_m["retries"] == 2

    # Verify raw outputs file contains attempts for movie 0 and movie 1
    raw_file = out_dir / "semantic_raw_outputs.jsonl"
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_lines = [json.loads(line) for line in f if line.strip()]

    # Movie 0 has attempt 1 and attempt 2
    m0_attempts = [r for r in raw_lines if r["movie_idx"] == 0]
    assert len(m0_attempts) == 2
    assert m0_attempts[0]["attempt"] == 1
    assert m0_attempts[1]["attempt"] == 2

    # Movie 1 failure is recorded in failures.jsonl
    fail_file = out_dir / "semantic_failures.jsonl"
    with open(fail_file, "r", encoding="utf-8") as f:
        fail_lines = [json.loads(line) for line in f if line.strip()]
    assert len(fail_lines) == 1
    assert fail_lines[0]["movie_idx"] == 1
    assert fail_lines[0]["error_type"] == "invalid_json"


# ==============================================================================
# 8. PRODUCTION SUMMARY & EMOTION STATISTICS TESTS
# ==============================================================================

def test_production_summary_and_emotion_descriptive_stats(tmp_path: Path):
    """Verify that production_summary.json contains all required schema metrics and stats."""
    out_dir = tmp_path / "prod_summary_test"

    summary = run_production_extraction(
        backend_type="mock",
        output_dir=out_dir,
        start_idx=0,
        end_idx=4,
    )

    # Core keys
    assert summary["production_mode"] is True
    assert summary["model_id"] == DEFAULT_MODEL_ID
    assert summary["semantic_prompt_version"] == DEFAULT_PROMPT_VERSION
    assert summary["emotion_prompt_version"] == DEFAULT_EMOTION_PROMPT_VERSION
    assert summary["emotion_kg_threshold"] == DEFAULT_EMOTION_EDGE_THRESHOLD
    assert summary["target_movies_in_run"] == 4

    # Emotion descriptive statistics
    emo_stats = summary["emotion_descriptive_statistics"]
    assert emo_stats["total_vectors"] == 4
    assert "emotions" in emo_stats
    for emo in PLUTCHIK_EMOTIONS:
        assert emo in emo_stats["emotions"]
        e_info = emo_stats["emotions"][emo]
        assert "mean" in e_info
        assert "median" in e_info
        assert "zero_count" in e_info
        assert "count_above_threshold" in e_info


# ==============================================================================
# 9. COMPLETION VALIDATION UTILITY TESTS
# ==============================================================================

def test_completion_validation_utility_detects_errors(tmp_path: Path):
    """Verify that validate_production_run detects missing records, duplicates, and inconsistencies."""
    out_dir = tmp_path / "val_test"

    # Step 1: Run valid extraction on 3 movies
    run_production_extraction(
        backend_type="mock",
        output_dir=out_dir,
        start_idx=0,
        end_idx=3,
    )
    val_clean = validate_production_run(output_dir=out_dir, expected_indices=set(range(3)))
    assert val_clean["is_valid"] is True
    assert val_clean["error_count"] == 0

    # Step 2: Invalidate by injecting duplicate into semantic_validated.jsonl
    sem_val_p = out_dir / "semantic_validated.jsonl"
    with open(sem_val_p, "r", encoding="utf-8") as f:
        first_line = f.readline()
    with open(sem_val_p, "a", encoding="utf-8") as f:
        f.write(first_line)  # Inject duplicate

    val_dup = validate_production_run(output_dir=out_dir, expected_indices=set(range(3)))
    assert val_dup["is_valid"] is False
    assert any("duplicate" in err.lower() for err in val_dup["errors"])

    # Step 3: Invalidate by injecting evidence inconsistency in emotion_validated_evidence.jsonl
    ev_p = out_dir / "emotion_validated_evidence.jsonl"
    lines = []
    with open(ev_p, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(json.loads(line))
    # Make joy score 0.8 with empty evidence
    lines[0]["emotions"]["joy"] = {"score": 0.8, "evidence": []}
    with open(ev_p, "w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item) + "\n")

    val_incon = validate_production_run(output_dir=out_dir, expected_indices=set(range(3)))
    assert val_incon["is_valid"] is False
    assert any("evidence is empty" in err.lower() for err in val_incon["errors"])


# ==============================================================================
# 10. RUNTIME ESTIMATION UTILITY TESTS
# ==============================================================================

def test_runtime_estimation_utility():
    """Verify runtime estimation calculations across optimistic, moderate, and conservative scenarios."""
    est = estimate_production_runtime(num_movies=3810, batch_size=16)

    assert est["num_movies"] == 3810
    assert est["batch_size"] == 16
    assert est["optimistic"]["total_seconds"] > 0
    assert est["moderate"]["total_seconds"] > est["optimistic"]["total_seconds"]
    assert est["conservative"]["total_seconds"] > est["moderate"]["total_seconds"]
    assert est["recommended_slurm_walltime"] == "06:00:00"
