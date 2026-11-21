"""Unit tests for the Pilot Comparison Utility.

Verifies:
1. Exact metric computations (success rate, emotion mean/median/zeros/prominents,
   movies with zero prominent emotions, semantic entity averages and empty counts).
2. Side-by-side run comparison and delta calculation.
3. ASCII table formatting.
4. Directory, CSV, and DataFrame input loading.
5. Error handling on missing columns or empty data.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.knowledge_graph.pilot_comparison import (
    compare_pilot_runs,
    compute_pilot_metrics,
    format_comparison_table,
    load_pilot_review_df,
)


def create_synthetic_pilot_review_df() -> pd.DataFrame:
    """Create a controlled 4-movie review DataFrame for analytical verification."""
    return pd.DataFrame([
        {
            "movie_idx": 0,
            "movie_id": 100,
            "title": "Movie 0",
            "themes": "friendship; loyalty",
            "settings": "suburbia",
            "narrative_devices": "flashback",
            "protagonist_types": "student",
            "based_on": "novel",
            "joy": 0.8,
            "trust": 0.6,
            "fear": 0.2,
            "surprise": 0.1,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.7,
            "prominent_emotions": "joy; trust; anticipation",
            "status": "SUCCESS",
        },
        {
            "movie_idx": 1,
            "movie_id": 101,
            "title": "Movie 1",
            "themes": "",
            "settings": "",
            "narrative_devices": "",
            "protagonist_types": "",
            "based_on": "",
            "joy": 0.0,
            "trust": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.0,
            "prominent_emotions": "",
            "status": "SUCCESS",
        },
        {
            "movie_idx": 2,
            "movie_id": 102,
            "title": "Movie 2",
            "themes": "survival",
            "settings": "ocean",
            "narrative_devices": "",
            "protagonist_types": "sailor",
            "based_on": "",
            "joy": 0.4,
            "trust": 0.3,
            "fear": 0.5,  # <= 0.5, so NOT prominent
            "surprise": 0.2,
            "sadness": 0.4,
            "disgust": 0.1,
            "anger": 0.3,
            "anticipation": 0.4,
            "prominent_emotions": "",
            "status": "SUCCESS",
        },
        {
            "movie_idx": 3,
            "movie_id": 103,
            "title": "Movie 3",
            "themes": "",
            "settings": "",
            "narrative_devices": "",
            "protagonist_types": "",
            "based_on": "",
            "joy": 0.0,
            "trust": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.0,
            "prominent_emotions": "",
            "status": "FAILED",
        },
    ])


def test_compute_pilot_metrics_exact_values():
    """Verify that compute_pilot_metrics computes all analytical numbers accurately."""
    df = create_synthetic_pilot_review_df()
    metrics = compute_pilot_metrics(df, emotion_threshold=0.5)

    # 1. Execution summary
    assert metrics["total_movies"] == 4
    assert metrics["successful_movies"] == 3
    assert metrics["failed_movies"] == 1
    assert metrics["schema_success_rate"] == 75.0

    # 2. Movies with no prominent emotions (>0.5):
    # Movie 0: joy=0.8, trust=0.6, anticipation=0.7 (3 prominent)
    # Movie 1: all 0.0 (0 prominent) -> NO prominent
    # Movie 2: fear=0.5 (not > 0.5), all others < 0.5 (0 prominent) -> NO prominent
    # (Failed movies are excluded from valid emotion analysis)
    assert metrics["movies_with_no_prominent_emotions"] == 2

    # 3. Emotion stats for 'joy' across 3 successful movies: [0.8, 0.0, 0.4]
    # mean = (0.8 + 0.0 + 0.4) / 3 = 0.4
    # median = 0.4
    # zero count = 1 (movie 1)
    # count above threshold (>0.5) = 1 (movie 0)
    joy_stats = metrics["emotions"]["joy"]
    assert joy_stats["mean"] == 0.4
    assert joy_stats["median"] == 0.4
    assert joy_stats["zero_count"] == 1
    assert joy_stats["count_above_threshold"] == 1

    # Emotion stats for 'fear': [0.2, 0.0, 0.5]
    fear_stats = metrics["emotions"]["fear"]
    assert fear_stats["mean"] == round((0.2 + 0.0 + 0.5) / 3, 3)
    assert fear_stats["zero_count"] == 1
    assert fear_stats["count_above_threshold"] == 0  # 0.5 is not > 0.5

    # 4. Semantic categories across all 4 movies:
    # 'narrative_devices':
    # movie 0: ["flashback"] -> count=1
    # movie 1: [] -> count=0
    # movie 2: [] -> count=0
    # movie 3: [] -> count=0
    # empty count = 3
    # avg entity count = 1 / 4 = 0.25
    nd_stats = metrics["semantic_categories"]["narrative_devices"]
    assert nd_stats["empty_count"] == 3
    assert nd_stats["avg_entity_count"] == 0.25

    # 'themes':
    # movie 0: ["friendship", "loyalty"] -> count=2
    # movie 1: [] -> count=0
    # movie 2: ["survival"] -> count=1
    # movie 3: [] -> count=0
    # empty count = 2
    # avg count = 3 / 4 = 0.75
    th_stats = metrics["semantic_categories"]["themes"]
    assert th_stats["empty_count"] == 2
    assert th_stats["avg_entity_count"] == 0.75


def test_compare_pilot_runs_and_formatting():
    """Verify side-by-side run comparison, delta calculation, and ASCII table rendering."""
    df_a = create_synthetic_pilot_review_df()
    # Create df_b with slight improvements
    df_b = df_a.copy()
    # In df_b, movie 1 has calibrated emotions
    df_b.loc[1, "joy"] = 0.6  # Now prominent (>0.5)
    df_b.loc[1, "status"] = "SUCCESS"
    df_b.loc[3, "status"] = "SUCCESS"  # Fixed failure
    df_b.loc[3, "joy"] = 0.7  # Now prominent (>0.5)

    comparison = compare_pilot_runs(
        source_a=df_a,
        source_b=df_b,
        name_a="Pilot v1.0",
        name_b="Pilot v1.1",
    )

    assert "pilot_a" in comparison
    assert "pilot_b" in comparison
    assert "deltas" in comparison

    deltas = comparison["deltas"]
    # In B, success rate is 100%, A is 75% -> delta = +25.0%
    assert deltas["schema_success_rate"] == 25.0
    # In B, movies with no prominent emotion decreased
    assert deltas["movies_with_no_prominent_emotions"] < 0
    # In B, joy mean increased
    assert deltas["emotions"]["joy"]["mean"] > 0

    # Test report formatting
    report = format_comparison_table(comparison)
    assert "PILOT COMPARISON REPORT: Pilot v1.0 vs Pilot v1.1" in report
    assert "joy:" in report
    assert "themes:" in report
    assert "narrative_devices:" in report
    assert "OVERALL EXECUTION:" in report


def test_load_pilot_review_df_from_dir_and_csv(tmp_path: Path):
    """Verify load_pilot_review_df works from a directory or a direct CSV path."""
    df = create_synthetic_pilot_review_df()
    csv_file = tmp_path / "pilot_review.csv"
    df.to_csv(csv_file, index=False)

    # 1. Load from file
    df_from_file = load_pilot_review_df(csv_file)
    assert len(df_from_file) == 4

    # 2. Load from directory
    df_from_dir = load_pilot_review_df(tmp_path)
    assert len(df_from_dir) == 4

    # 3. Nonexistent directory raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_pilot_review_df(tmp_path / "nonexistent")


def test_missing_columns_rejected():
    """Verify ValueError is raised when essential columns are missing."""
    bad_df = pd.DataFrame({"movie_idx": [0], "title": ["Test"]})
    with pytest.raises(ValueError, match="missing required columns"):
        load_pilot_review_df(bad_df)
