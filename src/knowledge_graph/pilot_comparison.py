"""Lightweight Pilot Comparison Utility (Phase 3D Refinement).

Compares two completed pilot extraction/review runs (e.g. prompt v1.0.0 vs v1.1.0)
across schema validity, continuous emotion distributions, prominent emotion frequencies,
and semantic entity category counts.

Research Integrity Principle:
The metrics computed here are purely descriptive and intended to support manual
inspection. A higher count or higher mean score for an emotion is NOT defined as
inherently 'better'.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .kg_l_contract import (
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    KGL_SEMANTIC_RELATIONS,
    PLUTCHIK_EMOTIONS,
)

logger = logging.getLogger(__name__)

# Map semantic relations to the CSV review column names
SEMANTIC_CATEGORIES: List[str] = [
    "themes",
    "settings",
    "narrative_devices",
    "protagonist_types",
    "based_on",
]


def load_pilot_review_df(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Load and validate a pilot review DataFrame from a directory, CSV path, or existing DataFrame.

    Args:
        source: Directory containing pilot_review.csv, path to a CSV file, or a DataFrame.

    Returns:
        Validated review DataFrame.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if path.is_dir():
            csv_path = path / "pilot_review.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"pilot_review.csv not found in directory: {path}")
            df = pd.read_csv(csv_path)
        elif path.is_file():
            df = pd.read_csv(path)
        else:
            raise FileNotFoundError(f"Pilot review source path not found: {path}")

    # Validate essential columns
    required_cols = set(PLUTCHIK_EMOTIONS) | set(SEMANTIC_CATEGORIES) | {"status"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Pilot review DataFrame is missing required columns: {sorted(missing)}")

    return df


def compute_pilot_metrics(
    source: Union[str, Path, pd.DataFrame],
    emotion_threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD,
) -> Dict[str, Any]:
    """Compute descriptive statistics for a single pilot extraction run.

    Metrics reported:
    - Schema success rate
    - Mean and median per emotion
    - Zero count (score == 0.0) per emotion
    - Count above threshold (score > 0.5) per emotion
    - Number of movies with no emotion above threshold
    - Average semantic entity count per category
    - Empty count (count == 0) per semantic category

    Args:
        source: Pilot directory, CSV path, or DataFrame.
        emotion_threshold: Threshold for prominent emotions (default 0.5).

    Returns:
        Dictionary of computed metrics.
    """
    df = load_pilot_review_df(source)
    total_movies = len(df)

    if total_movies == 0:
        raise ValueError("Cannot compute metrics on an empty pilot DataFrame.")

    # Successful vs Failed extractions
    is_success = df["status"] == "SUCCESS"
    successful_count = int(is_success.sum())
    failed_count = total_movies - successful_count
    schema_success_rate = round((successful_count / total_movies) * 100.0, 2)

    df_valid = df[is_success]

    # Emotion statistics
    emotion_stats: Dict[str, Dict[str, Any]] = {}
    movies_with_no_prominent = 0

    if len(df_valid) > 0:
        # Check movies with 0 prominent emotions (emotions where score > threshold)
        prominent_counts_per_movie = np.zeros(len(df_valid), dtype=int)

        for emo in PLUTCHIK_EMOTIONS:
            scores = df_valid[emo].astype(float).values
            above_mask = scores > emotion_threshold
            prominent_counts_per_movie += above_mask.astype(int)

            mean_val = float(np.mean(scores))
            median_val = float(np.median(scores))
            zero_cnt = int(np.sum(scores == 0.0))
            above_cnt = int(np.sum(above_mask))

            emotion_stats[emo] = {
                "mean": round(mean_val, 3),
                "median": round(median_val, 3),
                "zero_count": zero_cnt,
                "count_above_threshold": above_cnt,
            }

        movies_with_no_prominent = int(np.sum(prominent_counts_per_movie == 0))
    else:
        for emo in PLUTCHIK_EMOTIONS:
            emotion_stats[emo] = {
                "mean": 0.0,
                "median": 0.0,
                "zero_count": 0,
                "count_above_threshold": 0,
            }

    # Semantic entity statistics
    semantic_stats: Dict[str, Dict[str, Any]] = {}
    for cat in SEMANTIC_CATEGORIES:
        counts: List[int] = []
        empty_cnt = 0

        for val in df[cat]:
            if pd.isna(val) or not str(val).strip():
                item_count = 0
            else:
                items = [s.strip() for s in str(val).split(";") if s.strip()]
                item_count = len(items)

            counts.append(item_count)
            if item_count == 0:
                empty_cnt += 1

        avg_cnt = float(np.mean(counts)) if counts else 0.0
        semantic_stats[cat] = {
            "avg_entity_count": round(avg_cnt, 2),
            "empty_count": empty_cnt,
        }

    return {
        "total_movies": total_movies,
        "successful_movies": successful_count,
        "failed_movies": failed_count,
        "schema_success_rate": schema_success_rate,
        "movies_with_no_prominent_emotions": movies_with_no_prominent,
        "emotions": emotion_stats,
        "semantic_categories": semantic_stats,
    }


def compare_pilot_runs(
    source_a: Union[str, Path, pd.DataFrame],
    source_b: Union[str, Path, pd.DataFrame],
    name_a: str = "Pilot v1.0",
    name_b: str = "Pilot v1.1",
    emotion_threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD,
) -> Dict[str, Any]:
    """Compare two pilot extraction runs side by side.

    Args:
        source_a: Path or DataFrame for run A (e.g. prompt v1.0.0).
        source_b: Path or DataFrame for run B (e.g. prompt v1.1.0).
        name_a: Descriptive label for run A.
        name_b: Descriptive label for run B.
        emotion_threshold: Threshold for prominent emotions (default 0.5).

    Returns:
        Structured comparison dictionary.
    """
    metrics_a = compute_pilot_metrics(source_a, emotion_threshold=emotion_threshold)
    metrics_b = compute_pilot_metrics(source_b, emotion_threshold=emotion_threshold)

    # Compute deltas (B - A)
    deltas: Dict[str, Any] = {
        "schema_success_rate": round(metrics_b["schema_success_rate"] - metrics_a["schema_success_rate"], 2),
        "movies_with_no_prominent_emotions": (
            metrics_b["movies_with_no_prominent_emotions"] - metrics_a["movies_with_no_prominent_emotions"]
        ),
        "emotions": {},
        "semantic_categories": {},
    }

    for emo in PLUTCHIK_EMOTIONS:
        ea = metrics_a["emotions"][emo]
        eb = metrics_b["emotions"][emo]
        deltas["emotions"][emo] = {
            "mean": round(eb["mean"] - ea["mean"], 3),
            "median": round(eb["median"] - ea["median"], 3),
            "zero_count": eb["zero_count"] - ea["zero_count"],
            "count_above_threshold": eb["count_above_threshold"] - ea["count_above_threshold"],
        }

    for cat in SEMANTIC_CATEGORIES:
        ca = metrics_a["semantic_categories"][cat]
        cb = metrics_b["semantic_categories"][cat]
        deltas["semantic_categories"][cat] = {
            "avg_entity_count": round(cb["avg_entity_count"] - ca["avg_entity_count"], 2),
            "empty_count": cb["empty_count"] - ca["empty_count"],
        }

    return {
        "pilot_a": {"name": name_a, "metrics": metrics_a},
        "pilot_b": {"name": name_b, "metrics": metrics_b},
        "deltas": deltas,
    }


def format_comparison_table(comparison_result: Dict[str, Any]) -> str:
    """Format comparison results into a clean, human-readable ASCII report table.

    Args:
        comparison_result: Output from compare_pilot_runs.

    Returns:
        Formatted ASCII text table.
    """
    p_a = comparison_result["pilot_a"]
    p_b = comparison_result["pilot_b"]
    name_a = p_a["name"]
    name_b = p_b["name"]
    m_a = p_a["metrics"]
    m_b = p_b["metrics"]
    deltas = comparison_result["deltas"]

    lines: List[str] = [
        "=" * 84,
        f"               PILOT COMPARISON REPORT: {name_a} vs {name_b}",
        "=" * 84,
        f"{'Metric / Dimension':<36} {name_a:>14} {name_b:>14} {'Delta':>14}",
        "-" * 84,
        "OVERALL EXECUTION:",
        f"  Total Movies Sampled               {m_a['total_movies']:>14} {m_b['total_movies']:>14} {m_b['total_movies'] - m_a['total_movies']:>+14}",
        f"  Schema Success Rate                {m_a['schema_success_rate']:>13.1f}% {m_b['schema_success_rate']:>13.1f}% {deltas['schema_success_rate']:>+13.1f}%",
        f"  Movies with No Emotion > 0.5       {m_a['movies_with_no_prominent_emotions']:>14} {m_b['movies_with_no_prominent_emotions']:>14} {deltas['movies_with_no_prominent_emotions']:>+14}",
        "",
        "EMOTIONS (Plutchik 8 Dimensions):",
    ]

    for emo in PLUTCHIK_EMOTIONS:
        ea = m_a["emotions"][emo]
        eb = m_b["emotions"][emo]
        ed = deltas["emotions"][emo]
        lines.append(f"  {emo}:")
        lines.append(f"    Mean                             {ea['mean']:>14.3f} {eb['mean']:>14.3f} {ed['mean']:>+14.3f}")
        lines.append(f"    Median                           {ea['median']:>14.3f} {eb['median']:>14.3f} {ed['median']:>+14.3f}")
        lines.append(f"    Zero Count (score == 0.0)        {ea['zero_count']:>14} {eb['zero_count']:>14} {ed['zero_count']:>+14}")
        lines.append(f"    Prominent Count (score > 0.5)    {ea['count_above_threshold']:>14} {eb['count_above_threshold']:>14} {ed['count_above_threshold']:>+14}")

    lines.append("")
    lines.append("SEMANTIC ENTITY CATEGORIES:")
    for cat in SEMANTIC_CATEGORIES:
        ca = m_a["semantic_categories"][cat]
        cb = m_b["semantic_categories"][cat]
        cd = deltas["semantic_categories"][cat]
        lines.append(f"  {cat}:")
        lines.append(f"    Average Entity Count / Movie     {ca['avg_entity_count']:>14.2f} {cb['avg_entity_count']:>14.2f} {cd['avg_entity_count']:>+14.2f}")
        lines.append(f"    Empty Count (count == 0)         {ca['empty_count']:>14} {cb['empty_count']:>14} {cd['empty_count']:>+14}")

    lines.append("=" * 84)
    lines.append("NOTE: This comparison is descriptive and intended to assist manual inspection.")
    lines.append("Higher counts or nonzero values are not inherently superior.")
    lines.append("=" * 84)

    return "\n".join(lines)


# ==============================================================================
# PHASE 3E: DEDICATED EVIDENCE-GROUNDED EMOTION PILOT COMPARISON
# ==============================================================================

def load_emotion_review_df(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Load and validate an emotion review DataFrame (from emotion_review.csv or pilot_review.csv).

    Args:
        source: Directory, CSV path, or DataFrame.

    Returns:
        Validated emotion review DataFrame.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if path.is_dir():
            # Check for emotion_review.csv first, then fallback to pilot_review.csv
            csv_path = path / "emotion_review.csv"
            if not csv_path.exists():
                csv_path = path / "pilot_review.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Neither emotion_review.csv nor pilot_review.csv found in directory: {path}"
                )
            df = pd.read_csv(csv_path)
        elif path.is_file():
            df = pd.read_csv(path)
        else:
            raise FileNotFoundError(f"Emotion review source path not found: {path}")

    # Verify at least the status column and emotion score columns exist
    if "status" not in df.columns:
        raise ValueError("Emotion review DataFrame is missing required 'status' column.")

    return df


def compute_evidence_emotion_metrics(
    source: Union[str, Path, pd.DataFrame],
    emotion_threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD,
) -> Dict[str, Any]:
    """Compute descriptive emotion metrics and evidence integrity statistics.

    Supports both dedicated emotion reviews (with {emotion}_evidence and {emotion}_score)
    and legacy pilot reviews (with {emotion} as score columns).
    """
    df = load_emotion_review_df(source)
    total_movies = len(df)
    if total_movies == 0:
        raise ValueError("Cannot compute metrics on an empty DataFrame.")

    is_success = df["status"] == "SUCCESS"
    successful_count = int(is_success.sum())
    failed_count = total_movies - successful_count
    schema_success_rate = round((successful_count / total_movies) * 100.0, 2)

    df_valid = df[is_success]

    has_evidence_cols = any(f"{emo}_evidence" in df.columns for emo in PLUTCHIK_EMOTIONS)

    emotion_stats: Dict[str, Dict[str, Any]] = {}
    prominent_counts = np.zeros(len(df_valid), dtype=int)

    total_nonzero_scores = 0
    nonzero_with_evidence = 0
    total_zero_scores = 0
    zero_with_empty_evidence = 0

    for emo in PLUTCHIK_EMOTIONS:
        # Score column name
        score_col = f"{emo}_score" if f"{emo}_score" in df.columns else emo
        evidence_col = f"{emo}_evidence"

        scores = df_valid[score_col].astype(float).values if len(df_valid) > 0 else np.array([])
        above_mask = scores > emotion_threshold
        prominent_counts += above_mask.astype(int)

        mean_val = float(np.mean(scores)) if len(scores) > 0 else 0.0
        median_val = float(np.median(scores)) if len(scores) > 0 else 0.0
        zero_cnt = int(np.sum(scores == 0.0))
        above_cnt = int(np.sum(above_mask))

        # Evidence count if present
        if has_evidence_cols and evidence_col in df.columns:
            evidence_counts = []
            for score_val, ev_val in zip(df_valid[score_col].astype(float), df_valid[evidence_col]):
                if pd.isna(ev_val) or not str(ev_val).strip():
                    item_cnt = 0
                else:
                    item_cnt = len([s for s in str(ev_val).split(";") if s.strip()])
                evidence_counts.append(item_cnt)

                if score_val > 0.0:
                    total_nonzero_scores += 1
                    if item_cnt > 0:
                        nonzero_with_evidence += 1
                else:
                    total_zero_scores += 1
                    if item_cnt == 0:
                        zero_with_empty_evidence += 1

            avg_ev = round(float(np.mean(evidence_counts)), 2) if evidence_counts else 0.0
        else:
            avg_ev = None

        emotion_stats[emo] = {
            "mean": round(mean_val, 3),
            "median": round(median_val, 3),
            "zero_count": zero_cnt,
            "count_above_threshold": above_cnt,
            "avg_evidence_count": avg_ev,
        }

    movies_no_prominent = int(np.sum(prominent_counts == 0))

    nonzero_pct = (
        round((nonzero_with_evidence / total_nonzero_scores) * 100.0, 2)
        if total_nonzero_scores > 0
        else 100.0
    )
    zero_pct = (
        round((zero_with_empty_evidence / total_zero_scores) * 100.0, 2)
        if total_zero_scores > 0
        else 100.0
    )

    return {
        "total_movies": total_movies,
        "successful_movies": successful_count,
        "failed_movies": failed_count,
        "schema_success_rate": schema_success_rate,
        "movies_with_no_prominent_emotions": movies_no_prominent,
        "has_evidence": has_evidence_cols,
        "emotions": emotion_stats,
        "evidence_integrity": {
            "total_nonzero_scores": total_nonzero_scores,
            "nonzero_with_evidence_count": nonzero_with_evidence,
            "nonzero_with_evidence_pct": nonzero_pct,
            "total_zero_scores": total_zero_scores,
            "zero_with_empty_evidence_count": zero_with_empty_evidence,
            "zero_with_empty_evidence_pct": zero_pct,
        },
    }


def compare_three_way_emotion_pilots(
    source_v1_0: Union[str, Path, pd.DataFrame],
    source_v1_1: Union[str, Path, pd.DataFrame],
    source_emotion_v1: Union[str, Path, pd.DataFrame],
    name_v1_0: str = "Combined v1.0",
    name_v1_1: str = "Combined v1.1",
    name_emotion_v1: str = "Dedicated v1.0 (Evidence)",
    emotion_threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD,
) -> Dict[str, Any]:
    """Compare three emotion extraction approaches side by side."""
    m_1_0 = compute_evidence_emotion_metrics(source_v1_0, emotion_threshold=emotion_threshold)
    m_1_1 = compute_evidence_emotion_metrics(source_v1_1, emotion_threshold=emotion_threshold)
    m_emo = compute_evidence_emotion_metrics(source_emotion_v1, emotion_threshold=emotion_threshold)

    return {
        "pilot_v1_0": {"name": name_v1_0, "metrics": m_1_0},
        "pilot_v1_1": {"name": name_v1_1, "metrics": m_1_1},
        "emotion_pilot": {"name": name_emotion_v1, "metrics": m_emo},
    }


def format_three_way_emotion_table(comparison_result: Dict[str, Any]) -> str:
    """Format three-way comparison into an informative, descriptive ASCII table."""
    p1 = comparison_result["pilot_v1_0"]
    p2 = comparison_result["pilot_v1_1"]
    pe = comparison_result["emotion_pilot"]

    n1, m1 = p1["name"], p1["metrics"]
    n2, m2 = p2["name"], p2["metrics"]
    ne, me = pe["name"], pe["metrics"]

    lines: List[str] = [
        "=" * 96,
        "             THREE-WAY EMOTION PILOT COMPARISON REPORT (PHASE 3D vs 3E)",
        "=" * 96,
        f"{'Metric / Dimension':<36} {n1:>18} {n2:>18} {ne:>20}",
        "-" * 96,
        "OVERALL EXECUTION:",
        f"  Total Movies Sampled               {m1['total_movies']:>18} {m2['total_movies']:>18} {me['total_movies']:>20}",
        f"  Schema Success Rate                {m1['schema_success_rate']:>17.1f}% {m2['schema_success_rate']:>17.1f}% {me['schema_success_rate']:>19.1f}%",
        f"  Movies with No Emotion > 0.5       {m1['movies_with_no_prominent_emotions']:>18} {m2['movies_with_no_prominent_emotions']:>18} {me['movies_with_no_prominent_emotions']:>20}",
        "",
        "EMOTIONS (Plutchik 8 Dimensions):",
    ]

    for emo in PLUTCHIK_EMOTIONS:
        e1 = m1["emotions"][emo]
        e2 = m2["emotions"][emo]
        ee = me["emotions"][emo]
        lines.append(f"  {emo}:")
        lines.append(f"    Mean                             {e1['mean']:>18.3f} {e2['mean']:>18.3f} {ee['mean']:>20.3f}")
        lines.append(f"    Median                           {e1['median']:>18.3f} {e2['median']:>18.3f} {ee['median']:>20.3f}")
        lines.append(f"    Zero Count (score == 0.0)        {e1['zero_count']:>18} {e2['zero_count']:>18} {ee['zero_count']:>20}")
        lines.append(f"    Prominent Count (score > 0.5)    {e1['count_above_threshold']:>18} {e2['count_above_threshold']:>18} {ee['count_above_threshold']:>20}")
        if ee.get("avg_evidence_count") is not None:
            lines.append(f"    Avg Evidence Items / Movie       {'N/A':>18} {'N/A':>18} {ee['avg_evidence_count']:>20.2f}")

    lines.append("")
    lines.append("EVIDENCE INTEGRITY (Dedicated Emotion Pilot):")
    ei = me["evidence_integrity"]
    lines.append(f"  Nonzero Scores with >= 1 Evidence Item : {ei['nonzero_with_evidence_pct']:.1f}% ({ei['nonzero_with_evidence_count']}/{ei['total_nonzero_scores']})")
    lines.append(f"  Zero Scores with Empty Evidence []      : {ei['zero_with_empty_evidence_pct']:.1f}% ({ei['zero_with_empty_evidence_count']}/{ei['total_zero_scores']})")
    lines.append("=" * 96)
    lines.append("NOTE: This comparison is descriptive and intended to assist manual grounding review.")
    lines.append("Higher emotion scores or density are NOT automatically defined as superior.")
    lines.append("=" * 96)

    return "\n".join(lines)
