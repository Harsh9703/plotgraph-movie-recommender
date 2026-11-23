"""Unit and integration tests for Phase 5C One-Time Test Evaluation using synthetic/mock data.

Validates:
- Manifest checksum verification and tamper detection.
- Test coverage statistics and ceiling calculations.
- Evaluation of end-to-end population (including uncovered users with zero hits).
- Evaluation of conditional population (users with captured positives).
- Metric correctness under ranking adjustments.
- Win/loss distribution analysis.
- Relevant item rank movement tracking and threshold crossings.
- Bootstrap significance calculation and reproducibility.
- Zero dependency on real TEST split data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.final_eval.model_refit import compute_file_sha256
from src.final_eval.test_evaluator import (
    analyze_rank_movements,
    analyze_win_loss,
    compute_test_bootstrap_uncertainty,
    compute_test_coverage_stats,
    evaluate_test_predictions,
    verify_manifest_and_frozen_artifacts,
)


# -----------------------------------------------------------------------------
# 1. Manifest Hash Verification and Tamper Detection
# -----------------------------------------------------------------------------


def test_manifest_verification_tamper_detection(tmp_path: Path):
    """Verify that any modification or hash mismatch in frozen artifacts causes immediate failure."""
    file_a = tmp_path / "model.txt"
    file_a.write_text("frozen model content")
    sha_a = compute_file_sha256(file_a)

    manifest_data = {
        "manifest_version": "1.0.0",
        "created_at_utc": "2026-11-23T00:00:00Z",
        "frozen_methodology": {"num_model_features": 89},
        "candidate_pool_summary": {"total_candidate_rows": 604000},
        "artifact_checksums": {
            "model": {
                "path": str(file_a),
                "sha256": sha_a,
                "size_bytes": file_a.stat().st_size,
            }
        },
    }

    manifest_file = tmp_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # 1. Untampered manifest passes verification
    res = verify_manifest_and_frozen_artifacts(manifest_file)
    assert res is not None

    # 2. Tamper file content -> verification must fail
    file_a.write_text("tampered model content!")
    with pytest.raises(ValueError, match="ARTIFACT TAMPER DETECTED|SECURITY AUDIT FAILURE"):
        verify_manifest_and_frozen_artifacts(manifest_file)


# -----------------------------------------------------------------------------
# 2. Test Coverage & Candidate Pool Ceiling
# -----------------------------------------------------------------------------


def test_test_coverage_stats_synthetic():
    """Verify test set coverage stats and recall ceiling calculation on synthetic data."""
    # User 0: test positives {10, 99}. Candidate pool has {10, 20}.
    # User 1: test positives {30}. Candidate pool has {40, 50}. (0 captured)
    test_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "movie_idx": [10, 99, 30],
            "rating": [5.0, 4.0, 4.5],
            "positive": [1, 1, 1],
        }
    )
    candidates_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "movie_idx": [10, 20, 40, 50],
            "rank": [1, 2, 1, 2],
        }
    )

    stats = compute_test_coverage_stats(test_df, candidates_df)

    assert stats["total_test_interactions"] == 3
    assert stats["positive_test_interactions"] == 3
    assert stats["users_with_test_positive"] == 2
    assert stats["captured_test_positives"] == 1
    assert stats["users_with_captured_positive"] == 1
    assert stats["users_with_zero_captured_positive"] == 1
    # Global Recall = 1 / 3
    assert pytest.approx(stats["candidate_pool_recall100_global"], 0.001) == 1.0 / 3.0
    # Mean user recall: User 0 = 1/2 = 0.5, User 1 = 0/1 = 0.0 -> mean = 0.25
    assert pytest.approx(stats["candidate_pool_recall100_mean_user"], 0.001) == 0.25


# -----------------------------------------------------------------------------
# 3. Evaluation of End-to-End vs Conditional Populations
# -----------------------------------------------------------------------------


def test_evaluate_test_predictions_populations():
    """Verify that uncovered users are retained in end-to-end evaluation with 0 hits."""
    # User 0: test positives {10, 99}. Candidates: [20 (rank 1), 10 (rank 2)].
    #   EASE: [20, 10] -> rank 1 is neg, rank 2 is pos.
    #   PlotGraph: reranked_rank puts 10 at rank 1, 20 at rank 2. (improves)
    # User 1: test positives {30}. Candidates: [40, 50]. (0 captured in pool)
    #   Both models get 0 hits for User 1.
    test_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "movie_idx": [10, 99, 30],
            "rating": [5.0, 4.0, 5.0],
            "positive": [1, 1, 1],
        }
    )
    unlabeled_preds_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "movie_idx": [20, 10, 40, 50],
            "candidate_rank": [1, 2, 1, 2],  # EASE ordering
            "ease_score": [0.8, 0.5, 0.7, 0.6],
            "reranker_score": [0.1, 0.9, 0.3, 0.2],
            "reranked_rank": [2, 1, 1, 2],  # PlotGraph swaps User 0 candidates
        }
    )

    e2e, cond, per_user_df = evaluate_test_predictions(unlabeled_preds_df, test_df, cutoffs=[1, 2])

    # Population A (End-to-End): both users (N=2)
    assert e2e["num_users"] == 2
    # Population B (Conditional): only User 0 (N=1)
    assert cond["num_users"] == 1

    # User 1 has 0 hits:
    u1_rec = per_user_df[per_user_df["user_idx"] == 1].iloc[0]
    assert u1_rec["ease_ndcg@1"] == 0.0
    assert u1_rec["plotgraph_ndcg@1"] == 0.0
    assert u1_rec["ease_recall@1"] == 0.0
    assert u1_rec["plotgraph_recall@1"] == 0.0

    # User 0: EASE top-1 is 20 (neg), PlotGraph top-1 is 10 (pos!)
    u0_rec = per_user_df[per_user_df["user_idx"] == 0].iloc[0]
    assert u0_rec["ease_hr@1"] == 0.0
    assert u0_rec["plotgraph_hr@1"] == 1.0
    assert u0_rec["plotgraph_ndcg@1"] > u0_rec["ease_ndcg@1"]

    # In Population A: PlotGraph NDCG@1 should strictly exceed EASE NDCG@1
    assert e2e["plotgraph_ndcg@1"] > e2e["ease_ndcg@1"]


# -----------------------------------------------------------------------------
# 4. Win / Loss Analysis
# -----------------------------------------------------------------------------


def test_win_loss_analysis_synthetic():
    """Verify win/loss counts and average delta computations."""
    per_user_df = pd.DataFrame(
        {
            "user_idx": [0, 1, 2, 3],
            "delta_ndcg@10": [0.20, 0.10, 0.0, -0.05],
        }
    )

    wl = analyze_win_loss(per_user_df)

    assert wl["total_evaluated_users"] == 4
    assert wl["improved_users_count"] == 2
    assert wl["tied_users_count"] == 1
    assert wl["worsened_users_count"] == 1
    assert wl["improved_users_pct"] == 50.0
    assert wl["tied_users_pct"] == 25.0
    assert wl["worsened_users_pct"] == 25.0
    assert pytest.approx(wl["mean_delta_among_improved"], 0.001) == 0.15
    assert pytest.approx(wl["mean_delta_among_worsened"], 0.001) == -0.05


# -----------------------------------------------------------------------------
# 5. Rank Movement Analysis
# -----------------------------------------------------------------------------


def test_rank_movements_synthetic():
    """Verify rank delta, movement categorization, and top-10 threshold crossing."""
    # User 0:
    # Movie 10 is relevant: EASE rank 25 -> PlotGraph rank 5 (improved +20, crossed into top-10)
    # Movie 20 is relevant: EASE rank 3 -> PlotGraph rank 12 (worsened -9, crossed out of top-10)
    # Movie 30 is relevant: EASE rank 8 -> PlotGraph rank 8 (unchanged, stayed in top-10)
    test_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 0],
            "movie_idx": [10, 20, 30],
            "rating": [5.0, 5.0, 5.0],
            "positive": [1, 1, 1],
        }
    )
    unlabeled_preds_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 0],
            "movie_idx": [10, 20, 30],
            "candidate_rank": [25, 3, 8],
            "reranked_rank": [5, 12, 8],
        }
    )

    summary, df_mov = analyze_rank_movements(unlabeled_preds_df, test_df)

    assert summary["total_relevant_items_in_pool"] == 3
    assert summary["items_moved_upward_count"] == 1
    assert summary["items_moved_downward_count"] == 1
    assert summary["items_unchanged_count"] == 1
    assert summary["crossed_into_top10_count"] == 1
    assert summary["crossed_out_of_top10_count"] == 1
    assert summary["net_top10_gain"] == 0

    assert df_mov[df_mov["movie_idx"] == 10]["movement"].iloc[0] == "UP"
    assert df_mov[df_mov["movie_idx"] == 20]["movement"].iloc[0] == "DOWN"
    assert df_mov[df_mov["movie_idx"] == 30]["movement"].iloc[0] == "SAME"


# -----------------------------------------------------------------------------
# 6. Bootstrap Significance Reproducibility
# -----------------------------------------------------------------------------


def test_bootstrap_uncertainty_synthetic():
    """Verify paired bootstrap calculation on synthetic metric arrays."""
    np.random.seed(42)
    metric_arrays = {
        "ease": {"ndcg@10": np.array([0.1, 0.2, 0.0, 0.15] * 25)},
        "plotgraph": {"ndcg@10": np.array([0.15, 0.25, 0.05, 0.20] * 25)},
    }

    b_res = compute_test_bootstrap_uncertainty(metric_arrays, n_bootstraps=500, seed=42)

    assert "delta_ndcg@10" in b_res
    delta_stat = b_res["delta_ndcg@10"]
    assert delta_stat["point_delta"] > 0.0
    assert delta_stat["ci_lower_95"] > 0.0
    assert delta_stat["is_significant_95"] is True
    assert delta_stat["prob_a_superior"] == 1.0
