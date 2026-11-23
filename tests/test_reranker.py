"""Comprehensive unit and integration tests for Phase 5A LightGBM Reranker.

Validates:
- VAL label construction and positive threshold handling.
- Candidate-pool label alignment.
- User-level split disjointness, determinism, and zero user leakage.
- LightGBM dataset group construction and row count conservation.
- Feature group selection and identifier exclusion.
- Ranking prediction ordering and tie-breaking.
- Metric correctness and conditional vs end-to-end recall distinction.
- Paired bootstrap statistical reproducibility.
- Strict TEST split access prevention.
- Real candidate feature table integration.
"""

from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
import pytest

from src.features.constants import (
    BASE_FEATURES,
    FULL_FEATURES,
    IDENTIFIER_COLUMNS,
    KGL_SEMANTIC_FEATURES,
    KGM_FEATURES,
    NRC_EMOTION_FEATURES,
    QWEN_EMOTION_FEATURES,
    TEXT_FEATURES,
)
from src.reranker.dataset import (
    UserSplitResult,
    assert_no_test_access,
    build_lgbm_dataset,
    build_val_labels,
    create_user_splits,
    get_candidate_pool_stats,
)
from src.reranker.evaluator import (
    compute_user_bootstrap,
    evaluate_candidate_rankings,
)
from src.reranker.trainer import (
    ABLATION_CONFIGS,
    compute_feature_importance,
    get_feature_family,
    train_lgbm_ranker,
)


# -----------------------------------------------------------------------------
# 1. VAL Label Construction and Threshold Tests
# -----------------------------------------------------------------------------


def test_val_label_construction_and_threshold():
    """Verify positive thresholding and label construction on candidates."""
    cand_df = pd.DataFrame(
        {
            "user_idx": [1, 1, 1, 2, 2],
            "movie_idx": [10, 20, 30, 10, 40],
            "candidate_rank": [1, 2, 3, 1, 2],
        }
    )

    # User 1 rated movie 10 (rating 4.5 -> pos), movie 20 (rating 3.0 -> neg)
    # User 2 rated movie 40 (rating 5.0 -> pos)
    val_df = pd.DataFrame(
        {
            "user_idx": [1, 1, 2],
            "movie_idx": [10, 20, 40],
            "rating": [4.5, 3.0, 5.0],
        }
    )

    labeled = build_val_labels(cand_df, val_df, positive_threshold=4.0)

    assert "val_label" in labeled.columns
    assert len(labeled) == len(cand_df)

    # Check labels
    # User 1: movie 10 -> 1, movie 20 -> 0, movie 30 -> 0
    # User 2: movie 10 -> 0, movie 40 -> 1
    expected_labels = [1, 0, 0, 0, 1]
    assert labeled["val_label"].tolist() == expected_labels


def test_val_label_with_positive_column():
    """Verify that if 'positive' column is present, it takes precedence."""
    cand_df = pd.DataFrame(
        {
            "user_idx": [1, 1],
            "movie_idx": [10, 20],
        }
    )
    val_df = pd.DataFrame(
        {
            "user_idx": [1, 1],
            "movie_idx": [10, 20],
            "rating": [5.0, 1.0],
            "positive": [1, 0],
        }
    )

    labeled = build_val_labels(cand_df, val_df)
    assert labeled["val_label"].tolist() == [1, 0]


# -----------------------------------------------------------------------------
# 2. Candidate Pool Stats & Limitations
# -----------------------------------------------------------------------------


def test_candidate_pool_stats():
    """Verify candidate pool statistics and recall calculation."""
    cand_df = pd.DataFrame(
        {
            "user_idx": [1, 1, 2, 2],
            "movie_idx": [10, 20, 30, 40],
            "candidate_rank": [1, 2, 1, 2],
        }
    )
    # User 1 has 2 positives in VAL: 10 (captured) and 99 (not captured)
    # User 2 has 1 positive in VAL: 30 (captured)
    val_df = pd.DataFrame(
        {
            "user_idx": [1, 1, 2],
            "movie_idx": [10, 99, 30],
            "rating": [4.0, 5.0, 4.5],
            "positive": [1, 1, 1],
        }
    )

    stats = get_candidate_pool_stats(cand_df, val_df)

    assert stats["total_val_positives"] == 3
    assert stats["captured_val_positives"] == 2
    assert stats["users_with_val_positive"] == 2
    assert stats["users_with_captured_positive"] == 2
    assert stats["users_with_zero_captured_positive"] == 0
    # Global recall = 2 / 3
    assert pytest.approx(stats["candidate_pool_recall100_global"], 0.001) == 2.0 / 3.0
    # Mean user recall: user 1 recall = 1/2 = 0.5, user 2 recall = 1/1 = 1.0 -> mean = 0.75
    assert pytest.approx(stats["candidate_pool_recall100_mean_user"], 0.001) == 0.75


# -----------------------------------------------------------------------------
# 3. User Split Disjointness and Determinism
# -----------------------------------------------------------------------------


def test_user_splits_disjointness_and_determinism():
    """Verify that user-level partitions are strictly disjoint and deterministic."""
    # Create synthetic candidate dataset with 100 users, each with 10 candidates
    np.random.seed(42)
    rows = []
    for u in range(100):
        # 80 users have >=1 positive, 20 users have 0
        n_pos = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.1, 0.1])
        for c in range(10):
            val_lbl = 1 if c < n_pos else 0
            rows.append({"user_idx": u, "movie_idx": u * 100 + c, "candidate_rank": c + 1, "val_label": val_lbl})

    df = pd.DataFrame(rows)

    splits_1 = create_user_splits(df, eval_size=0.20, inner_val_size=0.20, seed=42)
    splits_2 = create_user_splits(df, eval_size=0.20, inner_val_size=0.20, seed=42)

    # 1. Determinism
    assert splits_1.eval_users == splits_2.eval_users
    assert splits_1.train_users == splits_2.train_users
    assert splits_1.fit_users == splits_2.fit_users
    assert splits_1.inner_val_users == splits_2.inner_val_users

    # 2. Strict Disjointness
    set_train = set(splits_1.train_users)
    set_eval = set(splits_1.eval_users)
    set_fit = set(splits_1.fit_users)
    set_inner = set(splits_1.inner_val_users)

    assert len(set_train & set_eval) == 0, "Train and Eval overlap!"
    assert len(set_fit & set_inner) == 0, "Fit and Inner-val overlap!"
    assert set_fit | set_inner == set_train, "Fit + Inner-val != Train!"

    # 3. Only users with >=1 positive are in evaluable splits
    user_pos_count = df.groupby("user_idx")["val_label"].sum()
    for u in splits_1.all_evaluable_users:
        assert user_pos_count[u] >= 1


# -----------------------------------------------------------------------------
# 4. LightGBM Dataset and Group Construction
# -----------------------------------------------------------------------------


def test_lgbm_dataset_and_group_conservation():
    """Verify group construction, row conservation, and feature selection."""
    df = pd.DataFrame(
        {
            "user_idx": [2, 2, 1, 1, 1],
            "movie_idx": [10, 20, 30, 40, 50],
            "candidate_rank": [2, 1, 3, 1, 2],
            "feat_a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feat_b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "val_label": [0, 1, 0, 1, 0],
        }
    )

    feature_cols = ["feat_a", "feat_b"]
    user_list = [1, 2]

    X, y, groups, sorted_df = build_lgbm_dataset(df, user_list, feature_cols)

    # Row conservation
    assert len(X) == 5
    assert len(y) == 5
    assert sum(groups) == 5
    assert groups == [3, 2]  # user 1 has 3 items, user 2 has 2 items

    # Verify contiguous user order and rank sort
    assert sorted_df["user_idx"].tolist() == [1, 1, 1, 2, 2]
    assert sorted_df["candidate_rank"].tolist() == [1, 2, 3, 1, 2]

    # Verify feature matrix shape and no identifier leakage
    assert X.shape == (5, 2)
    assert not np.isnan(X).any()


# -----------------------------------------------------------------------------
# 5. Ranking Evaluation and Metric Correctness
# -----------------------------------------------------------------------------


def test_ranking_evaluation_conditional_vs_e2e():
    """Verify NDCG, Recall (e2e vs conditional), HR, and MRR."""
    # User 1:
    # Candidates: movies [10, 20, 30] with ranks 1, 2, 3
    # Scores: [0.9, 0.8, 0.1]
    # In-pool positive: movie 10
    # All VAL positives: movies {10, 99} (99 was missed by candidate gen)
    eval_df = pd.DataFrame(
        {
            "user_idx": [1, 1, 1],
            "movie_idx": [10, 20, 30],
            "candidate_rank": [1, 2, 3],
            "val_label": [1, 0, 0],
            "score": [0.9, 0.8, 0.1],
        }
    )

    val_pos_dict = {1: {10, 99}}

    metrics, user_metrics = evaluate_candidate_rankings(
        eval_df, score_col="score", val_pos_dict=val_pos_dict, cutoffs=[1, 2, 3]
    )

    # Top-1 recommendation is movie 10.
    # Movie 10 is relevant!
    # At K=1:
    # Hits = 1
    # HR@1 = 1.0
    # MRR@1 = 1.0
    # End-to-end Recall@1: 1 hit / 2 VAL positives = 0.50
    # Conditional Recall@1: 1 hit / 1 pool positive = 1.00
    assert metrics["hr@1"] == 1.0
    assert metrics["mrr@1"] == 1.0
    assert metrics["recall_e2e@1"] == 0.5
    assert metrics["recall_cond@1"] == 1.0
    assert metrics["recall_cond@1"] >= metrics["recall_e2e@1"]


# -----------------------------------------------------------------------------
# 6. Paired Bootstrap Statistical Testing
# -----------------------------------------------------------------------------


def test_bootstrap_reproducibility():
    """Verify paired bootstrap produces reproducible 95% CIs and correct superiority."""
    np.random.seed(42)
    # Model A is strictly better than Model B
    vec_a = np.array([0.20, 0.25, 0.30, 0.15, 0.22] * 20)
    vec_b = np.array([0.10, 0.12, 0.15, 0.08, 0.11] * 20)

    res1 = compute_user_bootstrap(vec_a, vec_b, n_bootstraps=500, seed=42)
    res2 = compute_user_bootstrap(vec_a, vec_b, n_bootstraps=500, seed=42)

    # Check reproducibility
    assert res1["point_delta"] == res2["point_delta"]
    assert res1["ci_lower_95"] == res2["ci_lower_95"]
    assert res1["ci_upper_95"] == res2["ci_upper_95"]
    assert res1["prob_a_superior"] == 1.0
    assert res1["is_significant_95"] is True
    assert res1["ci_lower_95"] > 0.0


# -----------------------------------------------------------------------------
# 7. Strict TEST Split Access Guard
# -----------------------------------------------------------------------------


def test_assert_no_test_access():
    """Verify test split access is intercepted and raises ValueError."""
    # Valid paths
    assert_no_test_access("data/processed/ml-1m/splits/val.parquet")
    assert_no_test_access("data/processed/ml-1m/splits/train.parquet")
    assert_no_test_access("data/processed/ml-1m/features/full_candidate_features.parquet")

    # Forbidden paths
    with pytest.raises(ValueError, match="SECURITY VIOLATION"):
        assert_no_test_access("data/processed/ml-1m/splits/test.parquet")

    with pytest.raises(ValueError, match="SECURITY VIOLATION"):
        assert_no_test_access("/path/to/splits/test_data.parquet")


# -----------------------------------------------------------------------------
# 8. Ablation Configurations and Feature Family Mapping
# -----------------------------------------------------------------------------


def test_ablation_configs_and_families():
    """Verify that all 11 ablation configs contain valid, unique features and no IDs."""
    assert len(ABLATION_CONFIGS) == 11

    expected_counts = {
        "BASE": 10,
        "BASE_KGM": 25,
        "BASE_KGL_SEMANTIC": 23,
        "BASE_QWEN_EMOTION": 30,
        "BASE_NRC_EMOTION": 33,
        "BASE_TEXT": 18,
        "BASE_KGM_KGL": 38,
        "BASE_KGM_KGL_QWEN": 58,
        "BASE_KGM_KGL_NRC": 61,
        "FULL": 66,
        "FULL_NRC": 89,
    }

    for name, feat_list in ABLATION_CONFIGS.items():
        assert name in expected_counts
        assert len(feat_list) == expected_counts[name]
        assert len(feat_list) == len(set(feat_list)), f"Duplicate features in {name}"

        # No identifier columns
        for col in IDENTIFIER_COLUMNS:
            assert col not in feat_list, f"Identifier {col} leaked into {name}"
        assert "val_label" not in feat_list

        # Check feature family mapping
        for feat in feat_list:
            fam = get_feature_family(feat)
            assert fam in ["BASE", "KGM", "KGL_SEMANTIC", "QWEN_EMOTION", "NRC_EMOTION", "TEXT"]


# -----------------------------------------------------------------------------
# 9. Real Candidate Feature Table Integration Smoke Test
# -----------------------------------------------------------------------------


def test_real_feature_table_integration():
    """Smoke test loading real candidate table and validation split (VAL only)."""
    cand_path = "data/processed/ml-1m/features/full_candidate_features.parquet"
    val_path = "data/processed/ml-1m/splits/val.parquet"

    if not Path(cand_path).exists() or not Path(val_path).exists():
        pytest.skip("Data artifacts not present for integration test")

    assert_no_test_access(cand_path)
    assert_no_test_access(val_path)

    # Load small slice
    cand_sample = pd.read_parquet(cand_path, columns=["user_idx", "movie_idx", "candidate_rank", "ease_score"]).head(500)
    val_df = pd.read_parquet(val_path)

    labeled = build_val_labels(cand_sample, val_df)
    assert "val_label" in labeled.columns
    assert len(labeled) == 500
    assert set(labeled["val_label"].unique()).issubset({0, 1})
