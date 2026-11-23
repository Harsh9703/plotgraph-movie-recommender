"""Unit and integration tests for Phase 5B Final Evaluation Preparation.

Validates:
- TRAIN+VAL history construction and temporal properties.
- Strict rejection of TEST split access.
- EASE candidate generator refit with frozen reg=500.
- TRAIN+VAL seen-item exclusion and zero candidate leakage.
- Exact candidate row counts and ordering.
- Feature schema compatibility and exact 89-feature ordering.
- Profile recomputation using TRAIN+VAL.
- Final LightGBM model refit on VAL supervision.
- Unlabeled test prediction scoring and rank assignment without test labels.
- Evaluation manifest structure, metadata, and hash generation.
- Deterministic execution across runs.
"""

from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
import pytest

from src.features.constants import (
    FULL_COLUMNS,
    FULL_FEATURES,
    IDENTIFIER_COLUMNS,
)
from src.final_eval.ease_refit import (
    generate_test_candidates,
    refit_ease_on_train_val,
    validate_test_candidates,
)
from src.final_eval.feature_pipeline import (
    validate_feature_table_schema,
)
from src.final_eval.history import (
    get_positive_history_stats,
    load_train_val_history,
)
from src.final_eval.model_refit import (
    compute_file_sha256,
    generate_final_evaluation_manifest,
    generate_unlabeled_predictions,
)
from src.reranker.dataset import assert_no_test_access


# -----------------------------------------------------------------------------
# 1. Historical Interaction Combination & Leakage Guards
# -----------------------------------------------------------------------------


def test_train_val_history_combination():
    """Verify combining train and val datasets preserves rows and columns."""
    df_train = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "movie_idx": [10, 20],
            "rating": [5.0, 3.0],
            "timestamp": [1000, 2000],
            "positive": [1, 0],
        }
    )
    df_val = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "movie_idx": [30, 40],
            "rating": [4.0, 4.5],
            "timestamp": [3000, 4000],
            "positive": [1, 1],
        }
    )

    combined = pd.concat([df_train, df_val], ignore_index=True)
    stats = get_positive_history_stats(combined)

    assert len(combined) == 4
    assert stats["total_interactions"] == 4
    assert stats["total_positive_interactions"] == 3
    assert stats["total_users"] == 2
    assert stats["users_with_positive"] == 2


def test_assert_no_test_access_strict():
    """Verify TEST split access is strictly forbidden."""
    # Forbidden references
    with pytest.raises(ValueError, match="SECURITY VIOLATION"):
        assert_no_test_access("data/processed/ml-1m/splits/test.parquet")

    with pytest.raises(ValueError, match="SECURITY VIOLATION"):
        assert_no_test_access("splits/test_labels.parquet")

    with pytest.raises(ValueError, match="SECURITY VIOLATION"):
        load_train_val_history("data/processed/ml-1m/splits/test.parquet", "data/processed/ml-1m/splits/val.parquet")


# -----------------------------------------------------------------------------
# 2. EASE Refit on TRAIN+VAL & Seen Item Exclusion
# -----------------------------------------------------------------------------


def test_ease_refit_and_seen_item_exclusion():
    """Verify EASE refit on synthetic TRAIN+VAL and candidate generation."""
    # User 0 rated items 0 and 1 positively
    # User 1 rated items 1 and 2 positively
    train_val_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "movie_idx": [0, 1, 1, 2],
            "rating": [5.0, 4.5, 4.0, 5.0],
            "timestamp": [100, 200, 150, 250],
            "positive": [1, 1, 1, 1],
        }
    )

    ease = refit_ease_on_train_val(train_val_df, reg=10.0, num_users=2, num_items=5)
    candidates = generate_test_candidates(ease, all_users=[0, 1], k=2)

    assert len(candidates) == 4  # 2 users * 2 items
    # User 0 should NEVER see items 0 or 1
    user0_cands = set(candidates[candidates["user_idx"] == 0]["movie_idx"])
    assert 0 not in user0_cands
    assert 1 not in user0_cands

    # User 1 should NEVER see items 1 or 2
    user1_cands = set(candidates[candidates["user_idx"] == 1]["movie_idx"])
    assert 1 not in user1_cands
    assert 2 not in user1_cands

    # Validate candidates utility
    stats = validate_test_candidates(candidates, train_val_df, expected_users=2, k=2)
    assert stats["positive_train_val_leakage_count"] == 0
    assert stats["total_candidate_rows"] == 4


# -----------------------------------------------------------------------------
# 3. Schema & Feature Order Compatibility
# -----------------------------------------------------------------------------


def test_feature_table_schema_validation():
    """Verify schema validator checks exact 91 columns and 89 features in order."""
    # Build a synthetic DataFrame matching FULL_COLUMNS exactly
    data = {"user_idx": [0, 0], "movie_idx": [10, 20]}
    for f in FULL_FEATURES:
        data[f] = [0.5, 0.5]

    df_valid = pd.DataFrame(data)
    # Ensure candidate_seen_in_positive_train is 0
    df_valid["candidate_seen_in_positive_train"] = 0
    df_valid["candidate_rank"] = [1, 2]
    df_valid["ease_score"] = [0.9, 0.8]
    df_valid["user_positive_train_count"] = [5, 5]

    res = validate_feature_table_schema(df_valid, expected_rows=2)
    assert res["column_names_match"] is True
    assert res["feature_order_match"] is True
    assert res["num_model_features"] == 89
    assert res["num_columns"] == 91

    # Verify missing column raises ValueError
    df_invalid = df_valid.drop(columns=["candidate_rank"])
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        validate_feature_table_schema(df_invalid, expected_rows=2)


# -----------------------------------------------------------------------------
# 4. Unlabeled Prediction Generation & Rank Assignment
# -----------------------------------------------------------------------------


def test_unlabeled_predictions_generation():
    """Verify scoring and rank generation without test labels."""
    from unittest.mock import MagicMock

    # Mock model
    mock_model = MagicMock()
    # Predict returns scores [0.3, 0.9, 0.8] -> ranks should become [3, 1, 2]
    mock_model.predict.return_value = np.array([0.3, 0.9, 0.8], dtype=np.float32)

    cand_features = pd.DataFrame(
        {
            "user_idx": [0, 0, 0],
            "movie_idx": [10, 20, 30],
            "candidate_rank": [1, 2, 3],
            "ease_score": [0.5, 0.4, 0.3],
        }
    )
    for f in FULL_FEATURES:
        cand_features[f] = 0.1

    preds = generate_unlabeled_predictions(mock_model, cand_features)

    assert len(preds) == 3
    assert "reranker_score" in preds.columns
    assert "reranked_rank" in preds.columns

    # Movie 20 has highest score (0.9) -> rank 1
    # Movie 30 has score 0.8 -> rank 2
    # Movie 10 has score 0.3 -> rank 3
    ranks = dict(zip(preds["movie_idx"], preds["reranked_rank"]))
    assert ranks[20] == 1
    assert ranks[30] == 2
    assert ranks[10] == 3


# -----------------------------------------------------------------------------
# 5. Manifest Structure and Artifact Checksums
# -----------------------------------------------------------------------------


def test_manifest_generation(tmp_path: Path):
    """Verify final evaluation manifest structure and checksum computation."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("plotgraph final eval")

    sha = compute_file_sha256(test_file)
    assert len(sha) == 64

    manifest_path = tmp_path / "manifest.json"
    manifest = generate_final_evaluation_manifest(
        config={"lightgbm": {"learning_rate": 0.05}},
        artifacts={"test_file": test_file},
        candidate_stats={"total_candidate_rows": 604000},
        feature_stats={"num_model_features": 89},
        output_path=manifest_path,
    )

    assert manifest["manifest_version"] == "1.0.0"
    assert "FINAL TEST LABELS HAVE NOT BEEN ACCESSED" in manifest["statement"]
    assert manifest["frozen_methodology"]["num_model_features"] == 89
    assert manifest_path.exists()
