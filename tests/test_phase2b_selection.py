"""Unit tests for Phase 2B hyperparameter tuning, model selection, and candidate export."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from src.candidate_gen.ease import EASERecommender
from src.candidate_gen.evaluator import (
    tune_grid,
    generate_and_save_candidate_pool,
    evaluate_candidate_pool_recall,
)


@pytest.fixture
def synthetic_data():
    # 4 users, 6 items
    # Train
    train_df = pd.DataFrame({
        "user_idx": [0, 0, 1, 1, 2, 2, 3],
        "movie_idx": [0, 1, 1, 2, 2, 3, 0],
        "rating": [5.0, 4.0, 5.0, 4.0, 5.0, 4.0, 4.5],
        "positive": [1, 1, 1, 1, 1, 1, 1],
    })
    # Val: user 0 has positive on item 2, user 1 on item 3
    val_df = pd.DataFrame({
        "user_idx": [0, 1, 2],
        "movie_idx": [2, 3, 0],
        "rating": [5.0, 4.0, 2.0],  # user 2 has rating 2.0 (positive=0)
        "positive": [1, 1, 0],
    })
    return train_df, val_df


def test_tune_grid_selects_highest_validation_ndcg(synthetic_data):
    train_df, val_df = synthetic_data
    param_grid = {"reg": [1.0, 50.0, 100.0]}

    best_model, best_params, best_metric, all_results = tune_grid(
        model_class=EASERecommender,
        param_grid=param_grid,
        train_df=train_df,
        val_df=val_df,
        num_users=4,
        num_items=6,
        target_metric="ndcg@10",
        cutoffs=[2, 5],
    )

    assert len(all_results) == 3
    assert best_params in [{"reg": 1.0}, {"reg": 50.0}, {"reg": 100.0}]
    assert best_model.is_fitted
    # Best metric must be the maximum among all grid runs
    all_ndcgs = [r["target_value"] for r in all_results]
    assert best_metric == max(all_ndcgs)


def test_candidate_pool_export_and_recall(synthetic_data):
    train_df, val_df = synthetic_data
    ease = EASERecommender(reg=1.0).fit(train_df, num_users=4, num_items=6)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_candidates.parquet"
        cand_df = generate_and_save_candidate_pool(
            model=ease,
            all_users=[0, 1, 2, 3],
            k=3,
            output_path=out_path,
            exclude_seen=True,
        )

        assert out_path.is_file()
        assert len(cand_df) == 4 * 3  # 4 users x 3 candidates
        expected_cols = {"user_idx", "movie_idx", "rank", "score", "model_name"}
        assert expected_cols.issubset(set(cand_df.columns))

        # Check rank ordering (1, 2, 3) for each user
        for u in [0, 1, 2, 3]:
            u_cands = cand_df[cand_df["user_idx"] == u].sort_values("rank")
            assert list(u_cands["rank"]) == [1, 2, 3]
            # Scores must be descending
            scores = list(u_cands["score"])
            assert scores[0] >= scores[1] >= scores[2]
            # Seen positive items must not be in candidate list
            seen = ease.get_user_seen(u)
            assert set(u_cands["movie_idx"]).isdisjoint(seen)

        # Evaluate candidate coverage Recall@3
        recall_res = evaluate_candidate_pool_recall(
            candidates_df=cand_df,
            val_df=val_df,
            train_df=train_df,
            k=3,
        )

        assert "recall@3" in recall_res
        assert recall_res["eval_users"] == 2  # Users 0 and 1 have positive val items
        assert recall_res["seen_train_leakage_count"] == 0


def test_candidate_pool_detects_leakage(synthetic_data):
    train_df, val_df = synthetic_data
    # Construct a leaked candidate DataFrame where user 0 gets item 0 (which user 0 saw in train)
    leaked_cand_df = pd.DataFrame([
        {"user_idx": 0, "movie_idx": 0, "rank": 1, "score": 10.0, "model_name": "leaked"},
        {"user_idx": 0, "movie_idx": 4, "rank": 2, "score": 5.0, "model_name": "leaked"},
    ])
    recall_res = evaluate_candidate_pool_recall(
        candidates_df=leaked_cand_df,
        val_df=val_df,
        train_df=train_df,
        k=2,
    )
    assert recall_res["seen_train_leakage_count"] == 1  # Successfully flagged leakage
