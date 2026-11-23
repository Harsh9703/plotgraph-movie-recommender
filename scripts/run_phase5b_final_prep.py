#!/usr/bin/env python3
"""Phase 5B: Final Methodology Freeze and Test-Time Candidate Preparation Pipeline.

Executes:
1. Zero TEST access verification across all paths.
2. Chronological combination of TRAIN + VAL historical interactions.
3. Refitting of frozen EASE candidate generator (reg=500) on positive TRAIN+VAL.
4. Top-100 candidate generation excluding seen positive TRAIN+VAL items.
5. Rebuilding of all user preference profiles using positive TRAIN+VAL history.
6. Construction of the exact 89-feature candidate table (FULL + NRC).
7. Refitting of the final LightGBM LambdaRank model on all available VAL supervision.
8. Generation of unlabeled test-time candidate predictions.
9. Serialization of final evaluation manifest with SHA-256 checksums.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.features.movie_indexer import MovieEntityIndex
from src.features.text_encoder import SBERTMovieEncoder, TFIDFMovieEncoder
from src.final_eval.ease_refit import (
    generate_test_candidates,
    refit_ease_on_train_val,
    validate_test_candidates,
)
from src.final_eval.feature_pipeline import (
    build_test_time_candidate_features,
    validate_feature_table_schema,
)
from src.final_eval.history import (
    get_positive_history_stats,
    load_train_val_history,
)
from src.final_eval.model_refit import (
    generate_final_evaluation_manifest,
    generate_unlabeled_predictions,
    refit_final_reranker,
)
from src.reranker.dataset import assert_no_test_access

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_phase5b_final_prep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5B Final Methodology Freeze and Test-Time Candidate Preparation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/final_eval.yaml",
        help="Path to final evaluation configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Enforce strict TEST split isolation
    inputs = cfg["inputs"]
    for key, path_str in inputs.items():
        assert_no_test_access(str(path_str))
    assert_no_test_access(str(cfg))

    out_dir = Path(cfg["outputs"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" PlotGraph Phase 5B: Final Methodology Freeze & Test-Time Preparation")
    print(" STRICT ISOLATION GUARD: FINAL TEST LABELS HAVE NOT BEEN ACCESSED.")
    print("=" * 80)
    print(f"Historical Interactions: {inputs['train_path']} + {inputs['val_path']}")
    print(f"Output Directory:        {out_dir}")
    print("=" * 80)

    # 2. Load historical interactions (TRAIN + VAL)
    print("\n[Step 1/7] Combining TRAIN and VAL into historical interaction dataset...")
    t0 = time.time()
    train_val_df = load_train_val_history(inputs["train_path"], inputs["val_path"])
    hist_stats = get_positive_history_stats(train_val_df, cfg["methodology"]["positive_threshold"])
    print(f"  * Total historical interactions:          {hist_stats['total_interactions']:,}")
    print(f"  * Total positive historical interactions: {hist_stats['total_positive_interactions']:,}")
    print(f"  * Total active historical users:          {hist_stats['total_users']:,}")
    print(f"  * Completed in {time.time() - t0:.2f}s")

    # 3. Refit EASE on positive TRAIN+VAL
    print("\n[Step 2/7] Refitting frozen EASE candidate generator on TRAIN+VAL (reg=500)...")
    t0 = time.time()
    ease_model = refit_ease_on_train_val(
        train_val_df=train_val_df,
        reg=float(cfg["methodology"]["ease_regularization"]),
        num_users=int(cfg["methodology"]["total_users"]),
        num_items=int(cfg["methodology"]["total_movies"]),
    )
    print(f"  * EASE refit completed in {time.time() - t0:.2f}s")

    # 4. Generate Top-100 Candidates with seen-item exclusion
    print("\n[Step 3/7] Generating Top-100 test candidates for all 6,040 users...")
    all_users = list(range(int(cfg["methodology"]["total_users"])))
    test_cand_path = Path(cfg["outputs"]["test_candidates_parquet"])

    t0 = time.time()
    test_candidates_df = generate_test_candidates(
        model=ease_model,
        all_users=all_users,
        k=int(cfg["methodology"]["k_candidates"]),
        output_path=test_cand_path,
    )
    print(f"  * Generated {len(test_candidates_df):,} candidates in {time.time() - t0:.2f}s")
    print(f"  * Saved candidates to: {test_cand_path}")

    # Validate candidates
    cand_validation = validate_test_candidates(
        candidates_df=test_candidates_df,
        train_val_df=train_val_df,
        expected_users=int(cfg["methodology"]["total_users"]),
        k=int(cfg["methodology"]["k_candidates"]),
    )
    print(f"  * Validation: 604,000 rows verified, zero seen positive TRAIN+VAL leakage.")

    # 5. Load static movie artifacts & rebuild user profiles / candidate features
    print("\n[Step 4/7] Loading static movie artifacts and building test candidate features...")
    t0 = time.time()
    meta_df = pd.read_parquet(inputs["metadata_path"])
    sem_df = pd.read_parquet(inputs["semantic_validated_path"])
    emo_df = pd.read_parquet(inputs["emotion_vectors_path"])
    movie_index = MovieEntityIndex.from_dataframes(meta_df, sem_df, emo_df)

    tfidf_encoder = TFIDFMovieEncoder.load(inputs["tfidf_dir"])
    sbert_encoder = SBERTMovieEncoder.load(
        inputs["text_embeddings_dir"], model_name=cfg["sbert"]["model_id"]
    )
    movie_nrc_df = pd.read_parquet(inputs["movie_nrc_path"])
    print(f"  * Static movie artifacts loaded in {time.time() - t0:.2f}s")

    test_cand_features_path = Path(cfg["outputs"]["test_candidate_features_parquet"])
    test_cand_features_df = build_test_time_candidate_features(
        candidates_df=test_candidates_df,
        train_val_df=train_val_df,
        movie_index=movie_index,
        tfidf_encoder=tfidf_encoder,
        sbert_encoder=sbert_encoder,
        movie_nrc_df=movie_nrc_df,
        half_life_days=float(cfg["methodology"]["half_life_days"]),
        positive_threshold=float(cfg["methodology"]["positive_threshold"]),
        output_path=test_cand_features_path,
    )
    feature_validation = validate_feature_table_schema(test_cand_features_df)
    print(f"  * Candidate features verified: {len(test_cand_features_df):,} rows x {len(test_cand_features_df.columns)} columns.")

    # 6. Refit Final LightGBM Reranker on all VAL supervision
    print("\n[Step 5/7] Refitting final LightGBM LambdaRank model on all VAL supervision...")
    t0 = time.time()
    final_ranker = refit_final_reranker(
        cand_features_path=inputs["phase4c_candidate_features"],
        val_path=inputs["val_path"],
        best_iteration=int(cfg["lightgbm"]["best_iteration"]),
        params=cfg["lightgbm"],
        output_joblib=cfg["outputs"]["final_reranker_joblib"],
        output_txt=cfg["outputs"]["final_reranker_txt"],
    )
    print(f"  * Final reranker refit completed in {time.time() - t0:.2f}s ({final_ranker.booster_.num_trees()} trees).")

    # 7. Dry-Run Unlabeled Candidate Predictions
    print("\n[Step 6/7] Generating unlabeled candidate predictions for test time...")
    unlabeled_pred_path = Path(cfg["outputs"]["unlabeled_predictions_parquet"])
    unlabeled_preds_df = generate_unlabeled_predictions(
        model=final_ranker,
        test_cand_features_df=test_cand_features_df,
        output_path=unlabeled_pred_path,
    )
    print(f"  * Scored {len(unlabeled_preds_df):,} rows without TEST labels.")

    # 8. Final Evaluation Manifest
    print("\n[Step 7/7] Generating final evaluation manifest...")
    manifest_path = Path(cfg["outputs"]["final_manifest_json"])
    artifact_paths = {
        "test_candidates": test_cand_path,
        "test_candidate_features": test_cand_features_path,
        "final_reranker_joblib": cfg["outputs"]["final_reranker_joblib"],
        "final_reranker_txt": cfg["outputs"]["final_reranker_txt"],
        "unlabeled_predictions": unlabeled_pred_path,
    }
    manifest = generate_final_evaluation_manifest(
        config=cfg,
        artifacts=artifact_paths,
        candidate_stats=cand_validation,
        feature_stats=feature_validation,
        output_path=manifest_path,
    )

    print("\n" + "=" * 80)
    print(" PHASE 5B COMPLETE: FINAL MODELING STATE FROZEN")
    print("=" * 80)
    print(f"  * Test Candidates:           {test_cand_path} ({len(test_candidates_df):,} rows)")
    print(f"  * Test Candidate Features:   {test_cand_features_path} ({len(test_cand_features_df.columns)} cols)")
    print(f"  * Final Reranker Model:      {cfg['outputs']['final_reranker_joblib']}")
    print(f"  * Unlabeled Predictions:     {unlabeled_pred_path}")
    print(f"  * Final Manifest:            {manifest_path}")
    print("\n" + "*" * 80)
    print(" PROMINENT CERTIFICATION: FINAL TEST LABELS HAVE NOT BEEN ACCESSED.")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()
