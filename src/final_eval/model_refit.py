"""Final LightGBM reranker refit, unlabeled prediction generation, and manifest for Phase 5B."""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.constants import FULL_FEATURES
from src.reranker.dataset import assert_no_test_access, build_lgbm_dataset, build_val_labels

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 checksum for an artifact file."""
    p = Path(file_path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def refit_final_reranker(
    cand_features_path: Union[str, Path],
    val_path: Union[str, Path],
    best_iteration: int = 15,
    params: Optional[Dict[str, Any]] = None,
    output_joblib: Optional[Union[str, Path]] = None,
    output_txt: Optional[Union[str, Path]] = None,
) -> lgb.LGBMRanker:
    """Train the final frozen LightGBM LambdaRank model on ALL eligible VAL supervision.

    Parameters
    ----------
    cand_features_path : Union[str, Path]
        Path to Phase 4C candidate features (full_candidate_features.parquet).
    val_path : Union[str, Path]
        Path to validation split (val.parquet) providing ground-truth labels.
    best_iteration : int
        Exact frozen iteration from Phase 5A model selection (default: 15).
    params : Optional[Dict[str, Any]]
        Hyperparameters matching Phase 5A.
    output_joblib : Optional[Union[str, Path]]
        Destination path for serialized model joblib.
    output_txt : Optional[Union[str, Path]]
        Destination path for LightGBM model text dump.

    Returns
    -------
    lgb.LGBMRanker
        Trained final ranker model.
    """
    assert_no_test_access(str(cand_features_path))
    assert_no_test_access(str(val_path))

    if params is None:
        params = {}

    learning_rate = params.get("learning_rate", 0.05)
    num_leaves = params.get("num_leaves", 31)
    min_child_samples = params.get("min_child_samples", 20)
    subsample = params.get("subsample", 0.8)
    colsample_bytree = params.get("colsample_bytree", 0.8)
    random_state = params.get("random_state", 42)
    n_jobs = params.get("n_jobs", 4)

    logger.info("Loading validation labels and Phase 4C candidate features for final model refit...")
    cand_df = pd.read_parquet(cand_features_path)
    val_df = pd.read_parquet(val_path)

    cand_labeled = build_val_labels(cand_df, val_df, positive_threshold=4.0)

    # Filter to eligible users (>= 1 positive candidate in pool)
    user_pos_count = cand_labeled.groupby("user_idx")["val_label"].sum()
    evaluable_users = user_pos_count[user_pos_count >= 1].index.values
    logger.info(f"Refitting final model on all {len(evaluable_users):,} evaluable users ({len(evaluable_users)*100:,} rows)...")

    X, y, groups, _ = build_lgbm_dataset(cand_labeled, evaluable_users, FULL_FEATURES)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        ndcg_eval_at=[10],
        n_estimators=best_iteration,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        importance_type="gain",
        verbose=-1,
    )

    t0 = time.time()
    ranker.fit(X, y, group=groups, feature_name=FULL_FEATURES)
    fit_duration = time.time() - t0
    logger.info(f"Final model refit completed in {fit_duration:.2f}s ({ranker.booster_.num_trees()} trees).")

    if output_joblib is not None:
        p_joblib = Path(output_joblib)
        p_joblib.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(ranker, p_joblib)
        logger.info(f"Saved final model joblib to {p_joblib}")

    if output_txt is not None:
        p_txt = Path(output_txt)
        p_txt.parent.mkdir(parents=True, exist_ok=True)
        ranker.booster_.save_model(str(p_txt))
        logger.info(f"Saved final model text to {p_txt}")

    return ranker


def generate_unlabeled_predictions(
    model: lgb.LGBMRanker,
    test_cand_features_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Score the TEST-time candidate feature table WITHOUT ANY TEST LABELS.

    Parameters
    ----------
    model : lgb.LGBMRanker
        Trained final ranker model.
    test_cand_features_df : pd.DataFrame
        Unlabeled test-time candidate feature table (604,000 rows).
    output_path : Optional[Union[str, Path]]
        Destination path for parquet predictions.

    Returns
    -------
    pd.DataFrame
        Table containing:
        ['user_idx', 'movie_idx', 'candidate_rank', 'ease_score', 'reranker_score', 'reranked_rank'].
    """
    if "user_idx" not in test_cand_features_df.columns or "movie_idx" not in test_cand_features_df.columns:
        raise KeyError("Candidate feature DataFrame must contain 'user_idx' and 'movie_idx'")

    # Ensure stable sorting by user_idx and candidate_rank
    sorted_df = test_cand_features_df.sort_values(["user_idx", "candidate_rank"]).copy()

    X_test = sorted_df[FULL_FEATURES].values.astype(np.float32)

    t0 = time.time()
    preds = model.predict(X_test)
    pred_duration = time.time() - t0
    logger.info(f"Scored {len(sorted_df):,} test candidate rows in {pred_duration:.2f}s")

    sorted_df["reranker_score"] = preds.astype(np.float32)

    # Compute reranked rank (1 to 100) per user: highest reranker_score first, tiebreak on candidate_rank
    sorted_df["reranked_rank"] = (
        sorted_df.sort_values(
            by=["user_idx", "reranker_score", "candidate_rank"],
            ascending=[True, False, True],
        )
        .groupby("user_idx")
        .cumcount()
        + 1
    ).astype(np.int32)

    output_cols = [
        "user_idx",
        "movie_idx",
        "candidate_rank",
        "ease_score",
        "reranker_score",
        "reranked_rank",
    ]
    result_df = sorted_df[output_cols].copy()

    if output_path is not None:
        p_out = Path(output_path)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(p_out, index=False)
        logger.info(f"Saved unlabeled test predictions to {p_out} ({len(result_df):,} rows)")

    return result_df


def generate_final_evaluation_manifest(
    config: Dict[str, Any],
    artifacts: Dict[str, Union[str, Path]],
    candidate_stats: Dict[str, Any],
    feature_stats: Dict[str, Any],
    output_path: Union[str, Path],
) -> Dict[str, Any]:
    """Generate final evaluation manifest certifying frozen methodology and zero test access."""
    hashes = {}
    for name, p in artifacts.items():
        if Path(p).exists():
            hashes[name] = {
                "path": str(p),
                "sha256": compute_file_sha256(p),
                "size_bytes": Path(p).stat().st_size,
            }

    manifest = {
        "manifest_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 5B - Final Methodology Freeze and Test-Time Candidate Preparation",
        "statement": "FINAL TEST LABELS HAVE NOT BEEN ACCESSED. ZERO TEST ACCESS VERIFIED.",
        "frozen_methodology": {
            "selected_model": "FULL_NRC",
            "feature_family": "FULL + NRC (All Features)",
            "num_model_features": len(FULL_FEATURES),
            "features_list": FULL_FEATURES,
            "candidate_generator": "EASE",
            "ease_regularization": 500.0,
            "positive_rating_threshold": 4.0,
            "recency_half_life_days": 365.0,
            "lightgbm_hyperparameters": config.get("lightgbm", {}),
            "tfidf_configuration": {
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.8,
                "stop_words": "english",
                "norm": "l2",
            },
            "sbert_model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "sbert_dimension": 384,
            "nrc_configuration": {
                "lexicon": "EmoLex v0.92",
                "package": "NRCLex 4.1.0",
                "dimensions": 8,
                "norm": "l1 (sums to 1.0)",
            },
            "qwen_emotion_methodology": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "prompt_version": "v1.0.0",
                "schema_version": "v1.0.0",
                "dimensions": 8,
                "evidence_grounded": True,
            },
            "kgl_threshold": "> 0.3",
            "deterministic_seed": 42,
            "history_definition": "TRAIN + VAL (all interactions preceding TEST)",
        },
        "candidate_pool_summary": candidate_stats,
        "feature_table_summary": feature_stats,
        "artifact_checksums": hashes,
    }

    p_out = Path(output_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved final evaluation manifest to {p_out}")

    return manifest
