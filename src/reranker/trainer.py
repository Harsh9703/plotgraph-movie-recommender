"""LightGBM LambdaRank training and feature importance extraction for Phase 5A."""

import time
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.constants import (
    BASE_FEATURES,
    KGL_SEMANTIC_FEATURES,
    KGM_FEATURES,
    NRC_EMOTION_FEATURES,
    QWEN_EMOTION_FEATURES,
    TEXT_FEATURES,
)

# -----------------------------------------------------------------------------
# Standard Ablation Feature Configurations
# -----------------------------------------------------------------------------

ABLATION_CONFIGS: Dict[str, List[str]] = {
    "BASE": BASE_FEATURES,
    "BASE_KGM": BASE_FEATURES + KGM_FEATURES,
    "BASE_KGL_SEMANTIC": BASE_FEATURES + KGL_SEMANTIC_FEATURES,
    "BASE_QWEN_EMOTION": BASE_FEATURES + QWEN_EMOTION_FEATURES,
    "BASE_NRC_EMOTION": BASE_FEATURES + NRC_EMOTION_FEATURES,
    "BASE_TEXT": BASE_FEATURES + TEXT_FEATURES,
    "BASE_KGM_KGL": BASE_FEATURES + KGM_FEATURES + KGL_SEMANTIC_FEATURES,
    "BASE_KGM_KGL_QWEN": BASE_FEATURES + KGM_FEATURES + KGL_SEMANTIC_FEATURES + QWEN_EMOTION_FEATURES,
    "BASE_KGM_KGL_NRC": BASE_FEATURES + KGM_FEATURES + KGL_SEMANTIC_FEATURES + NRC_EMOTION_FEATURES,
    "FULL": BASE_FEATURES + KGM_FEATURES + KGL_SEMANTIC_FEATURES + QWEN_EMOTION_FEATURES + TEXT_FEATURES,
    "FULL_NRC": (
        BASE_FEATURES
        + KGM_FEATURES
        + KGL_SEMANTIC_FEATURES
        + QWEN_EMOTION_FEATURES
        + TEXT_FEATURES
        + NRC_EMOTION_FEATURES
    ),
}


def get_feature_family(feature_name: str) -> str:
    """Categorize a feature into its architectural feature family."""
    if feature_name in BASE_FEATURES:
        return "BASE"
    elif feature_name in KGM_FEATURES:
        return "KGM"
    elif feature_name in KGL_SEMANTIC_FEATURES:
        return "KGL_SEMANTIC"
    elif feature_name in QWEN_EMOTION_FEATURES:
        return "QWEN_EMOTION"
    elif feature_name in NRC_EMOTION_FEATURES:
        return "NRC_EMOTION"
    elif feature_name in TEXT_FEATURES:
        return "TEXT"
    return "OTHER"


def train_lgbm_ranker(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    group_fit: List[int],
    X_inner: np.ndarray,
    y_inner: np.ndarray,
    group_inner: List[int],
    feature_names: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[lgb.LGBMRanker, Dict[str, Any]]:
    """Train LightGBM LambdaRank model with early stopping on inner validation.

    Parameters
    ----------
    X_fit : np.ndarray
        Training feature matrix.
    y_fit : np.ndarray
        Training binary relevance labels.
    group_fit : List[int]
        Query group lengths for training.
    X_inner : np.ndarray
        Validation feature matrix for early stopping.
    y_inner : np.ndarray
        Validation binary relevance labels.
    group_inner : List[int]
        Query group lengths for validation.
    feature_names : List[str]
        List of feature column names.
    params : Optional[Dict[str, Any]]
        Hyperparameters for LGBMRanker.

    Returns
    -------
    Tuple[lgb.LGBMRanker, Dict[str, Any]]
        Trained model and execution metadata dictionary.
    """
    if params is None:
        params = {}

    n_estimators = params.get("n_estimators", 300)
    learning_rate = params.get("learning_rate", 0.05)
    num_leaves = params.get("num_leaves", 31)
    min_child_samples = params.get("min_child_samples", 20)
    subsample = params.get("subsample", 0.8)
    colsample_bytree = params.get("colsample_bytree", 0.8)
    random_state = params.get("random_state", 42)
    n_jobs = params.get("n_jobs", 4)
    early_stopping_rounds = params.get("early_stopping_rounds", 30)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        ndcg_eval_at=[10],
        n_estimators=n_estimators,
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
    ranker.fit(
        X_fit,
        y_fit,
        group=group_fit,
        eval_set=[(X_inner, y_inner)],
        eval_group=[group_inner],
        feature_name=feature_names,
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    )
    fit_duration = time.time() - t0

    metadata = {
        "best_iteration": int(ranker.best_iteration_ or ranker.n_estimators),
        "best_score": (
            float(ranker.best_score_["valid_0"]["ndcg@10"])
            if ranker.best_score_ and "valid_0" in ranker.best_score_
            else None
        ),
        "fit_duration_seconds": round(fit_duration, 4),
        "num_features": len(feature_names),
        "feature_names": feature_names,
    }

    return ranker, metadata


def compute_feature_importance(
    model: lgb.LGBMRanker,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract gain-based and split-based feature importances mapped to feature families."""
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")

    total_gain = float(np.sum(gains))
    gain_pct = (gains / total_gain * 100.0) if total_gain > 0 else np.zeros_like(gains)

    df_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_family": [get_feature_family(f) for f in feature_names],
            "gain": gains,
            "gain_pct": gain_pct,
            "split_count": splits,
        }
    )

    df_importance.sort_values("gain", ascending=False, inplace=True)
    df_importance.reset_index(drop=True, inplace=True)
    return df_importance
