"""Candidate feature construction pipeline on TRAIN+VAL history for Phase 5B."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.features.constants import (
    FULL_COLUMNS,
    FULL_FEATURES,
    IDENTIFIER_COLUMNS,
)
from src.features.feature_builder import build_candidate_features
from src.features.movie_indexer import MovieEntityIndex
from src.features.nrc_feature_builder import build_nrc_candidate_features
from src.features.nrc_profiler import build_nrc_user_profiles
from src.features.text_encoder import SBERTMovieEncoder, TFIDFMovieEncoder
from src.features.text_feature_builder import build_text_candidate_features
from src.features.text_profiler import build_text_user_profiles
from src.features.user_profiler import build_user_profiles

logger = logging.getLogger(__name__)


def build_test_time_candidate_features(
    candidates_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
    movie_index: MovieEntityIndex,
    tfidf_encoder: TFIDFMovieEncoder,
    sbert_encoder: SBERTMovieEncoder,
    movie_nrc_df: pd.DataFrame,
    half_life_days: float = 365.0,
    positive_threshold: float = 4.0,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build the exact frozen 89-feature candidate table using TRAIN+VAL historical profiles.

    Steps:
    1. Rebuild KG-M, KG-L, and Qwen emotion user profiles from positive TRAIN+VAL history.
    2. Rebuild TF-IDF and SBERT text user profiles from positive TRAIN+VAL history.
    3. Rebuild NRC emotion user profiles from positive TRAIN+VAL history.
    4. Compute Phase 4A candidate features (60 columns).
    5. Append Phase 4B text candidate features (68 columns).
    6. Append Phase 4C NRC candidate features (91 columns).
    7. Validate schema, dtypes, feature ordering, and absence of seen positive items.

    Returns
    -------
    pd.DataFrame
        Complete candidate feature table with exactly 604,000 rows and 91 columns.
    """
    logger.info("Rebuilding user preference profiles from positive TRAIN+VAL history...")

    # 1. KG-M / KG-L / Emotion user profiles
    t0 = time.time()
    user_profiles = build_user_profiles(
        train_df=train_val_df,
        movie_index=movie_index,
        half_life_days=half_life_days,
        positive_threshold=positive_threshold,
    )
    logger.info(f"Built KG/Emotion user profiles in {time.time() - t0:.2f}s")

    # 2. Text user profiles
    t0 = time.time()
    text_user_profiles = build_text_user_profiles(
        train_df=train_val_df,
        tfidf_encoder=tfidf_encoder,
        sbert_encoder=sbert_encoder,
        half_life_days=half_life_days,
        positive_threshold=positive_threshold,
    )
    logger.info(f"Built Text user profiles in {time.time() - t0:.2f}s")

    # 3. NRC emotion user profiles
    t0 = time.time()
    nrc_user_profiles = build_nrc_user_profiles(
        train_df=train_val_df,
        movie_nrc_df=movie_nrc_df,
        half_life_days=half_life_days,
        positive_threshold=positive_threshold,
    )
    logger.info(f"Built NRC emotion user profiles in {time.time() - t0:.2f}s")

    # 4. Feature Extraction: Phase 4A
    t0 = time.time()
    p4a_df = build_candidate_features(
        candidate_df=candidates_df,
        user_profiles=user_profiles,
        movie_index=movie_index,
        validate=True,
    )
    logger.info(f"Built Phase 4A candidate features ({len(p4a_df.columns)} cols) in {time.time() - t0:.2f}s")

    # 5. Feature Extraction: Phase 4B (Text)
    t0 = time.time()
    p4b_df = build_text_candidate_features(
        phase4a_df=p4a_df,
        text_user_profiles=text_user_profiles,
        tfidf_encoder=tfidf_encoder,
        sbert_encoder=sbert_encoder,
        validate=True,
    )
    logger.info(f"Built Phase 4B candidate features ({len(p4b_df.columns)} cols) in {time.time() - t0:.2f}s")

    # 6. Feature Extraction: Phase 4C (NRC)
    t0 = time.time()
    full_df = build_nrc_candidate_features(
        phase4b_df=p4b_df,
        nrc_user_profiles=nrc_user_profiles,
        movie_nrc_df=movie_nrc_df,
        validate=True,
    )
    logger.info(f"Built Phase 4C full candidate features ({len(full_df.columns)} cols) in {time.time() - t0:.2f}s")

    # 7. Strict Schema and Compatibility Validation
    validate_feature_table_schema(full_df, expected_rows=len(candidates_df))

    if output_path is not None:
        p_out = Path(output_path)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_parquet(p_out, index=False)
        logger.info(f"Saved test candidate features to {p_out} ({len(full_df):,} rows)")

    return full_df


def validate_feature_table_schema(
    df: pd.DataFrame,
    expected_rows: int = 604000,
) -> Dict[str, Any]:
    """Validate that candidate feature table matches the frozen schema exactly."""
    if len(df) != expected_rows:
        raise ValueError(f"Feature table row count mismatch: expected {expected_rows:,}, got {len(df):,}")

    actual_cols = list(df.columns)
    if actual_cols != FULL_COLUMNS:
        missing = set(FULL_COLUMNS) - set(actual_cols)
        extra = set(actual_cols) - set(FULL_COLUMNS)
        raise ValueError(
            f"Feature schema mismatch! Missing columns: {missing}, Extra columns: {extra}. "
            f"Expected {len(FULL_COLUMNS)} columns in exact order."
        )

    # Check identifiers and ranks
    for col in ["user_idx", "movie_idx", "candidate_rank", "ease_score", "user_positive_train_count"]:
        if df[col].isna().any():
            raise ValueError(f"Identifier/base column '{col}' contains unexpected nulls!")

    # Check that candidate_seen_in_positive_train is strictly 0
    seen_sum = int(df["candidate_seen_in_positive_train"].sum())
    if seen_sum > 0:
        raise ValueError(f"LEAKAGE: {seen_sum} candidate items were seen in user's historical positive history!")

    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "num_model_features": len(FULL_FEATURES),
        "column_names_match": True,
        "feature_order_match": True,
        "zero_positive_history_leakage": (seen_sum == 0),
    }
