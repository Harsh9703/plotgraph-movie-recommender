#!/usr/bin/env python3
"""PlotGraph Phase 4A: Feature Engineering Execution Pipeline.

Builds deterministic, leakage-safe KG-M, KG-L, and continuous emotion
features for all 604,000 Phase 2B candidate rows:
1. Loads TRAIN split strictly (asserts no VAL or TEST leakage).
2. Pre-indexes metadata, semantic extractions, and continuous emotion vectors.
3. Constructs recency-weighted user preference distributions.
4. Generates candidate feature matrix (affinities, overlap counts, emotion metrics).
5. Validates row counts, determinism, and leakage guards.
6. Serializes feature table, user profiles, and detailed summary metrics.
7. Prints interpretable sanity analysis and candidate matching examples.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.features.constants import (
    ALL_COLUMNS,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_POSITIVE_THRESHOLD,
    KGL_CATEGORIES,
    KGM_CATEGORIES,
    PLUTCHIK_EMOTIONS,
)
from src.features.feature_builder import build_candidate_features
from src.features.movie_indexer import MovieEntityIndex
from src.features.summary import generate_feature_summary, save_feature_summary
from src.features.user_profiler import build_user_profiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_phase4a_kg_features")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_sanity_analysis(
    features_df: pd.DataFrame,
    summary: Dict[str, Any],
    user_idx: int = 0,
) -> None:
    """Print informative sanity statistics and interpretable candidate examples."""
    print("\n" + "=" * 70)
    print("         PLOTGRAPH PHASE 4A: FEATURE ENGINEERING REPORT")
    print("=" * 70)
    print(f"Total Rows Generated          : {len(features_df):,}")
    print(f"Total Evaluated Users         : {features_df['user_idx'].nunique():,}")
    print(f"Total Feature Columns         : {len(features_df.columns)}")
    print(f"Zero Train-Candidate Leakage  : {summary['leakage_verification']['zero_positive_train_candidate_leakage']}")
    print("-" * 70)
    print("COVERAGE METRICS:")
    cov = summary["coverage_metrics"]
    print(f"  User Emotion Profiles       : {cov['user_emotion_profile_coverage']['active_users']:,} / {cov['user_emotion_profile_coverage']['total_users']:,} ({cov['user_emotion_profile_coverage']['coverage_pct']}%)")
    print(f"  Candidate Emotion Available : {cov['candidate_rows_emotion_coverage']['candidate_emotion_available_rows']:,} / {len(features_df):,} ({cov['candidate_rows_emotion_coverage']['candidate_emotion_available_pct']}%)")
    print(f"  Both Emotion Available      : {cov['candidate_rows_emotion_coverage']['both_user_and_candidate_emotion_available_rows']:,} / {len(features_df):,} ({cov['candidate_rows_emotion_coverage']['both_available_pct']}%)")
    print(f"  Candidate TMDB Link         : {cov['candidate_rows_metadata_coverage']['has_tmdb_link_pct']}%")
    print(f"  Candidate Plot Overview     : {cov['candidate_rows_metadata_coverage']['has_overview_pct']}%")
    print(f"  Candidate KG-M Metadata     : {cov['candidate_rows_metadata_coverage']['has_kgm_metadata_pct']}%")
    print(f"  Candidate KG-L Semantics    : {cov['candidate_rows_metadata_coverage']['has_kgl_semantics_pct']}%")
    print("-" * 70)
    print("SELECTED FEATURE DISTRIBUTIONS:")
    stats = summary["descriptive_statistics"]
    for feat in [
        "ease_score",
        "kgm_genre_affinity",
        "kgm_director_affinity",
        "kgm_actor_affinity",
        "kgl_theme_affinity",
        "kgl_setting_affinity",
        "emotion_cosine_similarity",
    ]:
        if feat in stats:
            s = stats[feat]
            print(f"  {feat:<26}: mean={s['mean']:.4f}, std={s['std']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}, non_zero={s['non_zero_pct']}%")
    print("=" * 70)

    # Print interpretable example for user_idx
    u_rows = features_df[features_df["user_idx"] == user_idx]
    if not u_rows.empty:
        print(f"\nINTERPRETABLE CANDIDATE MATCHING EXAMPLE (User Index: {user_idx}):")
        print(f"  Positive TRAIN items: {int(u_rows.iloc[0]['user_positive_train_count'])}")

        # Find candidate with highest combined affinity and lowest combined affinity
        combined = u_rows["kgm_affinity_mean"] + u_rows["kgl_semantic_affinity_mean"]
        high_idx = combined.idxmax()
        low_idx = combined.idxmin()

        high_row = u_rows.loc[high_idx]
        low_row = u_rows.loc[low_idx]

        print("\n  [HIGH COMPATIBILITY CANDIDATE]")
        print(f"    Movie Index             : {int(high_row['movie_idx'])} (Candidate Rank: {int(high_row['candidate_rank'])})")
        print(f"    EASE Score              : {high_row['ease_score']:.4f}")
        print(f"    KG-M Genre Affinity     : {high_row['kgm_genre_affinity']:.4f} (Shared Genres: {int(high_row['shared_genre_count'])})")
        print(f"    KG-M Actor Affinity     : {high_row['kgm_actor_affinity']:.4f} (Shared Actors: {int(high_row['shared_actor_count'])})")
        print(f"    KG-L Theme Affinity     : {high_row['kgl_theme_affinity']:.4f} (Shared Themes: {int(high_row['shared_theme_count'])})")
        if pd.notna(high_row["emotion_cosine_similarity"]):
            print(f"    Emotion Cosine Sim      : {high_row['emotion_cosine_similarity']:.4f}")

        print("\n  [LOW COMPATIBILITY CANDIDATE]")
        print(f"    Movie Index             : {int(low_row['movie_idx'])} (Candidate Rank: {int(low_row['candidate_rank'])})")
        print(f"    EASE Score              : {low_row['ease_score']:.4f}")
        print(f"    KG-M Genre Affinity     : {low_row['kgm_genre_affinity']:.4f} (Shared Genres: {int(low_row['shared_genre_count'])})")
        print(f"    KG-L Theme Affinity     : {low_row['kgl_theme_affinity']:.4f} (Shared Themes: {int(low_row['shared_theme_count'])})")
        if pd.notna(low_row["emotion_cosine_similarity"]):
            print(f"    Emotion Cosine Sim      : {low_row['emotion_cosine_similarity']:.4f}")
        print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PlotGraph Phase 4A: Feature Engineering Execution Pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/features.yaml"),
        help="Path to feature engineering configuration YAML.",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=None,
        help="Override recency half-life in days (default from config).",
    )
    parser.add_argument(
        "--skip-profile-save",
        action="store_true",
        help="Skip writing user profile artifacts to disk.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    input_paths = config.get("input_paths", {})
    output_paths = config.get("output_paths", {})

    train_path = Path(input_paths.get("train_split", "data/processed/ml-1m/splits/train.parquet"))
    cand_path = Path(input_paths.get("candidate_pool", "data/processed/ml-1m/candidates/top100_candidates.parquet"))
    meta_path = Path(input_paths.get("metadata", "data/processed/ml-1m/metadata/movies_with_tmdb.parquet"))
    sem_path = Path(input_paths.get("semantic_validated", "data/processed/ml-1m/kg_l/full_extraction/semantic_validated.parquet"))
    emo_path = Path(input_paths.get("emotion_vectors", "data/processed/ml-1m/kg_l/full_extraction/emotion_vectors.parquet"))

    out_features_path = Path(output_paths.get("features_parquet", "data/processed/ml-1m/features/kg_candidate_features.parquet"))
    out_summary_path = Path(output_paths.get("features_summary", "data/processed/ml-1m/features/kg_feature_summary.json"))
    out_profiles_dir = Path(output_paths.get("user_profiles_dir", "data/processed/ml-1m/features/user_profiles"))

    half_life_days = (
        args.half_life_days
        if args.half_life_days is not None
        else float(config.get("recency", {}).get("half_life_days", DEFAULT_HALF_LIFE_DAYS))
    )
    pos_thresh = float(config.get("filtering", {}).get("positive_threshold", DEFAULT_POSITIVE_THRESHOLD))

    # -------------------------------------------------------------------------
    # 1. Leakage Guard: Check that only TRAIN split is provided
    # -------------------------------------------------------------------------
    logger.info("Executing temporal and data leakage pre-flight checks...")
    if "val" in str(train_path).lower() or "test" in str(train_path).lower():
        raise ValueError(
            f"LEAKAGE VIOLATION: train_path '{train_path}' references validation or test data! "
            "Profiles must be constructed exclusively from TRAIN split."
        )

    # 2. Load Input Artifacts
    logger.info(f"Loading TRAIN split from {train_path}...")
    train_df = pd.read_parquet(train_path)

    logger.info(f"Loading candidate pool from {cand_path}...")
    candidate_df = pd.read_parquet(cand_path)

    logger.info(f"Loading movie metadata from {meta_path}...")
    meta_df = pd.read_parquet(meta_path)

    logger.info(f"Loading semantic extractions from {sem_path}...")
    if sem_path.suffix == ".parquet":
        sem_df = pd.read_parquet(sem_path)
    else:
        sem_df = pd.read_json(sem_path, lines=True)

    logger.info(f"Loading continuous emotion vectors from {emo_path}...")
    if emo_path.suffix == ".parquet":
        emo_df = pd.read_parquet(emo_path)
    else:
        emo_df = pd.read_json(emo_path, lines=True)

    # 3. Build Movie Index
    t0 = time.time()
    logger.info("Building in-memory MovieEntityIndex...")
    movie_index = MovieEntityIndex.from_dataframes(
        metadata_df=meta_df,
        semantic_df=sem_df,
        emotion_df=emo_df,
    )
    logger.info(f"Indexed {len(movie_index):,} movies in {time.time() - t0:.2f}s.")

    # 4. Build Leakage-Safe User Profiles
    t1 = time.time()
    logger.info(f"Building user profiles (recency half-life: {half_life_days} days)...")
    user_profiles = build_user_profiles(
        train_df=train_df,
        movie_index=movie_index,
        half_life_days=half_life_days,
        positive_threshold=pos_thresh,
    )
    logger.info(f"Constructed profiles for {len(user_profiles):,} users in {time.time() - t1:.2f}s.")

    # 5. Build Candidate Feature Matrix
    t2 = time.time()
    logger.info(f"Constructing candidate feature matrix for {len(candidate_df):,} rows...")
    features_df = build_candidate_features(
        candidate_df=candidate_df,
        user_profiles=user_profiles,
        movie_index=movie_index,
        validate=True,
    )
    logger.info(f"Feature matrix built in {time.time() - t2:.2f}s.")

    # 6. Save Features Table
    out_features_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Serializing candidate features to {out_features_path}...")
    features_df.to_parquet(out_features_path, index=False)
    file_size_mb = out_features_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {out_features_path} ({file_size_mb:.2f} MB).")

    # 7. Generate and Save Summary
    summary = generate_feature_summary(
        features_df=features_df,
        user_profiles=user_profiles,
        movie_index=movie_index,
        config=config,
    )
    save_feature_summary(summary, out_summary_path)

    # 8. Optionally Save User Profiles
    if not args.skip_profile_save:
        logger.info(f"Persisting reusable user profile artifacts to {out_profiles_dir}...")
        user_profiles.save_to_dir(out_profiles_dir)

    # 9. Print Sanity Analysis
    print_sanity_analysis(features_df, summary, user_idx=0)
    logger.info("Phase 4A Feature Engineering completed successfully!")


if __name__ == "__main__":
    main()
