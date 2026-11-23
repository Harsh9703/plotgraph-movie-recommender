#!/usr/bin/env python3
"""PlotGraph Phase 4C: NRC Lexicon Emotion Baseline Execution Pipeline.

Builds leakage-safe NRC lexicon emotion baseline features for all 604,000 candidate rows:
1. Validates strict TRAIN-only user profile construction (zero VAL/TEST leakage).
2. Computes and caches catalog movie NRC 8D continuous emotion vectors from plot overviews.
3. Constructs recency-weighted user NRC emotion profiles from positive TRAIN history.
4. Computes candidate-level NRC emotion features (cosine, dot, L1, L2, raw 8D vectors, coverage flags).
5. Produces descriptive comparisons between Qwen and NRC movie-level emotion vectors.
6. Validates exact candidate row count (604,000 rows) and column preservation (91 total columns).
7. Serializes final full candidate feature matrix and detailed summary report.
8. Prints interpretable movie-level and user-candidate sanity matching examples.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.features.constants import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_POSITIVE_THRESHOLD,
    FULL_COLUMNS,
    NRC_EMOTION_FEATURES,
    PHASE4B_COLUMNS,
    PLUTCHIK_EMOTIONS,
)
from src.features.nrc_feature_builder import build_nrc_candidate_features
from src.features.nrc_lexicon import NRCLexicon
from src.features.nrc_profiler import build_nrc_user_profiles
from src.features.nrc_summary import (
    compute_qwen_vs_nrc_comparison,
    generate_nrc_feature_summary,
    save_nrc_feature_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_phase4c_nrc_features")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_nrc_sanity_analysis(
    full_df: pd.DataFrame,
    summary: Dict[str, Any],
    metadata_df: pd.DataFrame,
    movie_nrc_df: pd.DataFrame,
    qwen_vectors_path: Path,
    user_idx: int = 0,
) -> None:
    """Print informative sanity statistics, Qwen-vs-NRC comparisons, and candidate examples."""
    print("\n" + "=" * 78)
    print("       PLOTGRAPH PHASE 4C: NRC EMOTION BASELINE REPORT & COMPARISON")
    print("=" * 78)
    print(f"Total Rows Generated          : {len(full_df):,}")
    print(f"Total Evaluated Users         : {full_df['user_idx'].nunique():,}")
    print(f"Total Feature Columns         : {len(full_df.columns)} (Phase 4B: 68, New NRC: 23)")
    print(f"Lexicon Source                : {summary['lexicon_metadata']['lexicon_name']}")
    print(f"Attribution                   : {summary['lexicon_metadata']['authors']} ({summary['lexicon_metadata']['package_version']})")
    print("-" * 78)
    print("COVERAGE METRICS:")
    cat_cov = summary["catalog_coverage"]
    cand_cov = summary["candidate_pool_coverage"]
    print(f"  Catalog Movie Overviews     : {cat_cov['usable_overviews']:,} / {cat_cov['total_catalog']:,} ({cat_cov['usable_overviews_pct']}%)")
    print(f"  Catalog Movies with NRC Vec : {cat_cov['movies_with_valid_nrc_vector']:,} / {cat_cov['total_catalog']:,} ({cat_cov['movies_with_valid_nrc_vector_pct']}%)")
    print(f"  Avg Matched Tokens / Movie  : {cat_cov['average_matched_tokens_per_movie']:.2f} tokens")
    print(f"  Candidate NRC Available     : {cand_cov['candidate_nrc_available_rows']:,} / {len(full_df):,} ({cand_cov['candidate_nrc_available_pct']}%)")
    print(f"  User NRC Profile Available  : {cand_cov['user_nrc_profile_available_rows']:,} / {len(full_df):,} ({cand_cov['user_nrc_profile_available_pct']}%)")
    print(f"  Both Available (Similarity) : {cand_cov['both_available_rows']:,} / {len(full_df):,} ({cand_cov['both_available_pct']}%)")
    print("-" * 78)
    print("NRC COMPATIBILITY DISTRIBUTIONS:")
    stats = summary["descriptive_statistics"]
    for feat in ["nrc_emotion_cosine_similarity", "nrc_emotion_dot_product", "nrc_emotion_l1_distance", "nrc_emotion_l2_distance"]:
        if feat in stats:
            s = stats[feat]
            print(f"  {feat:<30}: mean={s['mean']:.4f}, std={s['std']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}, p50={s['p50']:.4f}")

    # Descriptive comparison between Qwen and NRC
    q_comp = summary.get("qwen_vs_nrc_comparison", {})
    if "common_movies_count" in q_comp:
        print("-" * 78)
        print(f"DESCRIPTIVE COMPARISON: QWEN vs NRC ({q_comp['common_movies_count']:,} common movies):")
        print(f"  Average Cosine Similarity   : {q_comp['average_cosine_similarity']:.4f} (std={q_comp['cosine_similarity_std']:.4f}, median={q_comp['cosine_similarity_median']:.4f})")
        print("  Per-Emotion Comparison:")
        print(f"    {'Emotion':<14} {'NRC Mean':<12} {'NRC Std':<12} {'Qwen Mean':<12} {'Qwen Std':<12} {'Pearson r':<10}")
        print("    " + "-" * 68)
        for emo in PLUTCHIK_EMOTIONS:
            pe = q_comp["per_emotion_comparison"].get(emo, {})
            r_str = f"{pe.get('pearson_correlation'):.4f}" if pe.get("pearson_correlation") is not None else "N/A"
            print(f"    {emo:<14} {pe.get('nrc_mean', 0.0):<12.4f} {pe.get('nrc_std', 0.0):<12.4f} {pe.get('qwen_mean', 0.0):<12.4f} {pe.get('qwen_std', 0.0):<12.4f} {r_str:<10}")

    print("=" * 78)

    # 1. Movie-level Sanity Examples
    title_map = dict(zip(metadata_df["movie_idx"], metadata_df["ml_title"]))
    ov_map = dict(zip(metadata_df["movie_idx"], metadata_df["overview"]))

    # Load Qwen vectors for comparison if available
    qwen_dict: Dict[int, np.ndarray] = {}
    if qwen_vectors_path.exists():
        qwen_df = pd.read_parquet(qwen_vectors_path)
        for _, qrow in qwen_df.iterrows():
            q_idx = int(qrow["movie_idx"])
            q_vec = qrow[list(PLUTCHIK_EMOTIONS)].to_numpy(dtype=np.float32)
            qwen_dict[q_idx] = q_vec

    sample_movie_indices = [0, 259, 1192]  # Toy Story (0), Star Wars IV (259), Star Wars V (1192)
    # Check if indices exist
    valid_samples = [m for m in sample_movie_indices if m in title_map]

    print("\nREPRESENTATIVE MOVIE EMOTION REPRESENTATIONS (NRC vs Qwen):")
    for s_m_idx in valid_samples:
        m_title = title_map.get(s_m_idx, f"Movie {s_m_idx}")
        m_ov = ov_map.get(s_m_idx, "")
        nrc_row = movie_nrc_df[movie_nrc_df["movie_idx"] == s_m_idx]
        if not nrc_row.empty:
            r = nrc_row.iloc[0]
            m_matched = int(r["candidate_nrc_matched_token_count"])
            nrc_v = [float(r[f"candidate_nrc_{e}"]) for e in PLUTCHIK_EMOTIONS]
            qwen_v = qwen_dict.get(s_m_idx)

            print(f"\n  [MOVIE: {m_title}] (ID: {s_m_idx + 1})")
            ov_snippet = (m_ov[:110] + "...") if len(m_ov) > 110 else m_ov
            print(f"    Plot Overview     : \"{ov_snippet}\"")
            print(f"    Matched NRC Tokens: {m_matched}")
            print(f"    NRC Vector (sum=1): joy={nrc_v[0]:.2f}, trust={nrc_v[1]:.2f}, fear={nrc_v[2]:.2f}, surp={nrc_v[3]:.2f}, sad={nrc_v[4]:.2f}, disg={nrc_v[5]:.2f}, ang={nrc_v[6]:.2f}, antic={nrc_v[7]:.2f}")
            if qwen_v is not None:
                print(f"    Qwen Continuous   : joy={qwen_v[0]:.2f}, trust={qwen_v[1]:.2f}, fear={qwen_v[2]:.2f}, surp={qwen_v[3]:.2f}, sad={qwen_v[4]:.2f}, disg={qwen_v[5]:.2f}, ang={qwen_v[6]:.2f}, antic={qwen_v[7]:.2f}")

    # 2. User Candidate Matching Examples
    u_rows = full_df[full_df["user_idx"] == user_idx]
    if not u_rows.empty and u_rows.iloc[0]["user_nrc_profile_available"] == 1:
        u_row0 = u_rows.iloc[0]
        u_nrc = [float(u_row0[f"user_nrc_{e}"]) for e in PLUTCHIK_EMOTIONS]
        print(f"\nINTERPRETABLE NRC USER MATCHING EXAMPLES (User Index: {user_idx}):")
        print(f"  Positive TRAIN items: {int(u_row0['user_positive_train_count'])}")
        print(f"  User NRC Profile    : joy={u_nrc[0]:.2f}, trust={u_nrc[1]:.2f}, fear={u_nrc[2]:.2f}, surp={u_nrc[3]:.2f}, sad={u_nrc[4]:.2f}, disg={u_nrc[5]:.2f}, ang={u_nrc[6]:.2f}, antic={u_nrc[7]:.2f}")

        # Top NRC candidate
        top_nrc_idx = u_rows["nrc_emotion_cosine_similarity"].idxmax()
        row_top = u_rows.loc[top_nrc_idx]
        t_m_idx = int(row_top["movie_idx"])
        print("\n  [HIGHEST NRC EMOTION SIMILARITY CANDIDATE]")
        print(f"    Movie                   : {title_map.get(t_m_idx, f'Movie {t_m_idx}')} (Rank: {int(row_top['candidate_rank'])})")
        print(f"    EASE Score              : {row_top['ease_score']:.4f}")
        print(f"    NRC Cosine Sim          : {row_top['nrc_emotion_cosine_similarity']:.4f}")
        print(f"    NRC L2 Distance         : {row_top['nrc_emotion_l2_distance']:.4f}")
        print(f"    Qwen Emotion Cosine Sim : {row_top['emotion_cosine_similarity']:.4f}")

        # Lowest NRC candidate
        low_nrc_idx = u_rows["nrc_emotion_cosine_similarity"].idxmin()
        row_low = u_rows.loc[low_nrc_idx]
        l_m_idx = int(row_low["movie_idx"])
        print("\n  [LOWEST NRC EMOTION SIMILARITY CANDIDATE]")
        print(f"    Movie                   : {title_map.get(l_m_idx, f'Movie {l_m_idx}')} (Rank: {int(row_low['candidate_rank'])})")
        print(f"    EASE Score              : {row_low['ease_score']:.4f}")
        print(f"    NRC Cosine Sim          : {row_low['nrc_emotion_cosine_similarity']:.4f}")
        print(f"    NRC L2 Distance         : {row_low['nrc_emotion_l2_distance']:.4f}")
        print(f"    Qwen Emotion Cosine Sim : {row_low['emotion_cosine_similarity']:.4f}")

    print("=" * 78 + "\n")


def main() -> None:
    t_start = time.time()
    parser = argparse.ArgumentParser(
        description="PlotGraph Phase 4C: NRC Lexicon Emotion Baseline Execution Pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/nrc_features.yaml"),
        help="Path to NRC features configuration YAML.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of movie NRC vectors even if cached.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    input_paths = config.get("input_paths", {})
    output_paths = config.get("output_paths", {})

    p4b_path = Path(input_paths.get("phase4b_features", "data/processed/ml-1m/features/kg_text_candidate_features.parquet"))
    meta_path = Path(input_paths.get("metadata", "data/processed/ml-1m/metadata/movies_with_tmdb.parquet"))
    train_path = Path(input_paths.get("train_split", "data/processed/ml-1m/splits/train.parquet"))
    qwen_path = Path(input_paths.get("qwen_vectors", "data/processed/ml-1m/kg_l/full_extraction/emotion_vectors.parquet"))

    out_features_path = Path(output_paths.get("features_parquet", "data/processed/ml-1m/features/full_candidate_features.parquet"))
    out_summary_path = Path(output_paths.get("features_summary", "data/processed/ml-1m/features/nrc_feature_summary.json"))
    nrc_dir = Path(output_paths.get("nrc_dir", "data/processed/ml-1m/features/nrc"))

    recency_cfg = config.get("recency", {})
    half_life_days = float(recency_cfg.get("half_life_days", DEFAULT_HALF_LIFE_DAYS))
    pos_thresh = float(config.get("filtering", {}).get("positive_threshold", DEFAULT_POSITIVE_THRESHOLD))

    # -------------------------------------------------------------------------
    # 1. Leakage Guard Assertions
    # -------------------------------------------------------------------------
    logger.info("Executing temporal and data leakage pre-flight checks...")
    if "val" in str(train_path).lower() or "test" in str(train_path).lower():
        raise ValueError(
            f"LEAKAGE VIOLATION: train_path '{train_path}' references validation or test data! "
            "Profiles must be constructed exclusively from TRAIN split."
        )

    # 2. Load Inputs
    logger.info(f"Loading Phase 4B candidate features from {p4b_path}...")
    p4b_df = pd.read_parquet(p4b_path)

    logger.info(f"Loading movie metadata from {meta_path}...")
    meta_df = pd.read_parquet(meta_path)

    logger.info(f"Loading TRAIN split strictly from {train_path}...")
    train_df = pd.read_parquet(train_path)

    catalog_size = len(meta_df)

    # -------------------------------------------------------------------------
    # 3. Load or Compute Movie NRC Vectors
    # -------------------------------------------------------------------------
    movie_nrc_path = nrc_dir / "movie_nrc_vectors.parquet"
    if not args.force_recompute and movie_nrc_path.exists():
        logger.info(f"Loading cached movie NRC vectors from {movie_nrc_path}...")
        movie_nrc_df = NRCLexicon.load_vectors(nrc_dir)
    else:
        logger.info("Initializing NRC Emotion Lexicon...")
        lexicon = NRCLexicon()
        movie_nrc_df = lexicon.compute_catalog_vectors(meta_df, catalog_size=catalog_size)
        NRCLexicon.save_vectors(movie_nrc_df, nrc_dir)

    # -------------------------------------------------------------------------
    # 4. Build Leakage-Safe NRC User Profiles (TRAIN only)
    # -------------------------------------------------------------------------
    logger.info(f"Building NRC user profiles from positive TRAIN history (half-life: {half_life_days} days)...")
    nrc_user_profiles = build_nrc_user_profiles(
        train_df=train_df,
        movie_nrc_df=movie_nrc_df,
        half_life_days=half_life_days,
        positive_threshold=pos_thresh,
        total_users=6040,
    )

    # -------------------------------------------------------------------------
    # 5. Append NRC Emotion Features
    # -------------------------------------------------------------------------
    logger.info("Appending NRC emotion baseline features to candidate rows...")
    full_df = build_nrc_candidate_features(
        phase4b_df=p4b_df,
        nrc_user_profiles=nrc_user_profiles,
        movie_nrc_df=movie_nrc_df,
        validate=True,
    )

    # -------------------------------------------------------------------------
    # 6. Serialize Output Feature Matrix & Summary
    # -------------------------------------------------------------------------
    out_features_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Serializing final candidate feature matrix to {out_features_path}...")
    full_df.to_parquet(out_features_path, index=False)
    file_size_mb = out_features_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {out_features_path} ({file_size_mb:.2f} MB, {len(full_df.columns)} columns).")

    runtime_seconds = time.time() - t_start
    summary = generate_nrc_feature_summary(
        full_df=full_df,
        movie_nrc_df=movie_nrc_df,
        total_catalog_size=catalog_size,
        runtime_seconds=runtime_seconds,
        qwen_vectors_path=qwen_path,
        config=config,
    )
    save_nrc_feature_summary(summary, out_summary_path)

    # -------------------------------------------------------------------------
    # 7. Print Sanity Analysis
    # -------------------------------------------------------------------------
    print_nrc_sanity_analysis(
        full_df=full_df,
        summary=summary,
        metadata_df=meta_df,
        movie_nrc_df=movie_nrc_df,
        qwen_vectors_path=qwen_path,
        user_idx=0,
    )
    logger.info(f"Phase 4C NRC Emotion Baseline completed successfully in {runtime_seconds:.1f}s!")


if __name__ == "__main__":
    main()
