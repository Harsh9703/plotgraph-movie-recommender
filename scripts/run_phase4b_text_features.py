#!/usr/bin/env python3
"""PlotGraph Phase 4B: Text Semantic Features Execution Pipeline.

Builds leakage-safe TF-IDF and SBERT text-semantic features for all 604,000 candidate rows:
1. Validates strict TRAIN-only profile construction (zero VAL/TEST leakage).
2. Fits and caches catalog movie TF-IDF representations.
3. Encodes and caches SBERT dense embeddings (all-MiniLM-L6-v2) on CPU.
4. Constructs recency-weighted user TF-IDF and SBERT profiles from positive TRAIN history.
5. Appends text similarity features (cosine, dot product, availability flags).
6. Optionally builds a reusable FAISS index for semantic neighbor retrieval.
7. Validates exact row preservation (604,000 rows) and column schemas.
8. Serializes output feature table and detailed summary metrics.
9. Prints interpretable candidate matching examples.
"""

from __future__ import annotations

import os

# Ensure clean offline execution and avoid threading/OpenMP clashes on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
    ALL_COLUMNS,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_POSITIVE_THRESHOLD,
    PHASE4A_COLUMNS,
    TEXT_FEATURES,
)
from src.features.text_encoder import (
    FAISSIndexHelper,
    SBERTMovieEncoder,
    TFIDFMovieEncoder,
)
from src.features.text_feature_builder import build_text_candidate_features
from src.features.text_profiler import build_text_user_profiles
from src.features.text_summary import (
    generate_text_feature_summary,
    save_text_feature_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_phase4b_text_features")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_text_sanity_analysis(
    features_df: pd.DataFrame,
    summary: Dict[str, Any],
    metadata_df: pd.DataFrame,
    faiss_helper: Optional[FAISSIndexHelper] = None,
    user_idx: int = 0,
) -> None:
    """Print informative sanity statistics and interpretable candidate examples."""
    print("\n" + "=" * 70)
    print("         PLOTGRAPH PHASE 4B: TEXT SEMANTIC FEATURES REPORT")
    print("=" * 70)
    print(f"Total Rows Generated          : {len(features_df):,}")
    print(f"Total Evaluated Users         : {features_df['user_idx'].nunique():,}")
    print(f"Total Feature Columns         : {len(features_df.columns)} (Phase 4A: 60, New Text: 8)")
    print(f"SBERT Model                   : {summary['sbert_model_id']} (dim={summary['sbert_dimension']})")
    print(f"TF-IDF Vocabulary Size        : {summary['tfidf_vocab_size']:,} terms")
    print("-" * 70)
    print("COVERAGE METRICS:")
    cov = summary["coverage_metrics"]
    cat_cov = summary["catalog_overview_coverage"]
    print(f"  Catalog Movie Overviews     : {cat_cov['usable_overviews']:,} / {cat_cov['total_catalog']:,} ({cat_cov['coverage_pct']}%)")
    print(f"  TF-IDF Both Available       : {cov['tfidf']['both_available_rows']:,} / {len(features_df):,} ({cov['tfidf']['both_available_pct']}%)")
    print(f"  SBERT Both Available        : {cov['sbert']['both_available_rows']:,} / {len(features_df):,} ({cov['sbert']['both_available_pct']}%)")
    print("-" * 70)
    print("TEXT SIMILARITY DISTRIBUTIONS:")
    stats = summary["descriptive_statistics"]
    for feat in ["tfidf_cosine_similarity", "sbert_cosine_similarity"]:
        if feat in stats:
            s = stats[feat]
            print(f"  {feat:<26}: mean={s['mean']:.4f}, std={s['std']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}, p50={s['p50']:.4f}, non_zero={s['non_zero_pct']}%")
    print("=" * 70)

    # Interpretable candidate examples for user_idx
    u_rows = features_df[features_df["user_idx"] == user_idx]
    if not u_rows.empty:
        # Title lookup
        title_map = dict(zip(metadata_df["movie_idx"], metadata_df["ml_title"]))

        print(f"\nINTERPRETABLE TEXT MATCHING EXAMPLES (User Index: {user_idx}):")
        print(f"  Positive TRAIN items: {int(u_rows.iloc[0]['user_positive_train_count'])}")

        # Top TF-IDF candidate
        top_tfidf_idx = u_rows["tfidf_cosine_similarity"].idxmax()
        row_tfidf = u_rows.loc[top_tfidf_idx]
        t_m_idx = int(row_tfidf["movie_idx"])
        print("\n  [HIGHEST TF-IDF SIMILARITY CANDIDATE]")
        print(f"    Movie                   : {title_map.get(t_m_idx, f'Movie {t_m_idx}')} (Rank: {int(row_tfidf['candidate_rank'])})")
        print(f"    EASE Score              : {row_tfidf['ease_score']:.4f}")
        print(f"    TF-IDF Cosine Sim       : {row_tfidf['tfidf_cosine_similarity']:.4f}")
        print(f"    SBERT Cosine Sim        : {row_tfidf['sbert_cosine_similarity']:.4f}")

        # Top SBERT candidate
        top_sbert_idx = u_rows["sbert_cosine_similarity"].idxmax()
        row_sbert = u_rows.loc[top_sbert_idx]
        s_m_idx = int(row_sbert["movie_idx"])
        print("\n  [HIGHEST SBERT SIMILARITY CANDIDATE]")
        print(f"    Movie                   : {title_map.get(s_m_idx, f'Movie {s_m_idx}')} (Rank: {int(row_sbert['candidate_rank'])})")
        print(f"    EASE Score              : {row_sbert['ease_score']:.4f}")
        print(f"    SBERT Cosine Sim        : {row_sbert['sbert_cosine_similarity']:.4f}")
        print(f"    TF-IDF Cosine Sim       : {row_sbert['tfidf_cosine_similarity']:.4f}")

        # Lowest SBERT candidate
        low_sbert_idx = u_rows["sbert_cosine_similarity"].idxmin()
        row_low = u_rows.loc[low_sbert_idx]
        l_m_idx = int(row_low["movie_idx"])
        print("\n  [LOWEST SBERT SIMILARITY CANDIDATE]")
        print(f"    Movie                   : {title_map.get(l_m_idx, f'Movie {l_m_idx}')} (Rank: {int(row_low['candidate_rank'])})")
        print(f"    EASE Score              : {row_low['ease_score']:.4f}")
        print(f"    SBERT Cosine Sim        : {row_low['sbert_cosine_similarity']:.4f}")
        print(f"    TF-IDF Cosine Sim       : {row_low['tfidf_cosine_similarity']:.4f}")

        # Nearest catalog neighbors demonstration using FAISS/SBERT
        if faiss_helper is not None:
            # Query with Toy Story (movie_idx 0)
            print("\n  [SBERT/FAISS NEAREST SEMANTIC NEIGHBORS (Toy Story, ID 1)]:")
            toy_story_emb = faiss_helper.valid_embeddings[0] if len(faiss_helper.valid_embeddings) > 0 else None
            if toy_story_emb is not None:
                neighbors = faiss_helper.search_nearest(toy_story_emb, top_k=6)
                for n_rank, (n_m_idx, n_sim) in enumerate(neighbors[1:6], 1):  # Skip self
                    print(f"    {n_rank}. {title_map.get(n_m_idx, f'Movie {n_m_idx}')} (cosine sim: {n_sim:.4f})")

        print("=" * 70 + "\n")


def main() -> None:
    t_start = time.time()
    parser = argparse.ArgumentParser(
        description="PlotGraph Phase 4B: Text Semantic Features Execution Pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/text_features.yaml"),
        help="Path to text features configuration YAML.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of TF-IDF and SBERT embeddings even if cached.",
    )
    parser.add_argument(
        "--skip-faiss",
        action="store_true",
        help="Skip FAISS index construction.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    input_paths = config.get("input_paths", {})
    output_paths = config.get("output_paths", {})

    p4a_path = Path(input_paths.get("phase4a_features", "data/processed/ml-1m/features/kg_candidate_features.parquet"))
    meta_path = Path(input_paths.get("metadata", "data/processed/ml-1m/metadata/movies_with_tmdb.parquet"))
    train_path = Path(input_paths.get("train_split", "data/processed/ml-1m/splits/train.parquet"))

    out_features_path = Path(output_paths.get("features_parquet", "data/processed/ml-1m/features/kg_text_candidate_features.parquet"))
    out_summary_path = Path(output_paths.get("features_summary", "data/processed/ml-1m/features/text_feature_summary.json"))
    text_emb_dir = Path(output_paths.get("text_embeddings_dir", "data/processed/ml-1m/features/text_embeddings"))
    tfidf_dir = Path(output_paths.get("tfidf_dir", "data/processed/ml-1m/features/tfidf"))

    tfidf_cfg = config.get("tfidf", {})
    sbert_cfg = config.get("sbert", {})
    recency_cfg = config.get("recency", {})
    half_life_days = float(recency_cfg.get("half_life_days", DEFAULT_HALF_LIFE_DAYS))
    pos_thresh = float(config.get("filtering", {}).get("positive_threshold", DEFAULT_POSITIVE_THRESHOLD))
    sbert_model_id = str(sbert_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))

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
    logger.info(f"Loading Phase 4A candidate features from {p4a_path}...")
    p4a_df = pd.read_parquet(p4a_path)

    logger.info(f"Loading movie metadata from {meta_path}...")
    meta_df = pd.read_parquet(meta_path)

    logger.info(f"Loading TRAIN split strictly from {train_path}...")
    train_df = pd.read_parquet(train_path)

    catalog_size = len(meta_df)
    overviews_by_idx: Dict[int, str] = {}
    for _, row in meta_df.iterrows():
        m_idx = int(row["movie_idx"])
        ov = row.get("overview")
        overviews_by_idx[m_idx] = str(ov).strip() if pd.notna(ov) and str(ov).strip() else ""

    usable_overviews = sum(1 for text in overviews_by_idx.values() if text)
    logger.info(f"Identified {usable_overviews:,} usable movie plot overviews out of {catalog_size:,} catalog movies.")

    # -------------------------------------------------------------------------
    # 3. Fit or Load TF-IDF Encoder
    # -------------------------------------------------------------------------
    tfidf_encoder = None
    if not args.force_recompute and (tfidf_dir / "tfidf_vectorizer.joblib").exists():
        logger.info(f"Loading cached TF-IDF encoder from {tfidf_dir}...")
        tfidf_encoder = TFIDFMovieEncoder.load(tfidf_dir)
    else:
        logger.info("Fitting TF-IDF vectorizer on catalog movie overviews...")
        ngram_range = tuple(tfidf_cfg.get("ngram_range", [1, 2]))
        tfidf_encoder = TFIDFMovieEncoder(
            ngram_range=ngram_range,
            min_df=int(tfidf_cfg.get("min_df", 2)),
            max_df=float(tfidf_cfg.get("max_df", 0.8)),
            stop_words=str(tfidf_cfg.get("stop_words", "english")),
            sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", False)),
        )
        tfidf_encoder.fit_transform(overviews_by_idx, catalog_size=catalog_size)
        tfidf_encoder.save(tfidf_dir)

    # -------------------------------------------------------------------------
    # 4. Encode or Load SBERT Embeddings
    # -------------------------------------------------------------------------
    sbert_encoder = None
    if not args.force_recompute and (text_emb_dir / "movie_sbert_embeddings.npy").exists():
        logger.info(f"Loading cached SBERT embeddings from {text_emb_dir}...")
        sbert_encoder = SBERTMovieEncoder.load(text_emb_dir, model_name=sbert_model_id)
    else:
        logger.info(f"Encoding catalog movie overviews with SBERT '{sbert_model_id}'...")
        sbert_encoder = SBERTMovieEncoder(
            model_name=sbert_model_id,
            device=str(sbert_cfg.get("device", "cpu")),
        )
        sbert_encoder.encode_catalog(
            overviews_by_idx=overviews_by_idx,
            catalog_size=catalog_size,
            batch_size=int(sbert_cfg.get("batch_size", 64)),
            show_progress_bar=True,
        )
        sbert_encoder.save(text_emb_dir)

    # -------------------------------------------------------------------------
    # 5. Build Leakage-Safe Text User Profiles (TRAIN only)
    # -------------------------------------------------------------------------
    logger.info(f"Building text user profiles from positive TRAIN history (half-life: {half_life_days} days)...")
    text_user_profiles = build_text_user_profiles(
        train_df=train_df,
        tfidf_encoder=tfidf_encoder,
        sbert_encoder=sbert_encoder,
        half_life_days=half_life_days,
        positive_threshold=pos_thresh,
        total_users=6040,
    )

    # -------------------------------------------------------------------------
    # 6. Optional FAISS Index
    # -------------------------------------------------------------------------
    faiss_helper = None
    if not args.skip_faiss and config.get("faiss", {}).get("enabled", True):
        logger.info("Initializing FAISS / NumPy semantic neighbor index...")
        faiss_helper = FAISSIndexHelper(dimension=sbert_encoder.embedding_dimension)
        faiss_helper.build_index(
            embeddings_matrix=sbert_encoder.embeddings_matrix,
            has_embedding_mask=sbert_encoder.has_embedding_mask,
        )

    # -------------------------------------------------------------------------
    # 7. Append Text Semantic Candidate Features
    # -------------------------------------------------------------------------
    logger.info("Appending TF-IDF and SBERT text features to candidate rows...")
    features_df = build_text_candidate_features(
        phase4a_df=p4a_df,
        text_user_profiles=text_user_profiles,
        tfidf_encoder=tfidf_encoder,
        sbert_encoder=sbert_encoder,
        validate=True,
    )

    # -------------------------------------------------------------------------
    # 8. Serialize Output Table & Summary
    # -------------------------------------------------------------------------
    out_features_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Serializing Phase 4B feature matrix to {out_features_path}...")
    features_df.to_parquet(out_features_path, index=False)
    file_size_mb = out_features_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {out_features_path} ({file_size_mb:.2f} MB).")

    runtime_seconds = time.time() - t_start
    summary = generate_text_feature_summary(
        features_df=features_df,
        tfidf_vocab_size=tfidf_encoder.vocab_size,
        sbert_dimension=sbert_encoder.embedding_dimension,
        sbert_model_id=sbert_model_id,
        catalog_overview_count=usable_overviews,
        total_catalog_size=catalog_size,
        runtime_seconds=runtime_seconds,
        config=config,
    )
    save_text_feature_summary(summary, out_summary_path)

    # -------------------------------------------------------------------------
    # 9. Print Sanity Analysis
    # -------------------------------------------------------------------------
    print_text_sanity_analysis(
        features_df=features_df,
        summary=summary,
        metadata_df=meta_df,
        faiss_helper=faiss_helper,
        user_idx=0,
    )
    logger.info(f"Phase 4B Text Features completed successfully in {runtime_seconds:.1f}s!")


if __name__ == "__main__":
    main()
