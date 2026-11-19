"""End-to-end data processing pipeline for Phase 1 (MovieLens-1M)."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.data.loader import load_ml1m_ratings, load_ml1m_movies
from src.data.id_mapper import IDMapper
from src.data.splitter import split_chronological_per_user, verify_no_temporal_leakage


def run_pipeline(config_path: str = "configs/data.yaml") -> None:
    print(f"=== Running Phase 1 Data Pipeline with config: {config_path} ===")
    config = load_config(config_path)

    ml1m_dir = Path(config["paths"]["ml1m_raw_dir"])
    ratings_path = ml1m_dir / config["ml1m"]["ratings_file"]
    movies_path = ml1m_dir / config["ml1m"]["movies_file"]

    if not ratings_path.is_file() or not movies_path.is_file():
        print(f"[ERROR] MovieLens-1M raw files not found at {ml1m_dir.resolve()}.")
        print("Please download the dataset first using: python scripts/download_ml1m.py")
        sys.exit(1)

    threshold = float(config["splitting"]["positive_threshold"])
    print(f"[1/5] Loading MovieLens-1M ratings (positive threshold >= {threshold})...")
    ratings_df = load_ml1m_ratings(ratings_path, positive_threshold=threshold)
    print(f"      Loaded {len(ratings_df):,} ratings across {ratings_df['user_id'].nunique():,} users and {ratings_df['movie_id'].nunique():,} movies.")

    print("[2/5] Loading MovieLens-1M catalog...")
    movies_df = load_ml1m_movies(movies_path, encoding=config["ml1m"]["encoding"])
    print(f"      Loaded {len(movies_df):,} catalog movies.")

    print("[3/5] Fitting IDMapper and generating internal indices...")
    mapper = IDMapper()
    mapper.fit(ratings_df, movies_df)
    ratings_indexed = mapper.transform(ratings_df)
    movies_indexed = mapper.transform(movies_df)

    id_maps_dir = Path(config["paths"]["id_maps_dir"])
    id_maps_dir.mkdir(parents=True, exist_ok=True)
    mapper.save(id_maps_dir)
    print(f"      Saved ID mappings to {id_maps_dir.resolve()} ({mapper.num_users:,} users, {mapper.num_movies:,} movies).")

    print("[4/5] Executing chronological 80/10/10 per-user split...")
    train_ratio = float(config["splitting"]["train_ratio"])
    val_ratio = float(config["splitting"]["val_ratio"])
    test_ratio = float(config["splitting"]["test_ratio"])

    splits = split_chronological_per_user(
        ratings_indexed,
        user_col="user_id",
        timestamp_col="timestamp",
        item_col="movie_id",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    print("      Verifying temporal leakage...")
    verify_no_temporal_leakage(splits, user_col="user_id", timestamp_col="timestamp")
    print("      [PASSED] No temporal leakage detected across train, val, and test splits.")

    print("[5/5] Saving processed splits...")
    splits_dir = Path(config["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    splits["train"].to_parquet(splits_dir / "train.parquet", index=False)
    splits["val"].to_parquet(splits_dir / "val.parquet", index=False)
    splits["test"].to_parquet(splits_dir / "test.parquet", index=False)
    movies_indexed.to_parquet(Path(config["paths"]["processed_dir"]) / "movies.parquet", index=False)

    print("\n=== Split Summary Statistics ===")
    total_len = len(ratings_indexed)
    for split_name in ["train", "val", "test"]:
        df_split = splits[split_name]
        pos_count = int(df_split["positive"].sum())
        pos_pct = (pos_count / len(df_split)) * 100 if len(df_split) > 0 else 0
        split_pct = (len(df_split) / total_len) * 100
        print(f"  - {split_name.upper():<5}: {len(df_split):>8,} interactions ({split_pct:>5.1f}%) | "
              f"Positives (r >= {threshold}): {pos_count:>7,} ({pos_pct:>4.1f}%) | "
              f"Users: {df_split['user_id'].nunique():>5,} | "
              f"Items: {df_split['movie_id'].nunique():>5,}")

    print(f"\n[SUCCESS] Phase 1 pipeline completed successfully. Artifacts stored in {splits_dir.parent.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 data pipeline.")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to YAML configuration file.")
    args = parser.parse_args()
    run_pipeline(args.config)
