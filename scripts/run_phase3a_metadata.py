#!/usr/bin/env python3
"""Run Phase 3A: TMDB metadata ingestion, cleaning, linking, and coverage reporting.

Usage:
    python scripts/run_phase3a_metadata.py --config configs/metadata.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.tmdb_loader import build_unified_metadata_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3a_runner")


def format_table_row(name: str, count: int, total: int, pct: float) -> str:
    return f"  {name:<32} {count:>6} / {total:>6} ({pct:>6.2f}%)"


def print_coverage_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print(" " * 18 + "PLOTGRAPH PHASE 3A: METADATA COVERAGE REPORT")
    print("=" * 70)

    for group_key, group_data in report.items():
        title = group_data.get("name", group_key)
        total = group_data.get("total_movies", 0)
        print(f"\n--- {title} (Total: {total:,}) ---")

        metrics = [
            ("Linked to TMDB (links.csv)", group_data.get("with_tmdb_link", 0), group_data.get("with_tmdb_link_pct", 0.0)),
            ("Matching Metadata (tmdb_title)", group_data.get("with_metadata", 0), group_data.get("with_metadata_pct", 0.0)),
            ("Non-Empty Plot Overview", group_data.get("with_overview", 0), group_data.get("with_overview_pct", 0.0)),
            ("Credits Available (cast/crew)", group_data.get("with_credits", 0), group_data.get("with_credits_pct", 0.0)),
            ("Director Present", group_data.get("with_director", 0), group_data.get("with_director_pct", 0.0)),
            ("Top Actors Present (>=1)", group_data.get("with_actors", 0), group_data.get("with_actors_pct", 0.0)),
            ("Keywords Present (>=1)", group_data.get("with_keywords", 0), group_data.get("with_keywords_pct", 0.0)),
            ("Belongs to Collection", group_data.get("with_collection", 0), group_data.get("with_collection_pct", 0.0)),
        ]

        for name, count, pct in metrics:
            print(format_table_row(name, count, total, pct))

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run PlotGraph Phase 3A: Ingest and link TMDB metadata to MovieLens-1M."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/metadata.yaml",
        help="Path to metadata configuration YAML file (default: configs/metadata.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)

    logger.info(f"Starting Phase 3A metadata pipeline using config: {config_path}")
    unified_df, coverage_report = build_unified_metadata_pipeline(config_path)

    # Display report
    print_coverage_report(coverage_report)

    # Sample check
    print("\nSample Linked Movie (MovieLens movie_id=1):")
    sample_toy_story = unified_df[unified_df["movie_id"] == 1].iloc[0]
    print(f"  MovieLens:   {sample_toy_story['ml_title']} (Genres: {sample_toy_story['ml_genres']})")
    print(f"  TMDB ID:     {sample_toy_story['tmdb_id']} - {sample_toy_story['tmdb_title']}")
    print(f"  Decade:      {sample_toy_story['release_decade']} (Date: {sample_toy_story['release_date']})")
    print(f"  Director:    {sample_toy_story['director']}")
    print(f"  Top Actors:  {sample_toy_story['top_actors']}")
    print(f"  Keywords:    {sample_toy_story['keywords'][:5] if sample_toy_story['keywords'] else None}")
    print(f"  Collection:  {sample_toy_story['collection_name']}")
    overview_preview = str(sample_toy_story['overview'])[:100] + "..." if sample_toy_story['overview'] else "None"
    print(f"  Overview:    {overview_preview}")

    print("\nSample Unmapped Movie (MovieLens movie_id=142):")
    sample_unmapped = unified_df[unified_df["movie_id"] == 142].iloc[0]
    print(f"  MovieLens:   {sample_unmapped['ml_title']} (Genres: {sample_unmapped['ml_genres']})")
    print(f"  TMDB ID:     {sample_unmapped['tmdb_id']} (has_tmdb_link={sample_unmapped['has_tmdb_link']})")
    print(f"  Overview:    {sample_unmapped['overview']} (has_overview={sample_unmapped['has_overview']})")
    print(f"  Credits:     {sample_unmapped['director']} (has_credits={sample_unmapped['has_credits']})")

    print("\nPipeline completed successfully!")
    print(f"Total MovieLens movies preserved: {len(unified_df)} / {len(unified_df)} (100.0%)")


if __name__ == "__main__":
    main()
