#!/usr/bin/env python3
"""Run Phase 3C: Prepare KG-L Extraction Contract, Prompt Templates, and Deterministic Sample Export.

Usage:
    python scripts/run_phase3c_kg_l_prep.py --config configs/kg_l.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge_graph.kg_l_contract import (
    KGL_ALL_RELATIONS,
    KGL_SEMANTIC_RELATIONS,
    PLUTCHIK_EMOTIONS,
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    export_sample_movies,
    format_extraction_prompt,
    get_synthetic_fixtures,
    parse_and_validate_extraction,
)
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3c_runner")


def main():
    parser = argparse.ArgumentParser(
        description="Run PlotGraph Phase 3C: Prepare KG-L extraction contract and sample dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kg_l.yaml",
        help="Path to KG-L configuration YAML file (default: configs/kg_l.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\n" + "=" * 75)
    print(" " * 12 + "PLOTGRAPH PHASE 3C: KG-L EXTRACTION CONTRACT & PREPARATION")
    print("=" * 75)

    # 1. Validate Ontology Configuration Consistency
    cfg_semantic = tuple(config["ontology"]["semantic_relations"])
    cfg_emotions = tuple(config["ontology"]["plutchik_emotions"])

    assert cfg_semantic == KGL_SEMANTIC_RELATIONS, (
        f"Semantic relations mismatch: config {cfg_semantic} vs code {KGL_SEMANTIC_RELATIONS}"
    )
    assert cfg_emotions == PLUTCHIK_EMOTIONS, (
        f"Emotions mismatch: config {cfg_emotions} vs code {PLUTCHIK_EMOTIONS}"
    )

    print("\n--- 1. Fixed KG-L Ontology Specification ---")
    print(f"  Semantic Relations (5):     {', '.join(KGL_SEMANTIC_RELATIONS)}")
    print(f"  Emotion Relation (1):       {config['ontology']['emotion_relation']}")
    print(f"  Plutchik Emotions (8):      {', '.join(PLUTCHIK_EMOTIONS)}")
    print(f"  Emotion Edge Threshold:     {config['ontology'].get('emotion_edge_threshold', DEFAULT_EMOTION_EDGE_THRESHOLD)}")
    print(f"  Target LLM Identifier:      {config['llm']['model_id']}")
    print(f"  Prompt Version:             {config['llm']['prompt_version']}")
    print(f"  Schema Version:             {config['llm']['schema_version']}")

    # 2. Local Schema & Parser Sanity Checks
    print("\n--- 2. Schema Validation Sanity Checks ---")
    fixtures = get_synthetic_fixtures()

    # Valid check
    valid_ext = parse_and_validate_extraction(fixtures["valid_fenced_json"])
    print(f"  [PASS] Clean parsing of valid fenced JSON: '{valid_ext.title}'")
    print(f"         Extracted themes: {valid_ext.themes}")
    print(f"         Extracted emotions (joy={valid_ext.emotions.joy}, trust={valid_ext.emotions.trust})")

    # Deduplication check
    dedup_ext = parse_and_validate_extraction(json.dumps(fixtures["duplicate_entities"]))
    print(f"  [PASS] Duplicate entity deduplication: {dedup_ext.themes} (collapsed to unique canonical keys)")

    # 3. Export Deterministic Sample Movies
    print("\n--- 3. Deterministic Sample Movie Export ---")
    sample_df = export_sample_movies(config)
    print(f"  Selected {len(sample_df)} movies (Seed={config['sampling']['random_seed']}).")
    print(f"  Output Parquet: {config['paths']['sample_movies_parquet']}")

    print("\n--- 4. Selected Sample Movies (N=30) ---")
    print(f"  {'Idx':<5} {'ML ID':<6} {'TMDB ID':<8} {'Decade':<8} {'Title':<32} {'Genres'}")
    print("  " + "-" * 73)
    for _, row in sample_df.iterrows():
        t_id = str(int(row["tmdb_id"])) if row["has_tmdb_link"] else "N/A"
        dec = str(row["release_decade"]) if row["release_decade"] else "N/A"
        print(f"  {row['movie_idx']:<5} {row['movie_id']:<6} {t_id:<8} {dec:<8} {str(row['ml_title'])[:30]:<32} {str(row['ml_genres'])[:20]}")

    # 4. Prompt Template Preview for First Sample Movie
    first_movie = sample_df.iloc[0]
    sys_prompt, user_prompt = format_extraction_prompt(
        title=str(first_movie["ml_title"]),
        overview=str(first_movie["overview"]),
        movie_id=int(first_movie["movie_id"]),
        movie_idx=int(first_movie["movie_idx"]),
        tmdb_id=int(first_movie["tmdb_id"]) if first_movie["has_tmdb_link"] else None,
        prompt_version=config["llm"]["prompt_version"],
    )

    print("\n--- 5. Formatted Extraction Prompt Preview (Sample Movie #1) ---")
    print(">>> [SYSTEM PROMPT] (truncated preview):")
    print("   " + "\n   ".join(sys_prompt.splitlines()[:6]))
    print("   ...")
    print("\n>>> [USER PROMPT]:")
    print("   " + "\n   ".join(user_prompt.splitlines()[:12]))
    print("   ...")

    print("\n" + "=" * 75)
    print("  STATUS: Phase 3C Contract, Schema & Sample Export Successfully Prepared.")
    print("  NOTE:   NO LLM inference or GPU execution was performed in this phase.")
    print("          Full Qwen2.5-7B-Instruct extraction will run in Phase 4.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
