#!/usr/bin/env python3
"""CLI Script for Phase 3G: KG-L Knowledge Graph Construction and Validation.

Executes the end-to-end pipeline:
1. Loads Phase 3A metadata and Phase 3F production extraction artifacts.
2. Constructs the LLM-enriched Knowledge Graph (KG-L) with canonicalized entities.
3. Generates direct evokesEmotion edges (score > 0.3) and reified EmotionAssociation entities.
4. Serializes KG-L to Turtle format (data/processed/ml-1m/graphs/kg_l.ttl).
5. Reloads and strictly validates the graph against contract requirements.
6. Saves validation report and statistics (data/processed/ml-1m/graphs/kg_l_validation.json).
7. Demonstrates representative SPARQL queries and cross-graph (KG-M + KG-L) compatibility.

Usage:
  python scripts/run_phase3g_kg_l.py --config configs/kg_l.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge_graph.kg_l_builder import (
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    EXPECTED_EVOKES_EMOTION_TOTAL,
    EXPECTED_PER_EMOTION_COUNTS,
    build_and_save_kg_l,
)
from src.knowledge_graph.kg_l_query_engine import (
    KGLQueryEngine,
    demonstrate_cross_graph_compatibility,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_phase3g_kg_l")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3G: Construct, serialize, validate, and query KG-L.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kg_l.yaml",
        help="Path to kg_l.yaml configuration file.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Path to movies_with_tmdb.parquet.",
    )
    parser.add_argument(
        "--semantic-jsonl",
        type=str,
        default=None,
        help="Path to semantic_validated.jsonl.",
    )
    parser.add_argument(
        "--emotion-vectors",
        type=str,
        default=None,
        help="Path to emotion_vectors.jsonl.",
    )
    parser.add_argument(
        "--emotion-evidence",
        type=str,
        default=None,
        help="Path to emotion_validated_evidence.jsonl.",
    )
    parser.add_argument(
        "--emotion-failures",
        type=str,
        default=None,
        help="Path to emotion_failures.jsonl.",
    )
    parser.add_argument(
        "--output-ttl",
        type=str,
        default=None,
        help="Output path for kg_l.ttl.",
    )
    parser.add_argument(
        "--output-validation",
        type=str,
        default=None,
        help="Output path for kg_l_validation.json.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Emotion edge threshold (score > threshold). Default: {DEFAULT_EMOTION_EDGE_THRESHOLD}.",
    )
    parser.add_argument(
        "--kg-m-path",
        type=str,
        default="data/processed/ml-1m/graphs/kg_m.ttl",
        help="Path to Phase 3B kg_m.ttl for cross-graph compatibility demo.",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip representative SPARQL queries and cross-graph compatibility demo.",
    )
    return parser.parse_args()


def print_summary_report(stats: dict) -> None:
    """Print clean ASCII summary report of the constructed KG-L graph."""
    ents = stats.get("entities", {})
    rels = stats.get("relations", {})
    emos = stats.get("emotion_metrics", {})
    per_emo = emos.get("per_emotion_counts", {})
    cov = stats.get("semantic_coverage", {})

    report = (
        "\n"
        "======================================================================\n"
        "         PLOTGRAPH KG-L KNOWLEDGE GRAPH CONSTRUCTION REPORT\n"
        "======================================================================\n"
        f"Total RDF Triples        : {stats.get('total_triples', 0):,}\n"
        f"Total Distinct Entities  : {ents.get('total_entities', 0):,}\n"
        "----------------------------------------------------------------------\n"
        "ENTITY COUNTS BY CLASS:\n"
        f"  pgo:Movie               : {ents.get('movies', 0):,}\n"
        f"  pgo:Theme               : {ents.get('themes', 0):,}\n"
        f"  pgo:Setting             : {ents.get('settings', 0):,}\n"
        f"  pgo:NarrativeDevice     : {ents.get('narrative_devices', 0):,}\n"
        f"  pgo:ProtagonistType     : {ents.get('protagonist_types', 0):,}\n"
        f"  pgo:SourceMaterial      : {ents.get('source_materials', 0):,}\n"
        f"  pgo:Emotion             : {ents.get('emotions', 0):,} (Plutchik 8)\n"
        f"  pgo:EmotionAssociation  : {ents.get('emotion_associations', 0):,}\n"
        "----------------------------------------------------------------------\n"
        "RELATION TRIPLE COUNTS:\n"
        f"  pgo:hasTheme            : {rels.get('hasTheme', 0):,}\n"
        f"  pgo:hasSetting          : {rels.get('hasSetting', 0):,}\n"
        f"  pgo:hasNarrativeDevice  : {rels.get('hasNarrativeDevice', 0):,}\n"
        f"  pgo:hasProtagonistType  : {rels.get('hasProtagonistType', 0):,}\n"
        f"  pgo:basedOn             : {rels.get('basedOn', 0):,}\n"
        f"  pgo:evokesEmotion       : {rels.get('evokesEmotion', 0):,} (score > {emos.get('threshold', 0.3)})\n"
        f"  pgo:hasEmotionAssoc     : {rels.get('hasEmotionAssociation', 0):,}\n"
        "----------------------------------------------------------------------\n"
        "PER-EMOTION EVOKES-EMOTION EDGES (score > 0.3):\n"
        f"  joy          : {per_emo.get('joy', 0):>5}  (expected: {EXPECTED_PER_EMOTION_COUNTS['joy']})\n"
        f"  trust        : {per_emo.get('trust', 0):>5}  (expected: {EXPECTED_PER_EMOTION_COUNTS['trust']})\n"
        f"  fear         : {per_emo.get('fear', 0):>5}  (expected: {EXPECTED_PER_EMOTION_COUNTS['fear']})\n"
        f"  surprise     : {per_emo.get('surprise', 0):>5}  (expected: {EXPECTED_PER_EMOTION_COUNTS['surprise']})\n"
        f"  sadness      : {per_emo.get('sadness', 0):>5}  (expected: {EXPECTED_PER_EMOTION_COUNTS['sadness']})\n"
        f"  disgust      : {per_emo.get('disgust', 0):>5}  (expected: {EXPECTED_PER_EMOTION_COUNTS['disgust']})\n"
        f"  anger        : {per_emo.get('anger', 0):>5}  (expected: {EXPECTED_PER_EMOTION_COUNTS['anger']})\n"
        f"  anticipation : {per_emo.get('anticipation', 0):>5}  (expected: {EXPECTED_PER_EMOTION_COUNTS['anticipation']})\n"
        f"  TOTAL        : {emos.get('total_evokes_emotion_edges', 0):>5}  (expected: {EXPECTED_EVOKES_EMOTION_TOTAL})\n"
        "----------------------------------------------------------------------\n"
        "MOVIE COVERAGE METRICS:\n"
        f"  Movies with >= 1 evokesEmotion edge : {emos.get('movies_with_at_least_one_emotion_edge', 0):,} ({emos.get('emotion_edge_movie_coverage_pct', 0.0)}%)\n"
        f"  Movies with no emotion edge         : {emos.get('movies_with_no_emotion_edge', 0):,}\n"
        f"  Movies missing emotion extraction   : {emos.get('movies_missing_validated_emotion_extraction', 0):,} (22 failed)\n"
        f"  Theme Coverage                      : {cov.get('movies_with_theme', 0):,} ({cov.get('theme_coverage_pct', 0.0)}%)\n"
        f"  Setting Coverage                    : {cov.get('movies_with_setting', 0):,} ({cov.get('setting_coverage_pct', 0.0)}%)\n"
        f"  Narrative Device Coverage           : {cov.get('movies_with_narrative_device', 0):,} ({cov.get('device_coverage_pct', 0.0)}%)\n"
        f"  Protagonist Type Coverage           : {cov.get('movies_with_protagonist_type', 0):,} ({cov.get('protagonist_coverage_pct', 0.0)}%)\n"
        f"  Source Material Coverage            : {cov.get('movies_with_source_material', 0):,} ({cov.get('source_material_coverage_pct', 0.0)}%)\n"
        "======================================================================\n"
    )
    print(report, flush=True)


def run_demonstrations(engine: KGLQueryEngine, kg_m_path: Path) -> None:
    """Run representative SPARQL queries and cross-graph demonstration."""
    print("----------------------------------------------------------------------")
    print("REPRESENTATIVE SPARQL QUERIES DEMONSTRATION")
    print("----------------------------------------------------------------------")

    # Sample movie: Toy Story (1995) (movie_id 1)
    sample_id = 1
    themes = engine.get_movie_themes(movie_id=sample_id)
    print(f"MovieLens ID {sample_id} - Themes:")
    for t in themes:
        print(f"  * {t['theme_label']} (<{t['theme']}>)")

    settings = engine.get_movie_settings(movie_id=sample_id)
    print(f"\nMovieLens ID {sample_id} - Settings:")
    for s in settings:
        print(f"  * {s['setting_label']} (<{s['setting']}>)")

    devices = engine.get_movie_narrative_devices(movie_id=sample_id)
    print(f"\nMovieLens ID {sample_id} - Narrative Devices:")
    for d in devices:
        print(f"  * {d['device_label']} (<{d['device']}>)")

    emotions = engine.get_movie_emotions(movie_id=sample_id)
    print(f"\nMovieLens ID {sample_id} - Evoked Emotions (score > 0.3):")
    for e in emotions:
        print(f"  * {e['emotion_label'].title():<12} : intensity = {e['intensity']:.2f}")

    evidence = engine.get_movie_emotion_evidence(movie_id=sample_id)
    print(f"\nMovieLens ID {sample_id} - Emotion Supporting Plot Evidence:")
    for ev in evidence:
        quotes = " | ".join(f'"{q}"' for q in ev['evidence'])
        print(f"  * {ev['emotion_label'].title():<12} ({ev['intensity']:.2f}) : {quotes}")

    # Shared semantic path between Toy Story (1) and Jumanji (2)
    shared_sem = engine.find_shared_semantic_paths(1, 2)
    print(f"\nShared Semantic Entities between Toy Story (1) and Jumanji (2):")
    if shared_sem:
        for p in shared_sem:
            rel = str(p['relation']).split('/')[-1]
            print(f"  * [{rel}] {p['label']}")
    else:
        print("  (None shared)")

    # Cross-graph demonstration if kg_m.ttl exists
    if kg_m_path.exists():
        demonstrate_cross_graph_compatibility(
            kg_m_source=kg_m_path,
            kg_l_source=engine.graph,
            sample_movie_id=sample_id,
        )


def main() -> int:
    args = parse_args()

    # Load configuration
    cfg: dict = {}
    cfg_p = Path(args.config)
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    paths_cfg = cfg.get("paths", {})
    ontology_cfg = cfg.get("ontology", {})

    metadata_path = Path(args.metadata_path or paths_cfg.get("input_metadata_parquet", "data/processed/ml-1m/metadata/movies_with_tmdb.parquet"))
    semantic_jsonl = Path(args.semantic_jsonl or paths_cfg.get("production_semantic_validated_jsonl", "data/processed/ml-1m/kg_l/full_extraction/semantic_validated.jsonl"))
    emotion_vectors = Path(args.emotion_vectors or paths_cfg.get("production_emotion_vectors_jsonl", "data/processed/ml-1m/kg_l/full_extraction/emotion_vectors.jsonl"))
    emotion_evidence = Path(args.emotion_evidence or paths_cfg.get("production_emotion_validated_evidence_jsonl", "data/processed/ml-1m/kg_l/full_extraction/emotion_validated_evidence.jsonl"))
    emotion_failures = Path(args.emotion_failures or paths_cfg.get("production_emotion_failures_jsonl", "data/processed/ml-1m/kg_l/full_extraction/emotion_failures.jsonl"))

    output_ttl = Path(args.output_ttl or paths_cfg.get("kg_l_ttl", "data/processed/ml-1m/graphs/kg_l.ttl"))
    output_validation = Path(args.output_validation or paths_cfg.get("kg_l_validation_json", "data/processed/ml-1m/graphs/kg_l_validation.json"))

    threshold = float(args.threshold or ontology_cfg.get("emotion_edge_threshold", DEFAULT_EMOTION_EDGE_THRESHOLD))
    kg_m_path = Path(args.kg_m_path)

    logger.info("========================================================================")
    logger.info("          STARTING PHASE 3G: KG-L KNOWLEDGE GRAPH CONSTRUCTION")
    logger.info("========================================================================")
    logger.info(f"Metadata Reference   : {metadata_path}")
    logger.info(f"Semantic Extractions : {semantic_jsonl}")
    logger.info(f"Emotion Vectors      : {emotion_vectors}")
    logger.info(f"Emotion Evidence     : {emotion_evidence}")
    logger.info(f"Emotion Edge Thresh  : score > {threshold}")
    logger.info(f"Target Output TTL    : {output_ttl}")
    logger.info(f"Target Validation    : {output_validation}")
    logger.info("========================================================================")

    try:
        graph, stats = build_and_save_kg_l(
            metadata_path=metadata_path,
            semantic_jsonl_path=semantic_jsonl,
            emotion_vectors_path=emotion_vectors,
            emotion_evidence_path=emotion_evidence,
            emotion_failures_path=emotion_failures,
            output_ttl_path=output_ttl,
            output_validation_path=output_validation,
            emotion_threshold=threshold,
        )

        # Print statistics report
        print_summary_report(stats)

        # Run demonstrations
        if not args.skip_demo:
            engine = KGLQueryEngine(graph)
            run_demonstrations(engine, kg_m_path)

        logger.info("Phase 3G Completed Successfully!")
        return 0

    except Exception as exc:
        logger.error(f"Fatal error during Phase 3G KG-L construction: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
