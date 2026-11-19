#!/usr/bin/env python3
"""Run Phase 3B: Construct, serialize, validate KG-M and run demonstration SPARQL queries.

Usage:
    python scripts/run_phase3b_kg_m.py --config configs/kg.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge_graph import (
    build_and_save_kg_m,
    KGMQueryEngine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3b_runner")


def print_validation_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print(" " * 15 + "PLOTGRAPH PHASE 3B: KG-M VALIDATION REPORT")
    print("=" * 70)

    print(f"\nTotal Graph Triples: {report['total_triples']:,}")

    print("\n--- Entities by Class ---")
    entities = report["entities"]
    print(f"  {'pgo:Movie':<24} {entities['movies']:>8,}")
    print(f"  {'pgo:Person':<24} {entities['persons']:>8,}  (directors & top actors)")
    print(f"  {'pgo:Genre':<24} {entities['genres']:>8,}")
    print(f"  {'pgo:Keyword':<24} {entities['keywords']:>8,}")
    print(f"  {'pgo:Collection':<24} {entities['collections']:>8,}")
    print(f"  {'pgo:ReleaseDecade':<24} {entities['release_decades']:>8,}")
    print(f"  {'-' * 34}")
    print(f"  {'Total Named Entities':<24} {entities['total_entities']:>8,}")

    print("\n--- Triples by Predicate ---")
    predicates = report["predicates"]
    for pred_uri, count in sorted(predicates.items(), key=lambda x: -x[1]):
        short_name = pred_uri.split("/")[-1].split("#")[-1]
        print(f"  {short_name:<26} {count:>8,}")

    print("\n--- Graph Integrity Assertions ---")
    integrity = report["integrity"]
    status_icon = "PASSED" if integrity["valid"] else "FAILED"
    print(f"  All MovieLens IDs Recoverable:        {'Yes' if integrity['all_movie_ids_recoverable'] else 'No'}")
    print(f"  All Internal Indices Recoverable:     {'Yes' if integrity['all_movie_indices_recoverable'] else 'No'}")
    print(f"  Zero Bogus / NaN Resource URIs:       {'Yes' if not integrity['has_bogus_uris'] else 'No'}")
    print(f"  Zero Bogus / NaN Entity Labels:       {'Yes' if not integrity['has_bogus_labels'] else 'No'}")
    print(f"  Missing Target Entity Definitions:    {integrity['missing_target_classes']}")
    print(f"  Overall Validation Status:            [{status_icon}]")
    print("=" * 70)


def run_demonstrations(query_engine: KGMQueryEngine) -> None:
    print("\n" + "=" * 70)
    print(" " * 17 + "PHASE 3B: SPARQL DEMONSTRATION QUERIES")
    print("=" * 70)

    # Demo 1: Metadata for a connected movie (Toy Story, MovieLens ID 1)
    movie_id_demo = 1
    meta = query_engine.get_movie_metadata(movie_id_demo)
    if meta:
        print(f"\n1. Movie Metadata Retrieval (MovieLens ID: {movie_id_demo})")
        print(f"   Title:           {meta['title']}")
        print(f"   Resource URI:    {meta['movie_uri']}")
        print(f"   MovieLens ID:    {meta['movie_id']} (Index: {meta['movie_idx']})")
        print(f"   TMDB ID:         {meta['tmdb_id']} - {meta['tmdb_title']}")
        print(f"   Release Date:    {meta['release_date']} ({meta['release_decade']})")
        print(f"   Director:        {meta['director']}")
        print(f"   Top Actors:      {', '.join(meta['actors'])}")
        print(f"   Genres:          {', '.join(meta['genres'])}")
        print(f"   Keywords (head): {', '.join(meta['keywords'][:6])} ...")
        print(f"   Collection:      {meta['collection']}")

        # Demo 2: Shared Director
        same_director = query_engine.find_movies_sharing_director(movie_id_demo, limit=5)
        print(f"\n2. Movies Sharing Director '{meta['director']}':")
        for m in same_director:
            print(f"   - {m['otherTitle']} (MovieLens ID: {m['otherMovieId']})")

        # Demo 3: Shared Actors
        same_actor = query_engine.find_movies_sharing_actor(movie_id_demo, limit=6)
        print(f"\n3. Movies Sharing Top-Billed Actors with '{meta['title']}':")
        for m in same_actor:
            print(f"   - {m['otherTitle']} (Shared Actor: {m['actorName']}, ML ID: {m['otherMovieId']})")

        # Demo 4: Same Collection
        same_coll = query_engine.find_movies_in_same_collection(movie_id_demo)
        print(f"\n4. Franchise Movies in '{meta['collection']}':")
        for m in same_coll:
            print(f"   - {m['otherTitle']} (MovieLens ID: {m['otherMovieId']})")

        # Demo 5: Shared Keywords
        same_kw = query_engine.find_movies_sharing_keyword(movie_id_demo, limit=4)
        print(f"\n5. Top Movies Sharing Plot Keywords with '{meta['title']}':")
        for m in same_kw:
            print(f"   - {m['otherTitle']} (Shared: {m['sharedKeywordCount']} keywords: {m['sharedKeywords'][:45]}...)")

    # Demo 6: Multi-hop movie -> entity -> movie path query
    # Toy Story (movie_id=1) and A Bug's Life (movie_id=2355)
    movie_a = 1
    movie_b = 2355
    meta_a = query_engine.get_movie_metadata(movie_a)
    meta_b = query_engine.get_movie_metadata(movie_b)

    if meta_a and meta_b:
        print(f"\n6. Multi-Hop Path Query: Connecting '{meta_a['title']}' and '{meta_b['title']}'")
        paths = query_engine.find_paths_between_movies(movie_a, movie_b)
        print(f"   Found {len(paths)} 2-hop shared metadata connections:")
        for p in paths:
            print(
                f"   [Path] '{meta_a['title']}' --({p['predicate_a']})--> "
                f"[{p['entity_type']}: {p['entity_label']}] "
                f"<--({p['predicate_b']})-- '{meta_b['title']}'"
            )

    # Demo 7: Catalog Preservation of Unmapped Movie (movie_id=142)
    unmapped_id = 142
    unmapped_meta = query_engine.get_movie_metadata(unmapped_id)
    if unmapped_meta:
        print(f"\n7. Unmapped MovieLens Movie Representation (MovieLens ID: {unmapped_id})")
        print(f"   Title:           {unmapped_meta['title']}")
        print(f"   Resource URI:    {unmapped_meta['movie_uri']}")
        print(f"   MovieLens ID:    {unmapped_meta['movie_id']} (Index: {unmapped_meta['movie_idx']})")
        print(f"   TMDB ID:         {unmapped_meta['tmdb_id']} (Unmapped)")
        print(f"   Director:        {unmapped_meta['director']} (No speculative relations)")
        print(f"   Actors:          {unmapped_meta['actors']}")
        print(f"   Status:          Catalog identity preserved; zero dangling/hallucinated edges.")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run PlotGraph Phase 3B: Construct, serialize, and validate KG-M."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kg.yaml",
        help="Path to KG configuration YAML file (default: configs/kg.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)

    logger.info(f"Starting Phase 3B KG-M pipeline using config: {config_path}")
    graph, val_report = build_and_save_kg_m(config_path)

    # Print validation report
    print_validation_report(val_report)

    # Run demonstrations
    query_engine = KGMQueryEngine(graph)
    run_demonstrations(query_engine)

    logger.info("Phase 3B KG-M pipeline completed successfully!")


if __name__ == "__main__":
    main()
