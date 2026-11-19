"""KG-M Construction, Validation, and Turtle Serialization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD
import yaml

from .namespaces import (
    CLASS_COLLECTION,
    CLASS_GENRE,
    CLASS_KEYWORD,
    CLASS_MOVIE,
    CLASS_PERSON,
    CLASS_RELEASE_DECADE,
    PRED_DIRECTED_BY,
    PRED_HAS_GENRE,
    PRED_HAS_KEYWORD,
    PRED_IN_COLLECTION,
    PRED_INTERNAL_INDEX,
    PRED_MOVIELENS_ID,
    PRED_RELEASE_DATE,
    PRED_RELEASE_DECADE,
    PRED_STARS_ACTOR,
    PRED_TITLE,
    PRED_TMDB_ID,
    PRED_TMDB_TITLE,
    bind_namespaces,
    build_entity_uri,
    build_movie_uri,
)

logger = logging.getLogger(__name__)


def build_kg_m(
    df: pd.DataFrame,
    top_k_actors: int = 5,
) -> Graph:
    """Build the metadata Knowledge Graph (KG-M) from the Phase 3A unified metadata table.

    Design rules:
    - 100% deterministic URI construction (no random UUIDs).
    - Preserves all 3,883 MovieLens catalog movies as pgo:Movie entities.
    - Resolves rare TMDB ID collisions in source data (e.g. duplicate Kaggle links) cleanly
      by disambiguating the Movie URI, preventing silent entity merging.
    - Represents directors and actors as pgo:Person entities.
    - Adds verified metadata relations only; missing fields are omitted rather than hallucinated.

    Args:
        df: Unified metadata DataFrame (from movies_with_tmdb.parquet).
        top_k_actors: Maximum number of top-billed cast members to connect per movie.

    Returns:
        RDFlib Graph containing KG-M.
    """
    graph = Graph()
    bind_namespaces(graph)

    # Detect duplicate TMDB IDs across distinct MovieLens movies in the dataset
    tmdb_series = df["tmdb_id"].dropna()
    dup_tmdb_ids: Set[int] = set(
        df.loc[df["tmdb_id"].isin(tmdb_series[tmdb_series.duplicated()]), "tmdb_id"]
        .astype(int)
        .unique()
    )
    if dup_tmdb_ids:
        logger.info(
            f"Detected duplicate TMDB IDs in input catalog: {dup_tmdb_ids}. "
            "Disambiguating Movie URIs with MovieLens ID to avoid collision."
        )

    # Registry for target metadata entities: uri -> (class_uri, clean_label)
    # Ensures each metadata entity has exactly ONE rdf:type and ONE rdfs:label
    metadata_entities: Dict[URIRef, Tuple[URIRef, str]] = {}

    for _, row in df.iterrows():
        movie_id = int(row["movie_id"])
        movie_idx = int(row["movie_idx"])
        tmdb_id = int(row["tmdb_id"]) if pd.notna(row["tmdb_id"]) else None

        # 1. Movie Resource
        is_dup = tmdb_id in dup_tmdb_ids if tmdb_id is not None else False
        movie_uri = build_movie_uri(movie_id=movie_id, tmdb_id=tmdb_id, is_tmdb_duplicate=is_dup)

        ml_title = str(row["ml_title"]).strip()
        graph.add((movie_uri, RDF.type, CLASS_MOVIE))
        graph.add((movie_uri, RDFS.label, Literal(ml_title)))
        graph.add((movie_uri, PRED_TITLE, Literal(ml_title)))
        graph.add((movie_uri, PRED_MOVIELENS_ID, Literal(movie_id, datatype=XSD.integer)))
        graph.add((movie_uri, PRED_INTERNAL_INDEX, Literal(movie_idx, datatype=XSD.integer)))

        if tmdb_id is not None:
            graph.add((movie_uri, PRED_TMDB_ID, Literal(tmdb_id, datatype=XSD.integer)))

        if pd.notna(row.get("tmdb_title")) and str(row["tmdb_title"]).strip():
            graph.add((movie_uri, PRED_TMDB_TITLE, Literal(str(row["tmdb_title"]).strip())))

        if pd.notna(row.get("release_date")) and str(row["release_date"]).strip():
            graph.add((movie_uri, PRED_RELEASE_DATE, Literal(str(row["release_date"]).strip())))

        # 2. Release Decade Relation
        if pd.notna(row.get("release_decade")):
            dec_str = str(row["release_decade"]).strip()
            if dec_str and dec_str.lower() != "none":
                dec_uri = build_entity_uri("decade", dec_str)
                graph.add((movie_uri, PRED_RELEASE_DECADE, dec_uri))
                if dec_uri not in metadata_entities:
                    metadata_entities[dec_uri] = (CLASS_RELEASE_DECADE, dec_str)

        # 3. Director Relation
        if pd.notna(row.get("director")):
            dir_str = str(row["director"]).strip()
            if dir_str and dir_str.lower() != "none":
                dir_uri = build_entity_uri("person", dir_str)
                graph.add((movie_uri, PRED_DIRECTED_BY, dir_uri))
                if dir_uri not in metadata_entities:
                    metadata_entities[dir_uri] = (CLASS_PERSON, dir_str)

        # 4. Top Actors Relations
        actors = row.get("top_actors")
        if isinstance(actors, (list, tuple, np.ndarray)):
            for act in actors[:top_k_actors]:
                act_str = str(act).strip()
                if act_str and act_str.lower() != "none":
                    act_uri = build_entity_uri("person", act_str)
                    graph.add((movie_uri, PRED_STARS_ACTOR, act_uri))
                    if act_uri not in metadata_entities:
                        metadata_entities[act_uri] = (CLASS_PERSON, act_str)

        # 5. Genres Relations
        genres = row.get("tmdb_genres")
        if isinstance(genres, (list, tuple, np.ndarray)):
            for gen in genres:
                gen_str = str(gen).strip()
                if gen_str and gen_str.lower() != "none":
                    gen_uri = build_entity_uri("genre", gen_str)
                    graph.add((movie_uri, PRED_HAS_GENRE, gen_uri))
                    if gen_uri not in metadata_entities:
                        metadata_entities[gen_uri] = (CLASS_GENRE, gen_str)

        # 6. Keywords Relations
        keywords = row.get("keywords")
        if isinstance(keywords, (list, tuple, np.ndarray)):
            for kw in keywords:
                kw_str = str(kw).strip()
                if kw_str and kw_str.lower() != "none":
                    kw_uri = build_entity_uri("keyword", kw_str)
                    graph.add((movie_uri, PRED_HAS_KEYWORD, kw_uri))
                    if kw_uri not in metadata_entities:
                        metadata_entities[kw_uri] = (CLASS_KEYWORD, kw_str)

        # 7. Collection Relation
        if pd.notna(row.get("collection_name")):
            col_str = str(row["collection_name"]).strip()
            if col_str and col_str.lower() != "none":
                col_uri = build_entity_uri("collection", col_str)
                graph.add((movie_uri, PRED_IN_COLLECTION, col_uri))
                if col_uri not in metadata_entities:
                    metadata_entities[col_uri] = (CLASS_COLLECTION, col_str)

    # Add entity definitions deterministically sorted by URI
    for entity_uri in sorted(metadata_entities.keys(), key=lambda u: str(u)):
        class_uri, label = metadata_entities[entity_uri]
        graph.add((entity_uri, RDF.type, class_uri))
        graph.add((entity_uri, RDFS.label, Literal(label)))

    logger.info(
        f"KG-M construction complete: {len(graph):,} triples, "
        f"{len(df)} movies, {len(metadata_entities)} unique metadata entities."
    )
    return graph


def serialize_kg_m(graph: Graph, output_path: Union[str, Path]) -> Path:
    """Serialize the RDF graph to Turtle format with human-readable prefixes.

    Args:
        graph: RDFlib Graph.
        output_path: Target path for the .ttl file.

    Returns:
        Path of the saved file.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Serializing KG-M to {target} (Turtle format)...")
    graph.serialize(destination=target, format="turtle")
    logger.info(f"Saved {target.stat().st_size / (1024 * 1024):.2f} MB Turtle graph.")
    return target


def validate_kg_m(
    graph: Graph,
    input_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Validate graph integrity, entity counts, predicate distributions, and ID recoverability.

    Args:
        graph: Loaded or reloaded RDFlib Graph.
        input_df: Optional input DataFrame to verify full catalog recoverability.

    Returns:
        Validation report dictionary.
    """
    # 1. Entity counts by class
    def _count_class(cls_uri: URIRef) -> int:
        return len(list(graph.subjects(RDF.type, cls_uri)))

    movie_count = _count_class(CLASS_MOVIE)
    person_count = _count_class(CLASS_PERSON)
    genre_count = _count_class(CLASS_GENRE)
    keyword_count = _count_class(CLASS_KEYWORD)
    collection_count = _count_class(CLASS_COLLECTION)
    decade_count = _count_class(CLASS_RELEASE_DECADE)

    # 2. Relation counts by predicate
    pred_counts: Dict[str, int] = {}
    for _, p, _ in graph:
        p_str = str(p)
        pred_counts[p_str] = pred_counts.get(p_str, 0) + 1

    # 3. Check for any placeholder / NaN entities
    bogus_uris = []
    for s in set(graph.subjects()):
        s_str = str(s)
        local_name = s_str.split("/")[-1]
        parts = local_name.split("_", 1)
        if len(parts) == 2 and parts[1].lower() in ("nan", "none", ""):
            bogus_uris.append(s_str)
        elif local_name.endswith("_") or local_name.startswith("_"):
            bogus_uris.append(s_str)

    bogus_labels = []
    for _, _, o in graph.triples((None, RDFS.label, None)):
        o_str = str(o).strip().lower()
        if o_str in ("nan", "none", ""):
            bogus_labels.append(str(o))

    # 4. Check target entity validity for all relation edges
    relation_preds = [
        PRED_HAS_GENRE,
        PRED_DIRECTED_BY,
        PRED_STARS_ACTOR,
        PRED_HAS_KEYWORD,
        PRED_IN_COLLECTION,
        PRED_RELEASE_DECADE,
    ]
    missing_target_classes = 0
    missing_target_labels = 0
    for rel in relation_preds:
        for _, _, target in graph.triples((None, rel, None)):
            if not isinstance(target, URIRef):
                missing_target_classes += 1
                continue
            if not list(graph.objects(target, RDF.type)):
                missing_target_classes += 1
            if not list(graph.objects(target, RDFS.label)):
                missing_target_labels += 1

    # 5. MovieLens ID recoverability
    ml_id_recoverable = True
    ml_idx_recoverable = True
    if input_df is not None:
        expected_movie_ids = set(input_df["movie_id"].astype(int))
        expected_movie_idxs = set(input_df["movie_idx"].astype(int))

        recovered_ids = {
            int(o) for _, _, o in graph.triples((None, PRED_MOVIELENS_ID, None))
        }
        recovered_idxs = {
            int(o) for _, _, o in graph.triples((None, PRED_INTERNAL_INDEX, None))
        }

        ml_id_recoverable = (expected_movie_ids == recovered_ids)
        ml_idx_recoverable = (expected_movie_idxs == recovered_idxs)

    report = {
        "total_triples": len(graph),
        "entities": {
            "movies": movie_count,
            "persons": person_count,
            "genres": genre_count,
            "keywords": keyword_count,
            "collections": collection_count,
            "release_decades": decade_count,
            "total_entities": (
                movie_count
                + person_count
                + genre_count
                + keyword_count
                + collection_count
                + decade_count
            ),
        },
        "predicates": pred_counts,
        "integrity": {
            "has_bogus_uris": len(bogus_uris) > 0,
            "bogus_uri_count": len(bogus_uris),
            "has_bogus_labels": len(bogus_labels) > 0,
            "bogus_label_count": len(bogus_labels),
            "missing_target_classes": missing_target_classes,
            "missing_target_labels": missing_target_labels,
            "all_movie_ids_recoverable": ml_id_recoverable,
            "all_movie_indices_recoverable": ml_idx_recoverable,
            "valid": (
                len(bogus_uris) == 0
                and len(bogus_labels) == 0
                and missing_target_classes == 0
                and missing_target_labels == 0
                and ml_id_recoverable
                and ml_idx_recoverable
            ),
        },
    }
    return report


def build_and_save_kg_m(
    config_path_or_dict: Union[str, Path, Dict[str, Any]],
) -> Tuple[Graph, Dict[str, Any]]:
    """End-to-end pipeline: load Phase 3A parquet, build KG-M, serialize, reload, and validate.

    Args:
        config_path_or_dict: Path to kg.yaml or parsed config dict.

    Returns:
        Tuple of (reloaded_graph, validation_report).
    """
    if isinstance(config_path_or_dict, (str, Path)):
        with open(config_path_or_dict, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = config_path_or_dict

    paths = config["paths"]
    settings = config.get("settings", {})
    top_k_actors = settings.get("top_k_actors", 5)

    input_parquet = Path(paths["input_metadata_parquet"])
    if not input_parquet.exists():
        raise FileNotFoundError(
            f"Phase 3A input metadata artifact not found at {input_parquet}. "
            "Please run scripts/run_phase3a_metadata.py first."
        )

    logger.info(f"Loading Phase 3A metadata from {input_parquet}...")
    df = pd.read_parquet(input_parquet)

    # 1. Build KG-M
    logger.info("Building KG-M in-memory graph...")
    graph = build_kg_m(df, top_k_actors=top_k_actors)
    original_triple_count = len(graph)

    # 2. Serialize to Turtle
    output_ttl = Path(paths["output_graph_ttl"])
    serialize_kg_m(graph, output_ttl)

    # 3. Reload graph from Turtle to verify syntax and round-trip preservation
    logger.info(f"Reloading {output_ttl} to verify RDF/Turtle syntax and round-trip preservation...")
    reloaded_graph = Graph()
    bind_namespaces(reloaded_graph)
    reloaded_graph.parse(output_ttl, format="turtle")
    reloaded_triple_count = len(reloaded_graph)

    if reloaded_triple_count != original_triple_count:
        raise ValueError(
            f"Triple count mismatch after serialization round-trip: "
            f"original {original_triple_count} vs reloaded {reloaded_triple_count}"
        )
    logger.info(f"Round-trip validation passed: {reloaded_triple_count:,} triples confirmed.")

    # 4. Comprehensive graph validation
    logger.info("Validating entity integrity and ID recoverability...")
    val_report = validate_kg_m(reloaded_graph, input_df=df)

    if not val_report["integrity"]["valid"]:
        logger.error(f"Graph validation failed: {val_report['integrity']}")
        raise ValueError(f"KG-M validation failed integrity checks: {val_report['integrity']}")

    # 5. Save validation report JSON
    report_json_path = Path(paths["validation_report_json"])
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)
    logger.info(f"Saved validation report to {report_json_path}")

    return reloaded_graph, val_report
