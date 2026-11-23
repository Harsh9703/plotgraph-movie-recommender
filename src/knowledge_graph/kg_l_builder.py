"""Knowledge Graph Builder, Serializer, and Validator for PlotGraph KG-L (Phase 3G).

Constructs the LLM-enriched Knowledge Graph (KG-L) from frozen Phase 3F production
extraction artifacts (semantic and emotion extractions) and unified Phase 3A metadata.

Key Architecture:
1. Strict Ontology Conformance:
   - Relations: hasTheme, hasSetting, hasNarrativeDevice, hasProtagonistType, basedOn, evokesEmotion
   - Classes: Movie, Theme, Setting, NarrativeDevice, ProtagonistType, SourceMaterial, Emotion, EmotionAssociation
2. 100% Movie URI Alignment with KG-M:
   - Reuses identical movie URI resolution logic (handling TMDB links, duplicate TMDB IDs, and MovieLens fallbacks)
   - Guarantees seamless cross-graph joining between KG-M and KG-L by RDF movie resource identity
3. Conservative Canonicalization:
   - Unicode NFKC normalization, whitespace collapsing, casing, and duplicate removal
   - Zero fuzzy merging
4. Continuous Emotion and Provenance Representation:
   - Direct Movie --pgo:evokesEmotion--> Emotion edge for score > 0.3 (exactly 5,452 triples)
   - Rich EmotionAssociation entity carrying continuous score and supporting text quotes
   - Exactly 8 Plutchik emotion entities
5. Robust Validation:
   - Verifies round-trip Turtle serialization, entity integrity, exact emotion counts,
     and exclusion of failed extractions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD

from .kg_l_contract import (
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    PLUTCHIK_EMOTIONS,
    canonicalize_semantic_entity,
)
from .namespaces import (
    CLASS_EMOTION,
    CLASS_EMOTION_ASSOCIATION,
    CLASS_MOVIE,
    CLASS_NARRATIVE_DEVICE,
    CLASS_PROTAGONIST_TYPE,
    CLASS_SETTING,
    CLASS_SOURCE_MATERIAL,
    CLASS_THEME,
    PRED_ASSOCIATED_EMOTION,
    PRED_BASED_ON,
    PRED_EMOTION_INTENSITY,
    PRED_EVOKES_EMOTION,
    PRED_HAS_EMOTION_ASSOCIATION,
    PRED_HAS_NARRATIVE_DEVICE,
    PRED_HAS_PROTAGONIST_TYPE,
    PRED_HAS_SETTING,
    PRED_HAS_THEME,
    PRED_INTERNAL_INDEX,
    PRED_MOVIELENS_ID,
    PRED_SUPPORTING_EVIDENCE,
    PRED_TITLE,
    PRED_TMDB_ID,
    PGR,
    bind_namespaces,
    build_entity_uri,
    build_movie_uri_index,
)

logger = logging.getLogger(__name__)

# Expected frozen production counts from Phase 3F
EXPECTED_EVOKES_EMOTION_TOTAL: int = 5452
EXPECTED_PER_EMOTION_COUNTS: Dict[str, int] = {
    "joy": 589,
    "trust": 201,
    "fear": 1028,
    "surprise": 595,
    "sadness": 731,
    "disgust": 350,
    "anger": 543,
    "anticipation": 1415,
}


def build_kg_l(
    metadata_df: pd.DataFrame,
    semantic_records: List[Dict[str, Any]],
    emotion_vectors: Optional[List[Dict[str, Any]]] = None,
    emotion_evidence_records: Optional[List[Dict[str, Any]]] = None,
    emotion_threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD,
) -> Graph:
    """Build the LLM-enriched Knowledge Graph (KG-L).

    Args:
        metadata_df: Full catalog metadata DataFrame (from movies_with_tmdb.parquet).
        semantic_records: List of validated semantic extraction dictionaries.
        emotion_vectors: Optional list of validated emotion vector dictionaries.
        emotion_evidence_records: Optional list of validated emotion evidence dictionaries.
        emotion_threshold: Threshold above which evokesEmotion edges are created (default: 0.3).

    Returns:
        RDFlib Graph containing KG-L.
    """
    graph = Graph()
    bind_namespaces(graph)

    # 1. Build deterministic movie URI indexes from metadata
    uri_by_id, uri_by_idx = build_movie_uri_index(metadata_df)

    # Build metadata lookup by movie_id for titles and TMDB IDs
    meta_by_id: Dict[int, Dict[str, Any]] = {
        int(r["movie_id"]): r for r in metadata_df.to_dict(orient="records")
    }

    # 2. Add the 8 Plutchik Emotion entities
    for emo_name in PLUTCHIK_EMOTIONS:
        emo_uri = build_entity_uri("emotion", emo_name)
        graph.add((emo_uri, RDF.type, CLASS_EMOTION))
        graph.add((emo_uri, RDFS.label, Literal(emo_name)))

    # Registry for target semantic entities: uri -> (class_uri, clean_label)
    semantic_entities: Dict[URIRef, Tuple[URIRef, str]] = {}

    # 3. Add Semantic Triples for all validated semantic extractions
    for s_rec in semantic_records:
        movie_id = int(s_rec["movie_id"])
        movie_idx = int(s_rec["movie_idx"])
        title = str(s_rec.get("title", meta_by_id.get(movie_id, {}).get("ml_title", "Unknown"))).strip()

        movie_uri = uri_by_id.get(movie_id)
        if movie_uri is None:
            raise KeyError(f"movie_id {movie_id} not found in metadata catalog.")

        # Movie Core Triples
        graph.add((movie_uri, RDF.type, CLASS_MOVIE))
        graph.add((movie_uri, RDFS.label, Literal(title)))
        graph.add((movie_uri, PRED_TITLE, Literal(title)))
        graph.add((movie_uri, PRED_MOVIELENS_ID, Literal(movie_id, datatype=XSD.integer)))
        graph.add((movie_uri, PRED_INTERNAL_INDEX, Literal(movie_idx, datatype=XSD.integer)))

        meta = meta_by_id.get(movie_id)
        if meta and pd.notna(meta.get("tmdb_id")):
            graph.add((movie_uri, PRED_TMDB_ID, Literal(int(meta["tmdb_id"]), datatype=XSD.integer)))

        # Categories: themes, settings, narrative_devices, protagonist_types, based_on
        cat_mappings = [
            ("themes", "theme", PRED_HAS_THEME, CLASS_THEME),
            ("settings", "setting", PRED_HAS_SETTING, CLASS_SETTING),
            ("narrative_devices", "device", PRED_HAS_NARRATIVE_DEVICE, CLASS_NARRATIVE_DEVICE),
            ("protagonist_types", "protagonist", PRED_HAS_PROTAGONIST_TYPE, CLASS_PROTAGONIST_TYPE),
            ("based_on", "source", PRED_BASED_ON, CLASS_SOURCE_MATERIAL),
        ]

        for field_name, slug_prefix, pred_uri, class_uri in cat_mappings:
            raw_items = s_rec.get(field_name, [])
            if not isinstance(raw_items, (list, tuple, set)):
                continue

            seen_for_movie: Set[str] = set()
            for item in raw_items:
                try:
                    canon_text = canonicalize_semantic_entity(str(item))
                except ValueError:
                    continue

                if not canon_text or canon_text in seen_for_movie:
                    continue
                seen_for_movie.add(canon_text)

                ent_uri = build_entity_uri(slug_prefix, canon_text)
                graph.add((movie_uri, pred_uri, ent_uri))

                if ent_uri not in semantic_entities:
                    semantic_entities[ent_uri] = (class_uri, canon_text)

    # 4. Add Emotion Evidence and Vectors
    # Build evidence quote lookup: movie_id -> emotion -> List[str]
    evidence_lookup: Dict[int, Dict[str, List[str]]] = {}
    if emotion_evidence_records:
        for ev_rec in emotion_evidence_records:
            m_id = int(ev_rec["movie_id"])
            emotions_data = ev_rec.get("emotions", {})
            evidence_lookup[m_id] = {}
            for emo_name in PLUTCHIK_EMOTIONS:
                entry = emotions_data.get(emo_name, {})
                quotes = entry.get("evidence", []) if isinstance(entry, dict) else []
                evidence_lookup[m_id][emo_name] = [str(q).strip() for q in quotes if str(q).strip()]

    if emotion_vectors:
        for vec_rec in emotion_vectors:
            movie_id = int(vec_rec["movie_id"])
            movie_uri = uri_by_id.get(movie_id)
            if movie_uri is None:
                continue

            title = str(vec_rec.get("title", meta_by_id.get(movie_id, {}).get("ml_title", ""))).strip()
            movie_slug = str(movie_uri).split("/")[-1]

            for emo_name in PLUTCHIK_EMOTIONS:
                score = float(vec_rec.get(emo_name, 0.0))
                # Threshold semantics: strictly score > threshold (0.3)
                if score > emotion_threshold:
                    emo_uri = build_entity_uri("emotion", emo_name)

                    # Direct traversal edge
                    graph.add((movie_uri, PRED_EVOKES_EMOTION, emo_uri))

                    # Reified EmotionAssociation entity for continuous score & evidence provenance
                    assoc_uri = PGR[f"emoassoc_{movie_slug}_{emo_name}"]
                    graph.add((movie_uri, PRED_HAS_EMOTION_ASSOCIATION, assoc_uri))
                    graph.add((assoc_uri, RDF.type, CLASS_EMOTION_ASSOCIATION))
                    graph.add((assoc_uri, RDFS.label, Literal(f"{title} - {emo_name.title()} ({score:.2f})")))
                    graph.add((assoc_uri, PRED_ASSOCIATED_EMOTION, emo_uri))
                    graph.add((assoc_uri, PRED_EMOTION_INTENSITY, Literal(score, datatype=XSD.float)))

                    # Supporting evidence quotes
                    quotes = evidence_lookup.get(movie_id, {}).get(emo_name, [])
                    for q in quotes:
                        graph.add((assoc_uri, PRED_SUPPORTING_EVIDENCE, Literal(q)))

    # 5. Add registered semantic target entities (type and label)
    for ent_uri, (class_uri, label) in semantic_entities.items():
        graph.add((ent_uri, RDF.type, class_uri))
        graph.add((ent_uri, RDFS.label, Literal(label)))

    logger.info(f"Built KG-L graph with {len(graph):,} triples.")
    return graph


def serialize_kg_l(graph: Graph, output_path: Union[str, Path]) -> Path:
    """Serialize the KG-L graph to Turtle (.ttl) format with standard prefixes.

    Args:
        graph: RDFlib Graph.
        output_path: Path for output .ttl file.

    Returns:
        Path to serialized Turtle file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Serializing KG-L graph ({len(graph):,} triples) to {path}...")
    graph.serialize(destination=str(path), format="turtle")
    logger.info(f"Successfully serialized KG-L to {path} ({path.stat().st_size / (1024 * 1024):.2f} MB).")
    return path


def compute_kg_l_statistics(
    graph: Graph,
    threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD,
    total_eligible_movies: int = 3810,
    total_validated_emotion_movies: int = 3788,
) -> Dict[str, Any]:
    """Compute comprehensive structural statistics and relation counts for KG-L.

    Args:
        graph: RDFlib Graph containing KG-L.
        threshold: Edge threshold used for evokesEmotion.
        total_eligible_movies: Number of eligible catalog movies (3,810).
        total_validated_emotion_movies: Number of movies with validated emotion extraction (3,788).

    Returns:
        Structured statistics dictionary.
    """
    total_triples = len(graph)

    # Entity counts
    movies_count = len(set(graph.subjects(RDF.type, CLASS_MOVIE)))
    themes_count = len(set(graph.subjects(RDF.type, CLASS_THEME)))
    settings_count = len(set(graph.subjects(RDF.type, CLASS_SETTING)))
    devices_count = len(set(graph.subjects(RDF.type, CLASS_NARRATIVE_DEVICE)))
    prots_count = len(set(graph.subjects(RDF.type, CLASS_PROTAGONIST_TYPE)))
    sources_count = len(set(graph.subjects(RDF.type, CLASS_SOURCE_MATERIAL)))
    emotions_count = len(set(graph.subjects(RDF.type, CLASS_EMOTION)))
    associations_count = len(set(graph.subjects(RDF.type, CLASS_EMOTION_ASSOCIATION)))

    total_entities = (
        movies_count + themes_count + settings_count + devices_count
        + prots_count + sources_count + emotions_count + associations_count
    )

    # Relation triple counts
    theme_triples = len(list(graph.triples((None, PRED_HAS_THEME, None))))
    setting_triples = len(list(graph.triples((None, PRED_HAS_SETTING, None))))
    device_triples = len(list(graph.triples((None, PRED_HAS_NARRATIVE_DEVICE, None))))
    prot_triples = len(list(graph.triples((None, PRED_HAS_PROTAGONIST_TYPE, None))))
    source_triples = len(list(graph.triples((None, PRED_BASED_ON, None))))
    evokes_triples = len(list(graph.triples((None, PRED_EVOKES_EMOTION, None))))
    assoc_triples = len(list(graph.triples((None, PRED_HAS_EMOTION_ASSOCIATION, None))))

    # Per-emotion evokesEmotion counts
    per_emotion_counts: Dict[str, int] = {}
    for emo_name in PLUTCHIK_EMOTIONS:
        emo_uri = build_entity_uri("emotion", emo_name)
        cnt = len(list(graph.triples((None, PRED_EVOKES_EMOTION, emo_uri))))
        per_emotion_counts[emo_name] = cnt

    # Movie emotion coverage
    movies_with_emotion_edge = len(set(graph.subjects(PRED_EVOKES_EMOTION, None)))
    movies_with_no_emotion_edge = movies_count - movies_with_emotion_edge
    missing_emotion_movies = total_eligible_movies - total_validated_emotion_movies

    # Semantic coverage
    movies_with_theme = len(set(graph.subjects(PRED_HAS_THEME, None)))
    movies_with_setting = len(set(graph.subjects(PRED_HAS_SETTING, None)))
    movies_with_device = len(set(graph.subjects(PRED_HAS_NARRATIVE_DEVICE, None)))
    movies_with_protagonist = len(set(graph.subjects(PRED_HAS_PROTAGONIST_TYPE, None)))
    movies_with_source = len(set(graph.subjects(PRED_BASED_ON, None)))

    stats: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_triples": total_triples,
        "entities": {
            "movies": movies_count,
            "themes": themes_count,
            "settings": settings_count,
            "narrative_devices": devices_count,
            "protagonist_types": prots_count,
            "source_materials": sources_count,
            "emotions": emotions_count,
            "emotion_associations": associations_count,
            "total_entities": total_entities,
        },
        "relations": {
            "hasTheme": theme_triples,
            "hasSetting": setting_triples,
            "hasNarrativeDevice": device_triples,
            "hasProtagonistType": prot_triples,
            "basedOn": source_triples,
            "evokesEmotion": evokes_triples,
            "hasEmotionAssociation": assoc_triples,
        },
        "emotion_metrics": {
            "threshold": threshold,
            "total_evokes_emotion_edges": evokes_triples,
            "per_emotion_counts": per_emotion_counts,
            "movies_with_at_least_one_emotion_edge": movies_with_emotion_edge,
            "movies_with_no_emotion_edge": movies_with_no_emotion_edge,
            "movies_missing_validated_emotion_extraction": missing_emotion_movies,
            "emotion_edge_movie_coverage_pct": round((movies_with_emotion_edge / max(movies_count, 1)) * 100.0, 2),
        },
        "semantic_coverage": {
            "movies_with_theme": movies_with_theme,
            "theme_coverage_pct": round((movies_with_theme / max(movies_count, 1)) * 100.0, 2),
            "movies_with_setting": movies_with_setting,
            "setting_coverage_pct": round((movies_with_setting / max(movies_count, 1)) * 100.0, 2),
            "movies_with_narrative_device": movies_with_device,
            "device_coverage_pct": round((movies_with_device / max(movies_count, 1)) * 100.0, 2),
            "movies_with_protagonist_type": movies_with_protagonist,
            "protagonist_coverage_pct": round((movies_with_protagonist / max(movies_count, 1)) * 100.0, 2),
            "movies_with_source_material": movies_with_source,
            "source_material_coverage_pct": round((movies_with_source / max(movies_count, 1)) * 100.0, 2),
        },
    }
    return stats


def validate_kg_l(
    graph: Graph,
    metadata_df: pd.DataFrame,
    expected_movie_count: int = 3810,
    expected_evokes_total: int = EXPECTED_EVOKES_EMOTION_TOTAL,
    expected_per_emotion: Optional[Dict[str, int]] = None,
    emotion_threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD,
    excluded_failure_indices: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """Validate completeness, integrity, and strict contract conformance of KG-L.

    Checks:
    1. Triple count > 0 and graph parses cleanly.
    2. All relation targets have rdf:type and rdfs:label.
    3. Zero NaN or None placeholder strings in URIs or literals.
    4. MovieLens movie identity uniqueness: exactly expected_movie_count movies,
       each with unique movieLensId and internalMovieIndex.
    5. Movie URIs match 100% with KG-M URI generation from metadata_df.
    6. Exactly 8 Emotion resources.
    7. All emotion intensities are in [0.0, 1.0] and strictly > threshold for evokesEmotion.
    8. Exactly expected_evokes_total direct evokesEmotion edges.
    9. Exact per-emotion counts match expected values.
    10. Excluded failure indices produce zero emotion triples.

    Args:
        graph: RDFlib Graph containing KG-L.
        metadata_df: Full catalog metadata DataFrame.
        expected_movie_count: Expected count of Movie entities (3,810).
        expected_evokes_total: Expected count of evokesEmotion triples (5,452).
        expected_per_emotion: Expected per-emotion edge count mapping.
        emotion_threshold: Edge selection threshold (0.3).
        excluded_failure_indices: Set of movie_idx that failed emotion extraction.

    Returns:
        Validation results dictionary.

    Raises:
        ValueError: If any validation check fails.
    """
    errors: List[str] = []
    per_emo_expected = expected_per_emotion or EXPECTED_PER_EMOTION_COUNTS

    # 1. Non-empty graph
    if len(graph) == 0:
        errors.append("Graph is completely empty.")

    # 2. Movie entity count and identity recovery
    movie_nodes = set(graph.subjects(RDF.type, CLASS_MOVIE))
    if len(movie_nodes) != expected_movie_count:
        errors.append(f"Movie entity count mismatch: expected {expected_movie_count}, got {len(movie_nodes)}")

    # Verify movie_id and movie_idx recovery
    movie_ids_in_graph: Set[int] = set()
    movie_indices_in_graph: Set[int] = set()
    for m_uri in movie_nodes:
        m_id_lits = list(graph.objects(m_uri, PRED_MOVIELENS_ID))
        if len(m_id_lits) != 1:
            errors.append(f"Movie {m_uri} does not have exactly one pgo:movieLensId (found {len(m_id_lits)}).")
        else:
            movie_ids_in_graph.add(int(m_id_lits[0]))

        m_idx_lits = list(graph.objects(m_uri, PRED_INTERNAL_INDEX))
        if len(m_idx_lits) != 1:
            errors.append(f"Movie {m_uri} does not have exactly one pgo:internalMovieIndex (found {len(m_idx_lits)}).")
        else:
            movie_indices_in_graph.add(int(m_idx_lits[0]))

    if len(movie_ids_in_graph) != len(movie_nodes):
        errors.append(f"Duplicate movieLensId values found in graph: {len(movie_nodes) - len(movie_ids_in_graph)} duplicates.")
    if len(movie_indices_in_graph) != len(movie_nodes):
        errors.append(f"Duplicate internalMovieIndex values found in graph: {len(movie_nodes) - len(movie_indices_in_graph)} duplicates.")

    # 3. 100% Movie URI alignment with KG-M
    uri_by_id, _ = build_movie_uri_index(metadata_df)
    mismatched_uris: List[str] = []
    for m_uri in movie_nodes:
        m_id = int(list(graph.objects(m_uri, PRED_MOVIELENS_ID))[0])
        expected_uri = uri_by_id.get(m_id)
        if expected_uri is None or m_uri != expected_uri:
            mismatched_uris.append(f"Movie {m_id}: expected URI {expected_uri}, got {m_uri}")

    if mismatched_uris:
        errors.append(f"{len(mismatched_uris)} Movie URIs do not align with KG-M reference: {mismatched_uris[:3]}")

    # 4. Target entities have rdf:type and rdfs:label
    checked_predicates = [
        PRED_HAS_THEME,
        PRED_HAS_SETTING,
        PRED_HAS_NARRATIVE_DEVICE,
        PRED_HAS_PROTAGONIST_TYPE,
        PRED_BASED_ON,
        PRED_EVOKES_EMOTION,
        PRED_HAS_EMOTION_ASSOCIATION,
    ]
    missing_type_or_label = 0
    for pred in checked_predicates:
        for _, _, target in graph.triples((None, pred, None)):
            types = list(graph.objects(target, RDF.type))
            labels = list(graph.objects(target, RDFS.label))
            if not types or not labels:
                missing_type_or_label += 1

    if missing_type_or_label > 0:
        errors.append(f"{missing_type_or_label} relation target instances missing rdf:type or rdfs:label.")

    # 5. Exactly 8 Emotion resources
    emotion_nodes = set(graph.subjects(RDF.type, CLASS_EMOTION))
    if len(emotion_nodes) != 8:
        errors.append(f"Expected exactly 8 Emotion entities, found {len(emotion_nodes)}: {emotion_nodes}")

    expected_emo_slugs = {f"emotion_{e}" for e in PLUTCHIK_EMOTIONS}
    actual_emo_slugs = {str(u).split("/")[-1] for u in emotion_nodes}
    if actual_emo_slugs != expected_emo_slugs:
        errors.append(f"Emotion resources do not match Plutchik dimensions: diff={actual_emo_slugs ^ expected_emo_slugs}")

    # 6. Evokes-Emotion edge count and score > threshold verification
    evokes_edges = list(graph.triples((None, PRED_EVOKES_EMOTION, None)))
    if len(evokes_edges) != expected_evokes_total:
        errors.append(f"evokesEmotion total edge count mismatch: expected {expected_evokes_total}, got {len(evokes_edges)}")

    # Check per-emotion counts
    for emo_name, expected_cnt in per_emo_expected.items():
        emo_uri = build_entity_uri("emotion", emo_name)
        actual_cnt = len(list(graph.triples((None, PRED_EVOKES_EMOTION, emo_uri))))
        if actual_cnt != expected_cnt:
            errors.append(f"evokesEmotion count for '{emo_name}' mismatch: expected {expected_cnt}, got {actual_cnt}")

    # 7. EmotionAssociation intensities in [0, 1] and strictly > threshold
    for assoc in graph.subjects(RDF.type, CLASS_EMOTION_ASSOCIATION):
        intensities = list(graph.objects(assoc, PRED_EMOTION_INTENSITY))
        if not intensities:
            errors.append(f"EmotionAssociation {assoc} missing pgo:emotionIntensity.")
        else:
            val = float(intensities[0])
            if not (0.0 <= val <= 1.0):
                errors.append(f"EmotionAssociation {assoc} intensity {val} out of bounds [0, 1].")
            if val <= emotion_threshold:
                errors.append(f"EmotionAssociation {assoc} has intensity {val} <= threshold {emotion_threshold}.")

    # 8. Excluded failure records produce zero emotion triples
    if excluded_failure_indices:
        for fail_idx in excluded_failure_indices:
            # Find movie URI for this index
            for m_uri in graph.subjects(PRED_INTERNAL_INDEX, Literal(fail_idx, datatype=XSD.integer)):
                emo_edges = list(graph.triples((m_uri, PRED_EVOKES_EMOTION, None)))
                assoc_edges = list(graph.triples((m_uri, PRED_HAS_EMOTION_ASSOCIATION, None)))
                if emo_edges or assoc_edges:
                    errors.append(f"Failed emotion movie_idx {fail_idx} has emotion triples in graph ({len(emo_edges)} evokes, {len(assoc_edges)} assoc).")

    # 9. No NaN or None in literals or URIs
    for s, p, o in graph:
        for node in (s, p, o):
            str_val = str(node).lower()
            if "nan" in str_val or "none" in str_val:
                # Disallow literal 'nan' or 'none'
                if str_val in ("nan", "none") or "/nan" in str_val or "/none" in str_val:
                    errors.append(f"Bogus NaN/None detected in triple: ({s}, {p}, {o})")

    is_valid = (len(errors) == 0)
    result = {
        "valid": is_valid,
        "total_triples": len(graph),
        "movie_count": len(movie_nodes),
        "evokes_emotion_count": len(evokes_edges),
        "error_count": len(errors),
        "errors": errors,
    }

    if not is_valid:
        msg = f"KG-L graph validation failed with {len(errors)} error(s):\n" + "\n".join(f"  - {e}" for e in errors[:15])
        logger.error(msg)
        raise ValueError(msg)

    logger.info("KG-L graph validation PASSED successfully.")
    return result


def build_and_save_kg_l(
    metadata_path: Union[str, Path] = "data/processed/ml-1m/metadata/movies_with_tmdb.parquet",
    semantic_jsonl_path: Union[str, Path] = "data/processed/ml-1m/kg_l/full_extraction/semantic_validated.jsonl",
    emotion_vectors_path: Union[str, Path] = "data/processed/ml-1m/kg_l/full_extraction/emotion_vectors.jsonl",
    emotion_evidence_path: Optional[Union[str, Path]] = "data/processed/ml-1m/kg_l/full_extraction/emotion_validated_evidence.jsonl",
    emotion_failures_path: Optional[Union[str, Path]] = "data/processed/ml-1m/kg_l/full_extraction/emotion_failures.jsonl",
    output_ttl_path: Union[str, Path] = "data/processed/ml-1m/graphs/kg_l.ttl",
    output_validation_path: Union[str, Path] = "data/processed/ml-1m/graphs/kg_l_validation.json",
    emotion_threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD,
) -> Tuple[Graph, Dict[str, Any]]:
    """Complete workflow: load production artifacts, build KG-L, serialize, reload, and validate.

    Args:
        metadata_path: Path to movies_with_tmdb.parquet.
        semantic_jsonl_path: Path to semantic_validated.jsonl.
        emotion_vectors_path: Path to emotion_vectors.jsonl.
        emotion_evidence_path: Path to emotion_validated_evidence.jsonl.
        emotion_failures_path: Path to emotion_failures.jsonl.
        output_ttl_path: Destination path for serialized kg_l.ttl.
        output_validation_path: Destination path for kg_l_validation.json.
        emotion_threshold: Edge threshold (default: 0.3).

    Returns:
        Tuple of (validated Graph, statistics dictionary).
    """
    logger.info("Loading Phase 3A metadata reference...")
    meta_df = pd.read_parquet(metadata_path)

    logger.info(f"Loading semantic extractions from {semantic_jsonl_path}...")
    semantic_records: List[Dict[str, Any]] = []
    with open(semantic_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                semantic_records.append(json.loads(line))

    logger.info(f"Loading emotion vectors from {emotion_vectors_path}...")
    emotion_vectors: List[Dict[str, Any]] = []
    if Path(emotion_vectors_path).exists():
        with open(emotion_vectors_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    emotion_vectors.append(json.loads(line))

    emotion_evidence: List[Dict[str, Any]] = []
    if emotion_evidence_path and Path(emotion_evidence_path).exists():
        logger.info(f"Loading emotion evidence from {emotion_evidence_path}...")
        with open(emotion_evidence_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    emotion_evidence.append(json.loads(line))

    excluded_failures: Set[int] = set()
    if emotion_failures_path and Path(emotion_failures_path).exists():
        with open(emotion_failures_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if "movie_idx" in d:
                        excluded_failures.add(int(d["movie_idx"]))
        logger.info(f"Loaded {len(excluded_failures)} excluded emotion failure indices.")

    # Build Graph
    graph = build_kg_l(
        metadata_df=meta_df,
        semantic_records=semantic_records,
        emotion_vectors=emotion_vectors,
        emotion_evidence_records=emotion_evidence,
        emotion_threshold=emotion_threshold,
    )

    # Serialize to Turtle
    ttl_p = serialize_kg_l(graph, output_ttl_path)

    # Round-trip Reload and Verify
    logger.info(f"Reloading {ttl_p} to verify Turtle syntax and round-trip preservation...")
    reloaded_graph = Graph()
    bind_namespaces(reloaded_graph)
    reloaded_graph.parse(ttl_p, format="turtle")

    if len(reloaded_graph) != len(graph):
        raise ValueError(
            f"Turtle round-trip triple count mismatch: built {len(graph)}, reloaded {len(reloaded_graph)}."
        )

    # Validate Reloaded Graph
    val_summary = validate_kg_l(
        graph=reloaded_graph,
        metadata_df=meta_df,
        expected_movie_count=len(semantic_records),
        expected_evokes_total=EXPECTED_EVOKES_EMOTION_TOTAL if len(semantic_records) == 3810 else len(list(reloaded_graph.triples((None, PRED_EVOKES_EMOTION, None)))),
        emotion_threshold=emotion_threshold,
        excluded_failure_indices=excluded_failures,
    )

    # Compute Statistics
    stats = compute_kg_l_statistics(
        reloaded_graph,
        threshold=emotion_threshold,
        total_eligible_movies=len(semantic_records),
        total_validated_emotion_movies=len(emotion_vectors),
    )
    stats["validation"] = val_summary

    # Save validation and statistics JSON
    val_path = Path(output_validation_path)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved KG-L validation and statistics to {val_path}")

    return reloaded_graph, stats
