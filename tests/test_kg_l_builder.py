"""Unit and Integration Tests for PlotGraph KG-L Knowledge Graph (Phase 3G).

Verifies:
1. KG-M / KG-L Movie URI Identity and recovery.
2. Semantic entity canonicalization and lack of fuzzy merging.
3. Target classes and predicates for all semantic relations.
4. Exactly 8 Emotion resources.
5. Evokes-Emotion edge selection boundary (score > 0.3 strictly: 0.30 -> no edge, 0.31 -> edge).
6. Continuous score preservation and plot evidence provenance.
7. Failed emotion extractions excluded from emotion triples while retaining semantic data.
8. Turtle serialization, syntax validity, and round-trip triple preservation.
9. KGLQueryEngine SPARQL queries (themes, settings, devices, protagonists, sources, emotions, evidence, shared paths).
10. Cross-graph compatibility demonstration (KG-M + KG-L).
11. Failure classification regression (score=0 with evidence classified as schema_validation_error, NOT id_mismatch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import pytest
from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD

from src.knowledge_graph.kg_l_contract import (
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    PLUTCHIK_EMOTIONS,
    canonicalize_semantic_entity,
    ExtractionParseError,
    ExtractionValidationError,
)
from src.knowledge_graph.namespaces import (
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
    PGR,
    bind_namespaces,
    build_entity_uri,
    build_movie_uri,
    build_movie_uri_index,
    get_duplicate_tmdb_ids,
)
from src.knowledge_graph.kg_l_builder import (
    EXPECTED_EVOKES_EMOTION_TOTAL,
    EXPECTED_PER_EMOTION_COUNTS,
    build_kg_l,
    compute_kg_l_statistics,
    serialize_kg_l,
    validate_kg_l,
)
from src.knowledge_graph.kg_l_query_engine import (
    KGLQueryEngine,
    demonstrate_cross_graph_compatibility,
)
from src.knowledge_graph.qwen_production import categorize_extraction_error

METADATA_PARQUET_PATH = Path("data/processed/ml-1m/metadata/movies_with_tmdb.parquet")
KG_M_PATH = Path("data/processed/ml-1m/graphs/kg_m.ttl")


# ==============================================================================
# 1. MOVIE URI IDENTITY ALIGNMENT TESTS
# ==============================================================================

def test_kg_m_kg_l_movie_uri_identity():
    """Verify that KG-L movie URIs match KG-M movie URIs 100% across the full catalog."""
    if not METADATA_PARQUET_PATH.exists():
        pytest.skip(f"Metadata parquet not found at {METADATA_PARQUET_PATH}")

    df_meta = pd.read_parquet(METADATA_PARQUET_PATH)
    uri_by_id, uri_by_idx = build_movie_uri_index(df_meta)

    assert len(uri_by_id) == len(df_meta)
    assert len(uri_by_idx) == len(df_meta)

    # 1. Duplicate TMDB IDs must be handled with MovieLens ID suffix
    dup_tmdb_ids = get_duplicate_tmdb_ids(df_meta)
    if dup_tmdb_ids:
        for dup_id in dup_tmdb_ids:
            dup_rows = df_meta[df_meta["tmdb_id"] == dup_id]
            for _, row in dup_rows.iterrows():
                m_id = int(row["movie_id"])
                uri = uri_by_id[m_id]
                assert f"movie_tmdb_{dup_id}_ml_{m_id}" in str(uri)

    # 2. TMDB-linked unique movies must use movie_tmdb_{tmdb_id}
    toy_story = df_meta[df_meta["movie_id"] == 1].iloc[0]
    ts_uri = uri_by_id[1]
    assert ts_uri == PGR[f"movie_tmdb_{int(toy_story['tmdb_id'])}"]

    # 3. Unlinked MovieLens movies must use movie_ml_{movie_id}
    unlinked = df_meta[df_meta["tmdb_id"].isna()]
    if not unlinked.empty:
        unlinked_id = int(unlinked.iloc[0]["movie_id"])
        assert uri_by_id[unlinked_id] == PGR[f"movie_ml_{unlinked_id}"]

    # 4. Compare directly with kg_m.ttl if available on disk
    if KG_M_PATH.exists():
        g_m = Graph()
        g_m.parse(KG_M_PATH, format="turtle")
        for m_uri, _, m_id_lit in g_m.triples((None, PRED_MOVIELENS_ID, None)):
            m_id = int(m_id_lit)
            assert m_uri == uri_by_id[m_id], f"Mismatch for movie_id {m_id}: {m_uri} != {uri_by_id[m_id]}"


# ==============================================================================
# 2. CANONICALIZATION & NO FUZZY MERGING TESTS
# ==============================================================================

def test_semantic_canonicalization_rules():
    """Verify Unicode NFKC normalization, whitespace collapsing, and no fuzzy merging."""
    # Whitespace and case normalization
    assert canonicalize_semantic_entity("  Space Travel  ") == "space travel"
    assert canonicalize_semantic_entity("Space\n\tTravel") == "space travel"
    assert canonicalize_semantic_entity("UNRELIABLE NARRATOR") == "unreliable narrator"

    # Distinct concepts remain distinct (no fuzzy merging)
    assert canonicalize_semantic_entity("space travel") != canonicalize_semantic_entity("interstellar travel")
    assert canonicalize_semantic_entity("space travel") != canonicalize_semantic_entity("travel through space")
    assert canonicalize_semantic_entity("coming of age") != canonicalize_semantic_entity("youth")


# ==============================================================================
# 3. SYNTHETIC KG-L CONSTRUCTION & BOUNDARY TESTS
# ==============================================================================

def make_synthetic_metadata():
    return pd.DataFrame([
        {"movie_idx": 0, "movie_id": 1, "tmdb_id": 862, "ml_title": "Toy Story (1995)", "has_overview": True, "overview": "Plot 1"},
        {"movie_idx": 1, "movie_id": 2, "tmdb_id": 8844, "ml_title": "Jumanji (1995)", "has_overview": True, "overview": "Plot 2"},
        {"movie_idx": 2, "movie_id": 3, "tmdb_id": None, "ml_title": "Movie Three (1995)", "has_overview": True, "overview": "Plot 3"},
    ])


def test_kg_l_construction_and_threshold_boundaries():
    """Verify KG-L construction, relation mapping, and strictly score > 0.3 threshold boundary."""
    df_meta = make_synthetic_metadata()

    semantic_records = [
        {
            "movie_idx": 0,
            "movie_id": 1,
            "title": "Toy Story (1995)",
            "themes": ["Friendship", "Jealousy"],
            "settings": ["Child's bedroom"],
            "narrative_devices": ["Dramatic irony"],
            "protagonist_types": ["Sentient toy"],
            "based_on": ["Original story"],
        },
        {
            "movie_idx": 1,
            "movie_id": 2,
            "title": "Jumanji (1995)",
            "themes": ["Friendship"],
            "settings": ["Jungle", "Mansion"],
            "narrative_devices": [],
            "protagonist_types": ["Child"],
            "based_on": [],
        },
        {
            "movie_idx": 2,
            "movie_id": 3,
            "title": "Movie Three (1995)",
            "themes": [],
            "settings": [],
            "narrative_devices": [],
            "protagonist_types": [],
            "based_on": [],
        },
    ]

    # Test emotion boundary behavior:
    # Movie 1:
    # - joy: 0.31 -> edge (>0.3)
    # - trust: 0.30 -> NO edge (not >0.3)
    # - fear: 0.29 -> NO edge
    # - surprise: 0.40 -> edge (>0.3)
    # - sadness: 0.0 -> NO edge
    # - disgust: 0.0 -> NO edge
    # - anger: 0.0 -> NO edge
    # - anticipation: 0.8 -> edge (>0.3)
    emotion_vectors = [
        {
            "movie_idx": 0,
            "movie_id": 1,
            "title": "Toy Story (1995)",
            "joy": 0.31,
            "trust": 0.30,
            "fear": 0.29,
            "surprise": 0.40,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.80,
        },
        # Movie 2: all emotions <= 0.30
        {
            "movie_idx": 1,
            "movie_id": 2,
            "title": "Jumanji (1995)",
            "joy": 0.30,
            "trust": 0.25,
            "fear": 0.30,
            "surprise": 0.10,
            "sadness": 0.20,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.30,
        },
        # Movie 3 is omitted from emotion vectors (simulating failed emotion extraction)
    ]

    emotion_evidence = [
        {
            "movie_idx": 0,
            "movie_id": 1,
            "emotions": {
                "joy": {"score": 0.31, "evidence": ["Reunion celebration"]},
                "trust": {"score": 0.30, "evidence": ["Loyal toys"]},
                "fear": {"score": 0.29, "evidence": ["Sid's dog"]},
                "surprise": {"score": 0.40, "evidence": ["Buzz realizes he cannot fly"]},
                "anticipation": {"score": 0.80, "evidence": ["Chasing moving truck"]},
            },
        }
    ]

    graph = build_kg_l(
        metadata_df=df_meta,
        semantic_records=semantic_records,
        emotion_vectors=emotion_vectors,
        emotion_evidence_records=emotion_evidence,
        emotion_threshold=0.3,
    )

    # 1. Exactly 8 Emotion resources
    emotions = list(graph.subjects(RDF.type, CLASS_EMOTION))
    assert len(emotions) == 8
    for e in emotions:
        labels = list(graph.objects(e, RDFS.label))
        assert len(labels) == 1
        assert str(labels[0]) in PLUTCHIK_EMOTIONS

    # 2. Movie 1 Evokes-Emotion Edge Boundaries:
    m1_uri = PGR["movie_tmdb_862"]
    m1_evoked = {str(o).split("_")[-1] for _, _, o in graph.triples((m1_uri, PRED_EVOKES_EMOTION, None))}

    # 0.31 -> edge
    assert "joy" in m1_evoked
    # 0.40 -> edge
    assert "surprise" in m1_evoked
    # 0.80 -> edge
    assert "anticipation" in m1_evoked

    # 0.30 -> NO edge
    assert "trust" not in m1_evoked
    # 0.29 -> NO edge
    assert "fear" not in m1_evoked
    # 0.0 -> NO edge
    assert "sadness" not in m1_evoked

    assert len(m1_evoked) == 3

    # 3. Movie 2 (all <= 0.30): zero evokesEmotion edges
    m2_uri = PGR["movie_tmdb_8844"]
    m2_evoked = list(graph.triples((m2_uri, PRED_EVOKES_EMOTION, None)))
    assert len(m2_evoked) == 0

    # 4. Movie 3 (excluded from emotion): zero evokesEmotion edges
    m3_uri = PGR["movie_ml_3"]
    m3_evoked = list(graph.triples((m3_uri, PRED_EVOKES_EMOTION, None)))
    assert len(m3_evoked) == 0

    # 5. Continuous score and evidence provenance on EmotionAssociation
    assoc_list = list(graph.objects(m1_uri, PRED_HAS_EMOTION_ASSOCIATION))
    assert len(assoc_list) == 3

    for assoc in assoc_list:
        intensity = float(list(graph.objects(assoc, PRED_EMOTION_INTENSITY))[0])
        assert intensity > 0.3
        # Evidence quotes present
        ev_quotes = list(graph.objects(assoc, PRED_SUPPORTING_EVIDENCE))
        assert len(ev_quotes) >= 1

    # 6. Target entities have rdf:type and rdfs:label
    theme_friendship = build_entity_uri("theme", "friendship")
    assert (theme_friendship, RDF.type, CLASS_THEME) in graph
    assert (theme_friendship, RDFS.label, Literal("friendship")) in graph

    device_irony = build_entity_uri("device", "dramatic irony")
    assert (device_irony, RDF.type, CLASS_NARRATIVE_DEVICE) in graph


# ==============================================================================
# 4. SERIALIZATION & ROUND-TRIP PRESERVATION TESTS
# ==============================================================================

def test_kg_l_serialization_and_roundtrip(tmp_path: Path):
    """Verify Turtle serialization and round-trip preservation."""
    df_meta = make_synthetic_metadata()
    semantic_records = [
        {
            "movie_idx": 0,
            "movie_id": 1,
            "title": "Toy Story (1995)",
            "themes": ["Friendship"],
            "settings": ["City"],
            "narrative_devices": ["Flashback"],
            "protagonist_types": ["Toy"],
            "based_on": ["Book"],
        }
    ]
    emotion_vectors = [
        {
            "movie_idx": 0,
            "movie_id": 1,
            "joy": 0.75,
            "trust": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.0,
        }
    ]

    graph = build_kg_l(
        metadata_df=df_meta,
        semantic_records=semantic_records,
        emotion_vectors=emotion_vectors,
    )
    initial_triples = len(graph)

    out_file = tmp_path / "test_kg_l.ttl"
    serialize_kg_l(graph, out_file)
    assert out_file.exists()

    reloaded = Graph()
    reloaded.parse(out_file, format="turtle")
    assert len(reloaded) == initial_triples


# ==============================================================================
# 5. SPARQL QUERY ENGINE TESTS
# ==============================================================================

def test_kgl_query_engine_methods():
    """Verify that KGLQueryEngine executes queries across all semantic and emotion facets."""
    df_meta = make_synthetic_metadata()
    semantic_records = [
        {
            "movie_idx": 0,
            "movie_id": 1,
            "title": "Toy Story (1995)",
            "themes": ["Loyalty", "Friendship"],
            "settings": ["Suburbia"],
            "narrative_devices": ["Dramatic irony"],
            "protagonist_types": ["Sentient toy"],
            "based_on": [],
        },
        {
            "movie_idx": 1,
            "movie_id": 2,
            "title": "Jumanji (1995)",
            "themes": ["Friendship", "Survival"],
            "settings": ["Mansion"],
            "narrative_devices": [],
            "protagonist_types": ["Student"],
            "based_on": ["Children's book"],
        },
    ]
    emotion_vectors = [
        {
            "movie_idx": 0,
            "movie_id": 1,
            "title": "Toy Story (1995)",
            "joy": 0.85,
            "trust": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.70,
        },
        {
            "movie_idx": 1,
            "movie_id": 2,
            "title": "Jumanji (1995)",
            "joy": 0.50,
            "trust": 0.0,
            "fear": 0.60,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.65,
        },
    ]
    emotion_evidence = [
        {
            "movie_idx": 0,
            "movie_id": 1,
            "emotions": {
                "joy": {"score": 0.85, "evidence": ["Toys happily reunited"]},
                "anticipation": {"score": 0.70, "evidence": ["Catching the moving truck"]},
            },
        }
    ]

    graph = build_kg_l(
        metadata_df=df_meta,
        semantic_records=semantic_records,
        emotion_vectors=emotion_vectors,
        emotion_evidence_records=emotion_evidence,
    )

    engine = KGLQueryEngine(graph)

    # 1. get_movie_themes
    themes_1 = engine.get_movie_themes(movie_id=1)
    assert len(themes_1) == 2
    theme_labels = {t["theme_label"] for t in themes_1}
    assert "loyalty" in theme_labels
    assert "friendship" in theme_labels

    # 2. get_movie_settings
    settings_1 = engine.get_movie_settings(movie_id=1)
    assert len(settings_1) == 1
    assert settings_1[0]["setting_label"] == "suburbia"

    # 3. get_movie_narrative_devices
    devices_1 = engine.get_movie_narrative_devices(movie_id=1)
    assert len(devices_1) == 1
    assert devices_1[0]["device_label"] == "dramatic irony"

    # 4. get_movie_protagonist_types
    prots_1 = engine.get_movie_protagonist_types(movie_id=1)
    assert len(prots_1) == 1
    assert prots_1[0]["protagonist_label"] == "sentient toy"

    # 5. get_movie_source_materials
    sources_2 = engine.get_movie_source_materials(movie_id=2)
    assert len(sources_2) == 1
    assert sources_2[0]["source_label"] == "children's book"

    # 6. get_movie_emotions (continuous scores)
    emos_1 = engine.get_movie_emotions(movie_id=1)
    assert len(emos_1) == 2
    emo_dict_1 = {e["emotion_label"]: float(e["intensity"]) for e in emos_1}
    assert emo_dict_1["joy"] == 0.85
    assert emo_dict_1["anticipation"] == 0.70

    # 7. get_movie_emotion_evidence
    evidence_1 = engine.get_movie_emotion_evidence(movie_id=1, emotion="joy")
    assert len(evidence_1) == 1
    assert evidence_1[0]["evidence"] == ["Toys happily reunited"]

    # 8. find_movies_by_theme
    movies_friendship = engine.find_movies_by_theme("friendship")
    assert len(movies_friendship) == 2
    f_ids = {int(m["movie_id"]) for m in movies_friendship}
    assert f_ids == {1, 2}

    # 9. find_movies_by_emotion
    movies_joy = engine.find_movies_by_emotion("joy")
    assert len(movies_joy) == 2
    # Sorted by intensity desc: movie 1 (0.85) then movie 2 (0.50)
    assert int(movies_joy[0]["movie_id"]) == 1
    assert int(movies_joy[1]["movie_id"]) == 2

    # 10. find_shared_semantic_paths between 1 and 2
    shared_sem = engine.find_shared_semantic_paths(1, 2)
    assert len(shared_sem) == 1
    assert shared_sem[0]["label"] == "friendship"

    # 11. find_shared_emotion_paths between 1 and 2
    shared_emo = engine.find_shared_emotion_paths(1, 2)
    assert len(shared_emo) == 2  # shared joy and anticipation
    shared_labels = {e["emotion_label"] for e in shared_emo}
    assert shared_labels == {"joy", "anticipation"}


# ==============================================================================
# 6. CROSS-GRAPH COMPATIBILITY DEMONSTRATION TESTS
# ==============================================================================

def test_cross_graph_compatibility_demo():
    """Verify cross-graph joining and multi-hop traversal across KG-M and KG-L."""
    df_meta = make_synthetic_metadata()
    m1_uri = PGR["movie_tmdb_862"]

    # Create dummy KG-M
    g_m = Graph()
    from src.knowledge_graph.namespaces import CLASS_GENRE, PRED_DIRECTED_BY, PRED_HAS_GENRE
    bind_namespaces(g_m)
    dir_uri = build_entity_uri("person", "john lasseter")
    genre_uri = build_entity_uri("genre", "animation")
    g_m.add((m1_uri, RDF.type, CLASS_MOVIE))
    g_m.add((m1_uri, RDFS.label, Literal("Toy Story (1995)")))
    g_m.add((m1_uri, PRED_MOVIELENS_ID, Literal(1, datatype=XSD.integer)))
    g_m.add((m1_uri, PRED_DIRECTED_BY, dir_uri))
    g_m.add((dir_uri, RDFS.label, Literal("John Lasseter")))
    g_m.add((m1_uri, PRED_HAS_GENRE, genre_uri))
    g_m.add((genre_uri, RDFS.label, Literal("Animation")))

    # Create dummy KG-L
    g_l = build_kg_l(
        metadata_df=df_meta,
        semantic_records=[{"movie_idx": 0, "movie_id": 1, "title": "Toy Story (1995)", "themes": ["Friendship"]}],
        emotion_vectors=[{"movie_idx": 0, "movie_id": 1, "joy": 0.8, "anticipation": 0.0}],
    )

    report = demonstrate_cross_graph_compatibility(g_m, g_l, sample_movie_id=1)
    assert report["sample_movie_id"] == 1
    assert "John Lasseter" in report["kg_m_facts"]["directors"]
    assert "Animation" in report["kg_m_facts"]["genres"]
    assert "friendship" in report["kg_l_facts"]["themes"]
    assert "joy" in report["kg_l_facts"]["emotions"]


# ==============================================================================
# 7. FAILURE CLASSIFICATION REGRESSION TESTS
# ==============================================================================

def test_failure_classification_regression():
    """Verify that score=0 with evidence is classified as schema_validation_error, NOT id_mismatch."""
    # Bug scenario: error message contains the word 'evidence' (which contains substring 'id')
    err_evidence_schema = ExtractionValidationError(
        "Emotion schema validation failed: 1 validation error for MovieEmotionExtraction\n"
        "emotions.trust\n"
        "  Value error, Emotion score is 0.0, so evidence list must be empty [], got: ['daughter meets men']"
    )
    assert categorize_extraction_error(err_evidence_schema, "{}") == "schema_validation_error"

    # True ID mismatch
    err_id_mismatch = ExtractionValidationError("movie_id mismatch: expected 1, got 2")
    assert categorize_extraction_error(err_id_mismatch, "{}") == "id_mismatch"

    # Syntax error
    err_parse = ExtractionParseError("Malformed JSON syntax")
    assert categorize_extraction_error(err_parse, "{}") == "invalid_json"

    # Backend error
    err_backend = RuntimeError("CUDA out of memory in backend worker")
    assert categorize_extraction_error(err_backend, "{}") == "backend_error"


# ==============================================================================
# 8. REAL PRODUCTION ARTIFACT VERIFICATION TEST
# ==============================================================================

def test_real_production_kg_l_validation():
    """Verify that the generated production KG-L graph and validation report match all frozen criteria."""
    val_path = Path("data/processed/ml-1m/graphs/kg_l_validation.json")
    ttl_path = Path("data/processed/ml-1m/graphs/kg_l.ttl")

    if not val_path.exists() or not ttl_path.exists():
        pytest.skip("Production KG-L artifacts not yet generated on disk.")

    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Validate overall integrity
    assert val_data["validation"]["valid"] is True
    assert val_data["validation"]["error_count"] == 0
    assert val_data["total_triples"] == 97904
    assert val_data["entities"]["movies"] == 3810
    assert val_data["entities"]["emotions"] == 8
    assert val_data["entities"]["emotion_associations"] == 5452
    assert val_data["relations"]["evokesEmotion"] == 5452

    # Validate exact per-emotion counts matching production summary
    expected_emotions = {
        "joy": 589,
        "trust": 201,
        "fear": 1028,
        "surprise": 595,
        "sadness": 731,
        "disgust": 350,
        "anger": 543,
        "anticipation": 1415,
    }
    assert val_data["emotion_metrics"]["per_emotion_counts"] == expected_emotions

    # Validate movie coverage metrics
    assert val_data["emotion_metrics"]["movies_with_at_least_one_emotion_edge"] == 2870
    assert val_data["emotion_metrics"]["movies_with_no_emotion_edge"] == 940
    assert val_data["emotion_metrics"]["movies_missing_validated_emotion_extraction"] == 22
    assert val_data["semantic_coverage"]["movies_with_theme"] == 3801
    assert val_data["semantic_coverage"]["movies_with_setting"] == 3805

