"""Unit and integration tests for Phase 3B: Knowledge Graph (KG-M) Construction and SPARQL queries."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD

from src.knowledge_graph import (
    CLASS_COLLECTION,
    CLASS_GENRE,
    CLASS_KEYWORD,
    CLASS_MOVIE,
    CLASS_PERSON,
    CLASS_RELEASE_DECADE,
    PGO,
    PGR,
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
    KGMQueryEngine,
    build_entity_uri,
    build_kg_m,
    build_movie_uri,
    canonicalize_slug,
    serialize_kg_m,
    validate_kg_m,
)


def test_canonicalize_slug_determinism_and_unicode():
    """Verify slug generation handles whitespace, capitalization, punctuation, and Unicode."""
    # Basic casing and spacing
    assert canonicalize_slug("Tom Hanks") == "tom_hanks"
    assert canonicalize_slug("  TOM   HANKS  ") == "tom_hanks"

    # Punctuation and symbols
    assert canonicalize_slug("Sci-Fi & Fantasy") == "sci_fi_fantasy"
    assert canonicalize_slug("Wall-E") == "wall_e"
    assert canonicalize_slug("Grumpy Old Men (Collection)") == "grumpy_old_men_collection"
    assert canonicalize_slug("neo-nazi") == "neo_nazi"

    # Unicode letters (accents, umlauts, non-Latin)
    assert canonicalize_slug("François Truffaut") == "françois_truffaut"
    assert canonicalize_slug("Jean-Luc Godard") == "jean_luc_godard"
    assert canonicalize_slug("Hayao Miyazaki") == "hayao_miyazaki"

    # Fallback on all-punctuation strings
    slug_punct = canonicalize_slug("???!!!")
    assert slug_punct.startswith("ent_")

    # Error handling on empty/nan
    with pytest.raises(ValueError):
        canonicalize_slug("")
    with pytest.raises(ValueError):
        canonicalize_slug(None)
    with pytest.raises(ValueError):
        canonicalize_slug("nan")
    with pytest.raises(ValueError):
        canonicalize_slug("None")


def test_build_entity_and_movie_uris():
    """Verify entity URI construction and collision-resistant movie URI generation."""
    # Entity URIs
    person_uri = build_entity_uri("person", "John Lasseter")
    assert str(person_uri) == "https://plotgraph.org/resource/person_john_lasseter"

    genre_uri = build_entity_uri("genre", "Animation")
    assert str(genre_uri) == "https://plotgraph.org/resource/genre_animation"

    keyword_uri = build_entity_uri("keyword", "toy comes to life")
    assert str(keyword_uri) == "https://plotgraph.org/resource/keyword_toy_comes_to_life"

    coll_uri = build_entity_uri("collection", "Toy Story Collection")
    assert str(coll_uri) == "https://plotgraph.org/resource/collection_toy_story_collection"

    decade_uri = build_entity_uri("decade", "1990s")
    assert str(decade_uri) == "https://plotgraph.org/resource/decade_1990s"

    # Movie URIs: unique TMDB ID
    m_tmdb = build_movie_uri(movie_id=1, tmdb_id=862, is_tmdb_duplicate=False)
    assert str(m_tmdb) == "https://plotgraph.org/resource/movie_tmdb_862"

    # Movie URIs: duplicate TMDB ID (disambiguated by movie_id)
    m_dup1 = build_movie_uri(movie_id=690, tmdb_id=105045, is_tmdb_duplicate=True)
    m_dup2 = build_movie_uri(movie_id=1533, tmdb_id=105045, is_tmdb_duplicate=True)
    assert str(m_dup1) == "https://plotgraph.org/resource/movie_tmdb_105045_ml_690"
    assert str(m_dup2) == "https://plotgraph.org/resource/movie_tmdb_105045_ml_1533"
    assert m_dup1 != m_dup2

    # Movie URIs: unmapped movie (no TMDB ID)
    m_unmapped = build_movie_uri(movie_id=142, tmdb_id=None)
    assert str(m_unmapped) == "https://plotgraph.org/resource/movie_ml_142"


def test_synthetic_graph_construction_and_relations():
    """Verify graph construction on a controlled synthetic metadata DataFrame."""
    synthetic_df = pd.DataFrame({
        "movie_id": [1, 2, 3],
        "movie_idx": [0, 1, 2],
        "ml_title": ["Toy Story (1995)", "Jumanji (1995)", "Shadows (1988)"],
        "ml_genres": ["Animation|Comedy", "Adventure|Children's", "Drama"],
        "tmdb_id": [862, 8844, np.nan],
        "tmdb_title": ["Toy Story", "Jumanji", None],
        "overview": ["Toys come alive", "Magical board game", None],
        "release_date": ["1995-10-30", "1995-12-15", None],
        "release_decade": ["1990s", "1990s", None],
        "tmdb_genres": [["Animation", "Comedy"], ["Adventure"], None],
        "collection_name": ["Toy Story Collection", None, None],
        "director": ["John Lasseter", "Joe Johnston", None],
        "top_actors": [["Tom Hanks", "Tim Allen"], ["Robin Williams"], None],
        "keywords": [["toy", "friendship"], ["board game"], None],
        "has_tmdb_link": [True, True, False],
        "has_overview": [True, True, False],
        "has_credits": [True, True, False],
        "has_keywords": [True, True, False],
    })

    graph = build_kg_m(synthetic_df, top_k_actors=5)

    # 1. Total movie nodes
    movie_nodes = list(graph.subjects(RDF.type, CLASS_MOVIE))
    assert len(movie_nodes) == 3

    # 2. Movie 1 check (fully connected)
    m1_uri = PGR["movie_tmdb_862"]
    assert (m1_uri, RDF.type, CLASS_MOVIE) in graph
    assert (m1_uri, PRED_TITLE, Literal("Toy Story (1995)")) in graph
    assert (m1_uri, PRED_MOVIELENS_ID, Literal(1, datatype=XSD.integer)) in graph
    assert (m1_uri, PRED_INTERNAL_INDEX, Literal(0, datatype=XSD.integer)) in graph
    assert (m1_uri, PRED_TMDB_ID, Literal(862, datatype=XSD.integer)) in graph
    assert (m1_uri, PRED_RELEASE_DATE, Literal("1995-10-30")) in graph

    # Director relation
    dir_uri = PGR["person_john_lasseter"]
    assert (m1_uri, PRED_DIRECTED_BY, dir_uri) in graph
    assert (dir_uri, RDF.type, CLASS_PERSON) in graph
    assert (dir_uri, RDFS.label, Literal("John Lasseter")) in graph

    # Actors relation
    act_uri = PGR["person_tom_hanks"]
    assert (m1_uri, PRED_STARS_ACTOR, act_uri) in graph
    assert (act_uri, RDF.type, CLASS_PERSON) in graph

    # Collection relation
    col_uri = PGR["collection_toy_story_collection"]
    assert (m1_uri, PRED_IN_COLLECTION, col_uri) in graph
    assert (col_uri, RDF.type, CLASS_COLLECTION) in graph

    # Genre relation
    gen_uri = PGR["genre_animation"]
    assert (m1_uri, PRED_HAS_GENRE, gen_uri) in graph
    assert (gen_uri, RDF.type, CLASS_GENRE) in graph

    # Decade relation
    dec_uri = PGR["decade_1990s"]
    assert (m1_uri, PRED_RELEASE_DECADE, dec_uri) in graph
    assert (dec_uri, RDF.type, CLASS_RELEASE_DECADE) in graph

    # 3. Movie 3 check (unmapped catalog movie)
    m3_uri = PGR["movie_ml_3"]
    assert (m3_uri, RDF.type, CLASS_MOVIE) in graph
    assert (m3_uri, PRED_TITLE, Literal("Shadows (1988)")) in graph
    assert (m3_uri, PRED_MOVIELENS_ID, Literal(3, datatype=XSD.integer)) in graph
    assert (m3_uri, PRED_INTERNAL_INDEX, Literal(2, datatype=XSD.integer)) in graph
    # Must NOT have any dangling or hallucinated relations
    assert len(list(graph.objects(m3_uri, PRED_DIRECTED_BY))) == 0
    assert len(list(graph.objects(m3_uri, PRED_STARS_ACTOR))) == 0
    assert len(list(graph.objects(m3_uri, PRED_HAS_GENRE))) == 0
    assert len(list(graph.objects(m3_uri, PRED_HAS_KEYWORD))) == 0
    assert len(list(graph.objects(m3_uri, PRED_IN_COLLECTION))) == 0

    # 4. Validation report on synthetic graph
    report = validate_kg_m(graph, input_df=synthetic_df)
    assert report["integrity"]["valid"] is True
    assert report["entities"]["movies"] == 3
    assert report["integrity"]["all_movie_ids_recoverable"] is True
    assert report["integrity"]["all_movie_indices_recoverable"] is True


def test_graph_determinism():
    """Verify that building KG-M twice from identical data produces identical triples."""
    df = pd.DataFrame({
        "movie_id": [1, 2],
        "movie_idx": [0, 1],
        "ml_title": ["Toy Story (1995)", "Jumanji (1995)"],
        "tmdb_id": [862, 8844],
        "director": ["John Lasseter", "Joe Johnston"],
        "tmdb_genres": [["Animation"], ["Adventure"]],
        "release_decade": ["1990s", "1990s"],
    })

    g1 = build_kg_m(df)
    g2 = build_kg_m(df)

    assert len(g1) == len(g2)
    assert set(g1) == set(g2)


def test_turtle_serialization_and_round_trip(tmp_path):
    """Verify Turtle serialization and round-trip parsing."""
    df = pd.DataFrame({
        "movie_id": [1],
        "movie_idx": [0],
        "ml_title": ["Toy Story (1995)"],
        "tmdb_id": [862],
        "director": ["John Lasseter"],
        "top_actors": [["Tom Hanks"]],
        "tmdb_genres": [["Animation"]],
        "release_decade": ["1990s"],
    })

    graph = build_kg_m(df)
    ttl_path = tmp_path / "test.ttl"
    serialize_kg_m(graph, ttl_path)

    assert ttl_path.exists()
    assert ttl_path.stat().st_size > 0

    reloaded = Graph()
    reloaded.parse(ttl_path, format="turtle")
    assert len(reloaded) == len(graph)
    assert set(reloaded) == set(graph)


def test_sparql_query_engine_helpers():
    """Verify SPARQL query helper methods in KGMQueryEngine."""
    df = pd.DataFrame({
        "movie_id": [1, 2, 3],
        "movie_idx": [0, 1, 2],
        "ml_title": ["Toy Story (1995)", "Toy Story 2 (1999)", "Forrest Gump (1994)"],
        "tmdb_id": [862, 3114, 13],
        "tmdb_title": ["Toy Story", "Toy Story 2", "Forrest Gump"],
        "release_date": ["1995-10-30", "1999-10-30", "1994-07-06"],
        "release_decade": ["1990s", "1990s", "1990s"],
        "tmdb_genres": [["Animation", "Comedy"], ["Animation", "Comedy"], ["Drama", "Romance"]],
        "collection_name": ["Toy Story Collection", "Toy Story Collection", None],
        "director": ["John Lasseter", "John Lasseter", "Robert Zemeckis"],
        "top_actors": [["Tom Hanks", "Tim Allen"], ["Tom Hanks", "Tim Allen"], ["Tom Hanks", "Robin Wright"]],
        "keywords": [["toy", "friendship"], ["toy", "sequel"], ["running", "friendship"]],
    })

    graph = build_kg_m(df)
    engine = KGMQueryEngine(graph)

    # 1. get_movie_metadata
    meta1 = engine.get_movie_metadata(1)
    assert meta1 is not None
    assert meta1["title"] == "Toy Story (1995)"
    assert meta1["director"] == "John Lasseter"
    assert "Tom Hanks" in meta1["actors"]
    assert "Animation" in meta1["genres"]
    assert meta1["collection"] == "Toy Story Collection"

    # 2. find_movies_sharing_director
    shared_dir = engine.find_movies_sharing_director(1)
    assert len(shared_dir) == 1
    assert shared_dir[0]["otherTitle"] == "Toy Story 2 (1999)"

    # 3. find_movies_sharing_actor
    shared_actors = engine.find_movies_sharing_actor(1)
    shared_titles = {m["otherTitle"] for m in shared_actors}
    assert "Toy Story 2 (1999)" in shared_titles
    assert "Forrest Gump (1994)" in shared_titles

    # 4. find_movies_in_same_collection
    same_coll = engine.find_movies_in_same_collection(1)
    assert len(same_coll) == 1
    assert same_coll[0]["otherTitle"] == "Toy Story 2 (1999)"

    # 5. find_movies_sharing_keyword
    shared_kw = engine.find_movies_sharing_keyword(1)
    assert len(shared_kw) == 2
    # Toy Story 2 shares "toy", Forrest Gump shares "friendship"

    # 6. find_paths_between_movies
    paths = engine.find_paths_between_movies(1, 2)
    assert len(paths) >= 4  # shared director, actor, genres, collection, decade
    pred_types = {(p["predicate_a"], p["entity_type"]) for p in paths}
    assert ("directedBy", "Person") in pred_types
    assert ("starsActor", "Person") in pred_types
    assert ("hasGenre", "Genre") in pred_types
    assert ("inCollection", "Collection") in pred_types
    assert ("releaseDecade", "ReleaseDecade") in pred_types


def test_real_data_kg_m_integration():
    """Verify integration properties of the generated kg_m.ttl on real data."""
    ttl_path = Path("data/processed/ml-1m/graphs/kg_m.ttl")
    if not ttl_path.exists():
        pytest.skip("Phase 3B KG-M file not generated yet")

    engine = KGMQueryEngine(ttl_path)
    val_report = validate_kg_m(engine.graph)

    # 1. Total triples and movie count
    assert val_report["total_triples"] > 120_000
    assert val_report["entities"]["movies"] == 3883
    assert val_report["entities"]["persons"] > 10_000
    assert val_report["entities"]["genres"] == 20
    assert val_report["entities"]["keywords"] > 7_000
    assert val_report["entities"]["collections"] > 300

    # 2. Integrity assertions
    assert val_report["integrity"]["valid"] is True
    assert val_report["integrity"]["has_bogus_uris"] is False
    assert val_report["integrity"]["has_bogus_labels"] is False
    assert val_report["integrity"]["all_movie_ids_recoverable"] is True
    assert val_report["integrity"]["all_movie_indices_recoverable"] is True

    # 3. SPARQL test on real data: Toy Story (movie_id=1)
    meta = engine.get_movie_metadata(1)
    assert meta is not None
    assert meta["title"] == "Toy Story (1995)"
    assert meta["director"] == "John Lasseter"
    assert "Tom Hanks" in meta["actors"]
    assert "Animation" in meta["genres"]
    assert meta["collection"] == "Toy Story Collection"

    # 4. Multi-hop connection between Toy Story (1) and A Bug's Life (2355)
    paths = engine.find_paths_between_movies(1, 2355)
    assert len(paths) >= 4
    entity_labels = {p["entity_label"] for p in paths}
    assert "John Lasseter" in entity_labels
    assert "Animation" in entity_labels
    assert "1990s" in entity_labels
