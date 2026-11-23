"""SPARQL Query Wrapper and Graph Store Utilities for PlotGraph KG-L (Phase 3G).

Provides high-level querying methods over the LLM-enriched knowledge graph:
- Semantic entity retrieval per movie (themes, settings, narrative devices, protagonist types, source materials)
- Prominent emotion retrieval with continuous intensity scores
- Grounded plot evidence quote retrieval
- Semantic and emotion inverted entity lookups (movies sharing themes, settings, emotions)
- Graph path finding across semantic entities and emotions
- Cross-graph (KG-M + KG-L) joint query and path compatibility demonstrations
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
    PRED_DIRECTED_BY,
    PRED_EMOTION_INTENSITY,
    PRED_EVOKES_EMOTION,
    PRED_HAS_EMOTION_ASSOCIATION,
    PRED_HAS_GENRE,
    PRED_HAS_NARRATIVE_DEVICE,
    PRED_HAS_PROTAGONIST_TYPE,
    PRED_HAS_SETTING,
    PRED_HAS_THEME,
    PRED_INTERNAL_INDEX,
    PRED_MOVIELENS_ID,
    PRED_SUPPORTING_EVIDENCE,
    PRED_TITLE,
    PRED_TMDB_ID,
    bind_namespaces,
    build_entity_uri,
    get_sparql_prefixes_header,
)

logger = logging.getLogger(__name__)


class KGLQueryEngine:
    """SPARQL querying wrapper and graph analysis utility for PlotGraph KG-L."""

    def __init__(self, graph_or_path: Union[Graph, str, Path]) -> None:
        """Initialize query engine with an existing Graph or file path.

        Args:
            graph_or_path: In-memory RDFlib Graph or path to kg_l.ttl.
        """
        if isinstance(graph_or_path, (str, Path)):
            path = Path(graph_or_path)
            if not path.exists():
                raise FileNotFoundError(f"Knowledge graph file not found at {path}")
            logger.info(f"Loading KG-L graph from {path}...")
            self.graph = Graph()
            bind_namespaces(self.graph)
            self.graph.parse(path, format="turtle")
            logger.info(f"Loaded KG-L graph with {len(self.graph):,} triples.")
        else:
            self.graph = graph_or_path
            bind_namespaces(self.graph)

        self._prefixes_header = get_sparql_prefixes_header()

    def query(
        self,
        sparql_query: str,
        init_bindings: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a SPARQL query on the KG-L graph, returning results as native Python dicts."""
        full_query = sparql_query
        if "PREFIX" not in sparql_query.upper():
            full_query = self._prefixes_header + sparql_query

        results = self.graph.query(full_query, initBindings=init_bindings)
        rows: List[Dict[str, Any]] = []

        for row in results:
            row_dict: Dict[str, Any] = {}
            for var in results.vars:
                val = getattr(row, str(var), None)
                if val is not None:
                    if isinstance(val, Literal):
                        row_dict[str(var)] = val.toPython()
                    elif isinstance(val, URIRef):
                        row_dict[str(var)] = str(val)
                    else:
                        row_dict[str(var)] = str(val)
                else:
                    row_dict[str(var)] = None
            rows.append(row_dict)

        return rows

    def get_movie_uri(
        self,
        movie_id: Optional[int] = None,
        movie_idx: Optional[int] = None,
        tmdb_id: Optional[int] = None,
    ) -> Optional[URIRef]:
        """Resolve a movie resource URIRef from movie_id, movie_idx, or tmdb_id."""
        if movie_id is not None:
            matches = list(self.graph.subjects(PRED_MOVIELENS_ID, Literal(int(movie_id), datatype=XSD.integer)))
            if matches:
                return matches[0]

        if movie_idx is not None:
            matches = list(self.graph.subjects(PRED_INTERNAL_INDEX, Literal(int(movie_idx), datatype=XSD.integer)))
            if matches:
                return matches[0]

        if tmdb_id is not None:
            matches = list(self.graph.subjects(PRED_TMDB_ID, Literal(int(tmdb_id), datatype=XSD.integer)))
            if matches:
                return matches[0]

        return None

    def _resolve_movie(
        self,
        movie_id: Optional[int] = None,
        movie_idx: Optional[int] = None,
        movie_uri: Optional[Union[str, URIRef]] = None,
    ) -> Optional[URIRef]:
        """Helper to resolve a Movie URIRef from ID, index, or URI string."""
        if movie_uri is not None:
            return URIRef(str(movie_uri))
        return self.get_movie_uri(movie_id=movie_id, movie_idx=movie_idx)

    # --------------------------------------------------------------------------
    # Semantic Queries (A - E)
    # --------------------------------------------------------------------------

    def get_movie_themes(
        self,
        movie_id: Optional[int] = None,
        movie_idx: Optional[int] = None,
        movie_uri: Optional[Union[str, URIRef]] = None,
    ) -> List[Dict[str, str]]:
        """Retrieve all themes for a movie (Query A)."""
        m_uri = self._resolve_movie(movie_id, movie_idx, movie_uri)
        if not m_uri:
            return []
        sparql = """
        SELECT ?theme ?theme_label WHERE {
            ?movie pgo:hasTheme ?theme .
            ?theme rdfs:label ?theme_label .
        } ORDER BY ?theme_label
        """
        return self.query(sparql, init_bindings={"movie": m_uri})

    def get_movie_settings(
        self,
        movie_id: Optional[int] = None,
        movie_idx: Optional[int] = None,
        movie_uri: Optional[Union[str, URIRef]] = None,
    ) -> List[Dict[str, str]]:
        """Retrieve all settings for a movie (Query B)."""
        m_uri = self._resolve_movie(movie_id, movie_idx, movie_uri)
        if not m_uri:
            return []
        sparql = """
        SELECT ?setting ?setting_label WHERE {
            ?movie pgo:hasSetting ?setting .
            ?setting rdfs:label ?setting_label .
        } ORDER BY ?setting_label
        """
        return self.query(sparql, init_bindings={"movie": m_uri})

    def get_movie_narrative_devices(
        self,
        movie_id: Optional[int] = None,
        movie_idx: Optional[int] = None,
        movie_uri: Optional[Union[str, URIRef]] = None,
    ) -> List[Dict[str, str]]:
        """Retrieve all narrative devices for a movie (Query C)."""
        m_uri = self._resolve_movie(movie_id, movie_idx, movie_uri)
        if not m_uri:
            return []
        sparql = """
        SELECT ?device ?device_label WHERE {
            ?movie pgo:hasNarrativeDevice ?device .
            ?device rdfs:label ?device_label .
        } ORDER BY ?device_label
        """
        return self.query(sparql, init_bindings={"movie": m_uri})

    def get_movie_protagonist_types(
        self,
        movie_id: Optional[int] = None,
        movie_idx: Optional[int] = None,
        movie_uri: Optional[Union[str, URIRef]] = None,
    ) -> List[Dict[str, str]]:
        """Retrieve all protagonist types for a movie (Query D)."""
        m_uri = self._resolve_movie(movie_id, movie_idx, movie_uri)
        if not m_uri:
            return []
        sparql = """
        SELECT ?protagonist ?protagonist_label WHERE {
            ?movie pgo:hasProtagonistType ?protagonist .
            ?protagonist rdfs:label ?protagonist_label .
        } ORDER BY ?protagonist_label
        """
        return self.query(sparql, init_bindings={"movie": m_uri})

    def get_movie_source_materials(
        self,
        movie_id: Optional[int] = None,
        movie_idx: Optional[int] = None,
        movie_uri: Optional[Union[str, URIRef]] = None,
    ) -> List[Dict[str, str]]:
        """Retrieve all source materials ('based on') for a movie (Query E)."""
        m_uri = self._resolve_movie(movie_id, movie_idx, movie_uri)
        if not m_uri:
            return []
        sparql = """
        SELECT ?source ?source_label WHERE {
            ?movie pgo:basedOn ?source .
            ?source rdfs:label ?source_label .
        } ORDER BY ?source_label
        """
        return self.query(sparql, init_bindings={"movie": m_uri})

    # --------------------------------------------------------------------------
    # Emotion Queries (F - G)
    # --------------------------------------------------------------------------

    def get_movie_emotions(
        self,
        movie_id: Optional[int] = None,
        movie_idx: Optional[int] = None,
        movie_uri: Optional[Union[str, URIRef]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all evoked emotions with continuous intensity scores (Query F)."""
        m_uri = self._resolve_movie(movie_id, movie_idx, movie_uri)
        if not m_uri:
            return []
        sparql = """
        SELECT ?emotion ?emotion_label ?intensity WHERE {
            ?movie pgo:hasEmotionAssociation ?assoc .
            ?assoc pgo:associatedEmotion ?emotion ;
                   pgo:emotionIntensity ?intensity .
            ?emotion rdfs:label ?emotion_label .
        } ORDER BY DESC(?intensity)
        """
        return self.query(sparql, init_bindings={"movie": m_uri})

    def get_movie_emotion_evidence(
        self,
        movie_id: Optional[int] = None,
        movie_idx: Optional[int] = None,
        movie_uri: Optional[Union[str, URIRef]] = None,
        emotion: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve supporting text evidence quotes and scores per emotion (Query G)."""
        m_uri = self._resolve_movie(movie_id, movie_idx, movie_uri)
        if not m_uri:
            return []

        sparql = """
        SELECT ?emotion ?emotion_label ?intensity ?evidence WHERE {
            ?movie pgo:hasEmotionAssociation ?assoc .
            ?assoc pgo:associatedEmotion ?emotion ;
                   pgo:emotionIntensity ?intensity ;
                   pgo:supportingEvidence ?evidence .
            ?emotion rdfs:label ?emotion_label .
        } ORDER BY ?emotion_label
        """
        rows = self.query(sparql, init_bindings={"movie": m_uri})

        # Group evidence quotes by emotion
        grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"intensity": 0.0, "evidence": []})
        for r in rows:
            emo_lbl = str(r["emotion_label"]).lower()
            if emotion and emo_lbl != emotion.lower():
                continue
            grouped[emo_lbl]["emotion_uri"] = r["emotion"]
            grouped[emo_lbl]["emotion_label"] = emo_lbl
            grouped[emo_lbl]["intensity"] = float(r["intensity"])
            if r["evidence"]:
                grouped[emo_lbl]["evidence"].append(r["evidence"])

        return list(grouped.values())

    # --------------------------------------------------------------------------
    # Inverted Entity Lookups (H - L)
    # --------------------------------------------------------------------------

    def find_movies_by_theme(self, theme_text: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find movies connected to a specific theme (Query H)."""
        slug = canonicalize_semantic_entity(theme_text)
        t_uri = build_entity_uri("theme", slug)
        sparql = f"""
        SELECT ?movie ?movie_id ?title WHERE {{
            ?movie pgo:hasTheme <{t_uri}> ;
                   pgo:movieLensId ?movie_id ;
                   rdfs:label ?title .
        }} ORDER BY ?title LIMIT {int(limit)}
        """
        return self.query(sparql)

    def find_movies_by_setting(self, setting_text: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find movies connected to a specific setting (Query I)."""
        slug = canonicalize_semantic_entity(setting_text)
        s_uri = build_entity_uri("setting", slug)
        sparql = f"""
        SELECT ?movie ?movie_id ?title WHERE {{
            ?movie pgo:hasSetting <{s_uri}> ;
                   pgo:movieLensId ?movie_id ;
                   rdfs:label ?title .
        }} ORDER BY ?title LIMIT {int(limit)}
        """
        return self.query(sparql)

    def find_movies_by_narrative_device(self, device_text: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find movies using a specific narrative device (Query J)."""
        slug = canonicalize_semantic_entity(device_text)
        d_uri = build_entity_uri("device", slug)
        sparql = f"""
        SELECT ?movie ?movie_id ?title WHERE {{
            ?movie pgo:hasNarrativeDevice <{d_uri}> ;
                   pgo:movieLensId ?movie_id ;
                   rdfs:label ?title .
        }} ORDER BY ?title LIMIT {int(limit)}
        """
        return self.query(sparql)

    def find_movies_by_protagonist_type(self, protagonist_text: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find movies featuring a specific protagonist type (Query K)."""
        slug = canonicalize_semantic_entity(protagonist_text)
        p_uri = build_entity_uri("protagonist", slug)
        sparql = f"""
        SELECT ?movie ?movie_id ?title WHERE {{
            ?movie pgo:hasProtagonistType <{p_uri}> ;
                   pgo:movieLensId ?movie_id ;
                   rdfs:label ?title .
        }} ORDER BY ?title LIMIT {int(limit)}
        """
        return self.query(sparql)

    def find_movies_by_emotion(
        self,
        emotion_name: str,
        min_intensity: float = DEFAULT_EMOTION_EDGE_THRESHOLD,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find movies that evoke an emotion with intensity > min_intensity (Query L)."""
        emo_uri = build_entity_uri("emotion", emotion_name.lower())
        sparql = f"""
        SELECT ?movie ?movie_id ?title ?intensity WHERE {{
            ?movie pgo:hasEmotionAssociation ?assoc ;
                   pgo:movieLensId ?movie_id ;
                   rdfs:label ?title .
            ?assoc pgo:associatedEmotion <{emo_uri}> ;
                   pgo:emotionIntensity ?intensity .
            FILTER (?intensity > {float(min_intensity)})
        }} ORDER BY DESC(?intensity) LIMIT {int(limit)}
        """
        return self.query(sparql)

    # --------------------------------------------------------------------------
    # Multi-hop Path Queries (M - N)
    # --------------------------------------------------------------------------

    def find_shared_semantic_paths(
        self,
        movie_id_1: int,
        movie_id_2: int,
    ) -> List[Dict[str, Any]]:
        """Find semantic entities (themes, settings, devices, etc.) shared by two movies (Query M)."""
        m1 = self.get_movie_uri(movie_id=movie_id_1)
        m2 = self.get_movie_uri(movie_id=movie_id_2)
        if not m1 or not m2:
            return []

        sparql = f"""
        SELECT ?relation ?entity ?label WHERE {{
            <{m1}> ?relation ?entity .
            <{m2}> ?relation ?entity .
            ?entity rdfs:label ?label .
            FILTER (?relation IN (pgo:hasTheme, pgo:hasSetting, pgo:hasNarrativeDevice, pgo:hasProtagonistType, pgo:basedOn))
        }} ORDER BY ?relation ?label
        """
        return self.query(sparql)

    def find_shared_emotion_paths(
        self,
        movie_id_1: int,
        movie_id_2: int,
    ) -> List[Dict[str, Any]]:
        """Find evoked emotions shared by two movies with respective intensities (Query N)."""
        m1 = self.get_movie_uri(movie_id=movie_id_1)
        m2 = self.get_movie_uri(movie_id=movie_id_2)
        if not m1 or not m2:
            return []

        sparql = f"""
        SELECT ?emotion ?emotion_label ?intensity1 ?intensity2 WHERE {{
            <{m1}> pgo:hasEmotionAssociation ?assoc1 .
            ?assoc1 pgo:associatedEmotion ?emotion ;
                    pgo:emotionIntensity ?intensity1 .

            <{m2}> pgo:hasEmotionAssociation ?assoc2 .
            ?assoc2 pgo:associatedEmotion ?emotion ;
                    pgo:emotionIntensity ?intensity2 .

            ?emotion rdfs:label ?emotion_label .
        }} ORDER BY ?emotion_label
        """
        return self.query(sparql)


# ==============================================================================
# Cross-Graph (KG-M + KG-L) Compatibility Demonstrator
# ==============================================================================

def demonstrate_cross_graph_compatibility(
    kg_m_source: Union[Graph, str, Path],
    kg_l_source: Union[Graph, str, Path],
    sample_movie_id: int = 1,
) -> Dict[str, Any]:
    """Demonstrate that KG-M and KG-L use identical Movie URIs and can be queried jointly.

    Demonstrates:
    1. Single Movie resource participates in both metadata (KG-M) and semantic (KG-L) graphs.
    2. Cross-graph multi-hop traversal paths connecting movies via KG-M predicates
       (e.g. genre/director) and KG-L predicates (e.g. themes/emotions).

    Args:
        kg_m_source: In-memory Graph or path to kg_m.ttl.
        kg_l_source: In-memory Graph or path to kg_l.ttl.
        sample_movie_id: MovieLens ID to inspect (default: 1, Toy Story).

    Returns:
        Structured demonstration report dictionary.
    """
    logger.info("Setting up cross-graph compatibility demonstration...")
    union_graph = Graph()
    bind_namespaces(union_graph)

    if isinstance(kg_m_source, (str, Path)):
        logger.info(f"Parsing KG-M from {kg_m_source} into union graph...")
        union_graph.parse(str(kg_m_source), format="turtle")
    else:
        for t in kg_m_source:
            union_graph.add(t)

    if isinstance(kg_l_source, (str, Path)):
        logger.info(f"Parsing KG-L from {kg_l_source} into union graph...")
        union_graph.parse(str(kg_l_source), format="turtle")
    else:
        for t in kg_l_source:
            union_graph.add(t)

    logger.info(f"Combined union graph has {len(union_graph):,} triples.")

    prefixes = get_sparql_prefixes_header()

    # Query 1: Retrieve unified metadata + semantic profile for sample movie
    profile_query = prefixes + f"""
    SELECT ?movie ?title ?director ?genre ?theme ?setting ?emotion WHERE {{
        ?movie pgo:movieLensId {int(sample_movie_id)} ;
               rdfs:label ?title .

        OPTIONAL {{
            ?movie pgo:directedBy ?dir_res .
            ?dir_res rdfs:label ?director .
        }}
        OPTIONAL {{
            ?movie pgo:hasGenre ?genre_res .
            ?genre_res rdfs:label ?genre .
        }}
        OPTIONAL {{
            ?movie pgo:hasTheme ?theme_res .
            ?theme_res rdfs:label ?theme .
        }}
        OPTIONAL {{
            ?movie pgo:hasSetting ?set_res .
            ?set_res rdfs:label ?setting .
        }}
        OPTIONAL {{
            ?movie pgo:evokesEmotion ?emo_res .
            ?emo_res rdfs:label ?emotion .
        }}
    }}
    """
    profile_rows = list(union_graph.query(profile_query))

    movie_uri = None
    title = ""
    directors: Set[str] = set()
    genres: Set[str] = set()
    themes: Set[str] = set()
    settings: Set[str] = set()
    emotions: Set[str] = set()

    for r in profile_rows:
        movie_uri = str(r[0])
        title = str(r[1])
        if r[2]:
            directors.add(str(r[2]))
        if r[3]:
            genres.add(str(r[3]))
        if r[4]:
            themes.add(str(r[4]))
        if r[5]:
            settings.add(str(r[5]))
        if r[6]:
            emotions.add(str(r[6]))

    # Query 2: Cross-Graph Path Traversal
    # Find movies that share a Director or Genre in KG-M AND share a Theme or Emotion in KG-L
    cross_path_query = prefixes + f"""
    SELECT DISTINCT ?other_movie ?other_title ?bridge_metadata ?bridge_semantic WHERE {{
        <{movie_uri}> pgo:hasGenre ?genre_res .
        ?other_movie pgo:hasGenre ?genre_res ;
                     rdfs:label ?other_title ;
                     pgo:hasTheme ?theme_res .
        <{movie_uri}> pgo:hasTheme ?theme_res .

        ?genre_res rdfs:label ?bridge_metadata .
        ?theme_res rdfs:label ?bridge_semantic .

        FILTER (?other_movie != <{movie_uri}>)
    }} LIMIT 5
    """
    cross_rows = list(union_graph.query(cross_path_query))
    cross_matches = [
        {
            "other_movie_uri": str(r[0]),
            "other_title": str(r[1]),
            "kg_m_bridge_genre": str(r[2]),
            "kg_l_bridge_theme": str(r[3]),
        }
        for r in cross_rows
    ]

    report = {
        "sample_movie_id": sample_movie_id,
        "sample_movie_uri": movie_uri,
        "sample_movie_title": title,
        "kg_m_facts": {
            "directors": sorted(directors),
            "genres": sorted(genres),
        },
        "kg_l_facts": {
            "themes": sorted(themes),
            "settings": sorted(settings),
            "emotions": sorted(emotions),
        },
        "cross_graph_paths": cross_matches,
        "union_triple_count": len(union_graph),
    }

    # Print human-readable report
    print(
        "\n"
        "======================================================================\n"
        "       KG-M + KG-L UNIFIED COMPATIBILITY DEMONSTRATION REPORT\n"
        "======================================================================\n"
        f"Sample Movie: {title} (ID: {sample_movie_id})\n"
        f"Unified URI : {movie_uri}\n"
        "----------------------------------------------------------------------\n"
        "KG-M Structural Facts (Phase 3B Metadata):\n"
        f"  Directors : {', '.join(sorted(directors)) or 'None'}\n"
        f"  Genres    : {', '.join(sorted(genres)) or 'None'}\n"
        "----------------------------------------------------------------------\n"
        "KG-L Semantic Facts (Phase 3G LLM Extractions):\n"
        f"  Themes    : {', '.join(sorted(themes)) or 'None'}\n"
        f"  Settings  : {', '.join(sorted(settings)) or 'None'}\n"
        f"  Emotions  : {', '.join(sorted(emotions)) or 'None'}\n"
        "----------------------------------------------------------------------\n"
        f"Cross-Graph Path Matches (Shared KG-M Genre + Shared KG-L Theme): {len(cross_matches)}\n",
        flush=True,
    )
    for m in cross_matches:
        print(f"  * {title} --[hasGenre]--> {m['kg_m_bridge_genre']} <--[hasGenre]-- {m['other_title']}")
        print(f"    {title} --[hasTheme]--> {m['kg_l_bridge_theme']} <--[hasTheme]-- {m['other_title']}")
    print("======================================================================\n", flush=True)

    return report
