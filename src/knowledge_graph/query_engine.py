"""SPARQL Query Wrapper and Graph Store Utilities for KG-M."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD

from .namespaces import (
    CLASS_MOVIE,
    ONTOLOGY_URI,
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
    get_sparql_prefixes_header,
)

logger = logging.getLogger(__name__)


class KGMQueryEngine:
    """SPARQL querying wrapper and graph-store utility for PlotGraph KG-M."""

    def __init__(self, graph_or_path: Union[Graph, str, Path]) -> None:
        """Initialize query engine with an existing Graph or file path.

        Args:
            graph_or_path: In-memory RDFlib Graph, or file path to .ttl file.
        """
        if isinstance(graph_or_path, (str, Path)):
            path = Path(graph_or_path)
            if not path.exists():
                raise FileNotFoundError(f"Knowledge graph file not found at {path}")
            logger.info(f"Loading KG-M graph from {path}...")
            self.graph = Graph()
            bind_namespaces(self.graph)
            self.graph.parse(path, format="turtle")
            logger.info(f"Loaded graph with {len(self.graph):,} triples.")
        else:
            self.graph = graph_or_path
            bind_namespaces(self.graph)

        self._prefixes_header = get_sparql_prefixes_header()

    def query(
        self,
        sparql_query: str,
        init_bindings: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a SPARQL query on the knowledge graph.

        Prepends standard PlotGraph prefixes if not already declared in query.

        Args:
            sparql_query: SPARQL query string.
            init_bindings: Optional dictionary of variable bindings.

        Returns:
            List of result row dictionaries with native Python types.
        """
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
        """Find the Movie resource URI given a MovieLens ID, movie index, or TMDB ID.

        Args:
            movie_id: Original MovieLens movie ID.
            movie_idx: Internal 0-indexed movie index.
            tmdb_id: TMDB ID.

        Returns:
            Movie URIRef, or None if not found.
        """
        if movie_id is not None:
            val = Literal(int(movie_id), datatype=XSD.integer)
            for s in self.graph.subjects(PRED_MOVIELENS_ID, val):
                return s

        if movie_idx is not None:
            val = Literal(int(movie_idx), datatype=XSD.integer)
            for s in self.graph.subjects(PRED_INTERNAL_INDEX, val):
                return s

        if tmdb_id is not None:
            val = Literal(int(tmdb_id), datatype=XSD.integer)
            for s in self.graph.subjects(PRED_TMDB_ID, val):
                return s

        return None

    def _resolve_movie(self, movie: Union[int, URIRef]) -> Optional[URIRef]:
        """Resolve an integer MovieLens ID or URIRef to a valid Movie URIRef."""
        if isinstance(movie, URIRef):
            return movie
        if isinstance(movie, (int, str)):
            try:
                m_id = int(movie)
                return self.get_movie_uri(movie_id=m_id)
            except ValueError:
                return URIRef(str(movie))
        return None

    def get_movie_metadata(self, movie: Union[int, URIRef]) -> Optional[Dict[str, Any]]:
        """Get all structured metadata for a movie in KG-M.

        Args:
            movie: MovieLens movie ID (int) or Movie URIRef.

        Returns:
            Dictionary with title, IDs, director, actors, genres, keywords, collection, decade.
        """
        movie_uri = self._resolve_movie(movie)
        if movie_uri is None or (movie_uri, RDF.type, CLASS_MOVIE) not in self.graph:
            return None

        # Base attributes
        ml_id = next(self.graph.objects(movie_uri, PRED_MOVIELENS_ID), None)
        ml_idx = next(self.graph.objects(movie_uri, PRED_INTERNAL_INDEX), None)
        title = next(self.graph.objects(movie_uri, PRED_TITLE), None)
        tmdb_id = next(self.graph.objects(movie_uri, PRED_TMDB_ID), None)
        tmdb_title = next(self.graph.objects(movie_uri, PRED_TMDB_TITLE), None)
        rel_date = next(self.graph.objects(movie_uri, PRED_RELEASE_DATE), None)

        # Decade
        decade_uri = next(self.graph.objects(movie_uri, PRED_RELEASE_DECADE), None)
        decade_label = (
            next(self.graph.objects(decade_uri, RDFS.label), None) if decade_uri else None
        )

        # Director
        director_uri = next(self.graph.objects(movie_uri, PRED_DIRECTED_BY), None)
        director_name = (
            next(self.graph.objects(director_uri, RDFS.label), None) if director_uri else None
        )

        # Collection
        collection_uri = next(self.graph.objects(movie_uri, PRED_IN_COLLECTION), None)
        collection_name = (
            next(self.graph.objects(collection_uri, RDFS.label), None) if collection_uri else None
        )

        # Actors
        actors = []
        for act_uri in self.graph.objects(movie_uri, PRED_STARS_ACTOR):
            act_label = next(self.graph.objects(act_uri, RDFS.label), None)
            if act_label:
                actors.append(str(act_label))

        # Genres
        genres = []
        for gen_uri in self.graph.objects(movie_uri, PRED_HAS_GENRE):
            gen_label = next(self.graph.objects(gen_uri, RDFS.label), None)
            if gen_label:
                genres.append(str(gen_label))

        # Keywords
        keywords = []
        for kw_uri in self.graph.objects(movie_uri, PRED_HAS_KEYWORD):
            kw_label = next(self.graph.objects(kw_uri, RDFS.label), None)
            if kw_label:
                keywords.append(str(kw_label))

        return {
            "movie_uri": str(movie_uri),
            "movie_id": int(ml_id) if ml_id is not None else None,
            "movie_idx": int(ml_idx) if ml_idx is not None else None,
            "title": str(title) if title is not None else None,
            "tmdb_id": int(tmdb_id) if tmdb_id is not None else None,
            "tmdb_title": str(tmdb_title) if tmdb_title is not None else None,
            "release_date": str(rel_date) if rel_date is not None else None,
            "release_decade": str(decade_label) if decade_label is not None else None,
            "director": str(director_name) if director_name is not None else None,
            "actors": actors,
            "genres": genres,
            "keywords": keywords,
            "collection": str(collection_name) if collection_name is not None else None,
        }

    def find_movies_sharing_director(
        self,
        movie: Union[int, URIRef],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find other movies sharing a director with the given movie.

        Args:
            movie: MovieLens movie ID or Movie URIRef.
            limit: Maximum results to return.

        Returns:
            List of dictionaries with movie_uri, movie_id, title, and director_name.
        """
        movie_uri = self._resolve_movie(movie)
        if not movie_uri:
            return []

        q = """
        SELECT DISTINCT ?otherMovie ?otherTitle ?otherMovieId ?directorName WHERE {
            ?movie pgo:directedBy ?director .
            ?otherMovie pgo:directedBy ?director ;
                        pgo:title ?otherTitle ;
                        pgo:movieLensId ?otherMovieId .
            ?director rdfs:label ?directorName .
            FILTER(?otherMovie != ?movie)
        }
        ORDER BY ?otherTitle
        LIMIT %d
        """ % limit
        return self.query(q, init_bindings={"movie": movie_uri})

    def find_movies_sharing_actor(
        self,
        movie: Union[int, URIRef],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find other movies sharing at least one top-billed actor with the given movie.

        Args:
            movie: MovieLens movie ID or Movie URIRef.
            limit: Maximum results to return.

        Returns:
            List of dictionaries with movie_uri, movie_id, title, and actor_name.
        """
        movie_uri = self._resolve_movie(movie)
        if not movie_uri:
            return []

        q = """
        SELECT ?otherMovie ?otherTitle ?otherMovieId ?actorName WHERE {
            ?movie pgo:starsActor ?actor .
            ?otherMovie pgo:starsActor ?actor ;
                        pgo:title ?otherTitle ;
                        pgo:movieLensId ?otherMovieId .
            ?actor rdfs:label ?actorName .
            FILTER(?otherMovie != ?movie)
        }
        ORDER BY ?otherTitle
        LIMIT %d
        """ % limit
        return self.query(q, init_bindings={"movie": movie_uri})

    def find_movies_sharing_genre(
        self,
        movie: Union[int, URIRef],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find other movies sharing genres with the given movie, ranked by overlap.

        Args:
            movie: MovieLens movie ID or Movie URIRef.
            limit: Maximum results to return.

        Returns:
            List of dictionaries with movie_uri, movie_id, title, and shared_genres.
        """
        movie_uri = self._resolve_movie(movie)
        if not movie_uri:
            return []

        q = """
        SELECT ?otherMovie ?otherTitle ?otherMovieId (GROUP_CONCAT(DISTINCT ?genreName; separator=", ") AS ?sharedGenres)
        WHERE {
            ?movie pgo:hasGenre ?genre .
            ?otherMovie pgo:hasGenre ?genre ;
                        pgo:title ?otherTitle ;
                        pgo:movieLensId ?otherMovieId .
            ?genre rdfs:label ?genreName .
            FILTER(?otherMovie != ?movie)
        }
        GROUP BY ?otherMovie ?otherTitle ?otherMovieId
        ORDER BY DESC(COUNT(DISTINCT ?genre)) ?otherTitle
        LIMIT %d
        """ % limit
        return self.query(q, init_bindings={"movie": movie_uri})

    def find_movies_sharing_keyword(
        self,
        movie: Union[int, URIRef],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find other movies sharing plot keywords, ranked by number of shared keywords.

        Args:
            movie: MovieLens movie ID or Movie URIRef.
            limit: Maximum results to return.

        Returns:
            List of dictionaries with movie_uri, movie_id, title, shared_keywords, count.
        """
        movie_uri = self._resolve_movie(movie)
        if not movie_uri:
            return []

        q = """
        SELECT ?otherMovie ?otherTitle ?otherMovieId (COUNT(DISTINCT ?keyword) AS ?sharedKeywordCount) (GROUP_CONCAT(DISTINCT ?keywordName; separator=", ") AS ?sharedKeywords)
        WHERE {
            ?movie pgo:hasKeyword ?keyword .
            ?otherMovie pgo:hasKeyword ?keyword ;
                        pgo:title ?otherTitle ;
                        pgo:movieLensId ?otherMovieId .
            ?keyword rdfs:label ?keywordName .
            FILTER(?otherMovie != ?movie)
        }
        GROUP BY ?otherMovie ?otherTitle ?otherMovieId
        ORDER BY DESC(?sharedKeywordCount) ?otherTitle
        LIMIT %d
        """ % limit
        return self.query(q, init_bindings={"movie": movie_uri})

    def find_movies_in_same_collection(
        self,
        movie: Union[int, URIRef],
    ) -> List[Dict[str, Any]]:
        """Find other movies in the same movie franchise / collection.

        Args:
            movie: MovieLens movie ID or Movie URIRef.

        Returns:
            List of dictionaries with movie_uri, movie_id, title, and collection_name.
        """
        movie_uri = self._resolve_movie(movie)
        if not movie_uri:
            return []

        q = """
        SELECT DISTINCT ?otherMovie ?otherTitle ?otherMovieId ?collectionName WHERE {
            ?movie pgo:inCollection ?collection .
            ?otherMovie pgo:inCollection ?collection ;
                        pgo:title ?otherTitle ;
                        pgo:movieLensId ?otherMovieId .
            ?collection rdfs:label ?collectionName .
            FILTER(?otherMovie != ?movie)
        }
        ORDER BY ?otherTitle
        """
        return self.query(q, init_bindings={"movie": movie_uri})

    def find_paths_between_movies(
        self,
        movie_a: Union[int, URIRef],
        movie_b: Union[int, URIRef],
    ) -> List[Dict[str, Any]]:
        """Find all 2-hop shared metadata connections between two movies in KG-M.

        Movie A -[relA]-> SharedEntity <-[relB]- Movie B

        Grounds explainable recommendation paths across shared director, actors,
        genres, keywords, franchise collection, and release decade.

        Args:
            movie_a: First movie ID or URIRef.
            movie_b: Second movie ID or URIRef.

        Returns:
            List of paths with predicateA, predicateB, entity_type, and entity_label.
        """
        uri_a = self._resolve_movie(movie_a)
        uri_b = self._resolve_movie(movie_b)
        if not uri_a or not uri_b:
            return []

        q = """
        SELECT DISTINCT ?relA ?entity ?entityType ?entityLabel ?relB WHERE {
            ?movieA ?relA ?entity .
            ?movieB ?relB ?entity .
            ?entity rdf:type ?entityType ;
                    rdfs:label ?entityLabel .
            FILTER(?relA IN (pgo:hasGenre, pgo:directedBy, pgo:starsActor, pgo:hasKeyword, pgo:inCollection, pgo:releaseDecade))
            FILTER(?relB IN (pgo:hasGenre, pgo:directedBy, pgo:starsActor, pgo:hasKeyword, pgo:inCollection, pgo:releaseDecade))
        }
        ORDER BY ?relA ?entityLabel
        """
        raw_results = self.query(q, init_bindings={"movieA": uri_a, "movieB": uri_b})
        paths = []
        for r in raw_results:
            rel_a_name = r["relA"].replace(ONTOLOGY_URI, "")
            rel_b_name = r["relB"].replace(ONTOLOGY_URI, "")
            type_name = r["entityType"].replace(ONTOLOGY_URI, "")
            paths.append({
                "predicate_a": rel_a_name,
                "predicate_b": rel_b_name,
                "entity_uri": r["entity"],
                "entity_type": type_name,
                "entity_label": r["entityLabel"],
            })
        return paths
