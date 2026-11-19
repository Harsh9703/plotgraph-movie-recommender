"""RDF Namespaces, ontology terms, and deterministic URI canonicalization for KG-M."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Optional

from rdflib import Graph, Namespace, RDF, RDFS, URIRef, XSD

# Primary PlotGraph Namespaces
ONTOLOGY_URI = "https://plotgraph.org/ontology/"
RESOURCE_URI = "https://plotgraph.org/resource/"

PGO = Namespace(ONTOLOGY_URI)
PGR = Namespace(RESOURCE_URI)

# Standard namespace dictionary for Turtle prefixes and SPARQL query headers
STANDARD_PREFIXES = {
    "pgo": PGO,
    "pgr": PGR,
    "rdf": RDF,
    "rdfs": RDFS,
    "xsd": XSD,
}

# Ontology Classes
CLASS_MOVIE = PGO.Movie
CLASS_GENRE = PGO.Genre
CLASS_PERSON = PGO.Person
CLASS_KEYWORD = PGO.Keyword
CLASS_COLLECTION = PGO.Collection
CLASS_RELEASE_DECADE = PGO.ReleaseDecade

# Ontology Predicates (Relations)
PRED_HAS_GENRE = PGO.hasGenre
PRED_DIRECTED_BY = PGO.directedBy
PRED_STARS_ACTOR = PGO.starsActor
PRED_HAS_KEYWORD = PGO.hasKeyword
PRED_IN_COLLECTION = PGO.inCollection
PRED_RELEASE_DECADE = PGO.releaseDecade

# Ontology Predicates (Literals / Attributes)
PRED_TITLE = PGO.title
PRED_MOVIELENS_ID = PGO.movieLensId
PRED_INTERNAL_INDEX = PGO.internalMovieIndex
PRED_TMDB_ID = PGO.tmdbId
PRED_TMDB_TITLE = PGO.tmdbTitle
PRED_RELEASE_DATE = PGO.releaseDate


def bind_namespaces(graph: Graph) -> None:
    """Bind PlotGraph and standard namespaces to an RDFlib Graph for clean serialization."""
    for prefix, namespace in STANDARD_PREFIXES.items():
        graph.bind(prefix, namespace)


def get_sparql_prefixes_header() -> str:
    """Return standard SPARQL PREFIX declarations for query execution."""
    return (
        f"PREFIX pgo: <{ONTOLOGY_URI}>\n"
        f"PREFIX pgr: <{RESOURCE_URI}>\n"
        f"PREFIX rdf: <{RDF}>\n"
        f"PREFIX rdfs: <{RDFS}>\n"
        f"PREFIX xsd: <{XSD}>\n\n"
    )


def canonicalize_slug(text: str) -> str:
    """Deterministically convert an entity name into a URL-safe, collision-resistant slug.

    Strategy:
    1. Unicode normalization (NFKC) to unify equivalent characters.
    2. Strip leading/trailing whitespace and lowercase for canonical matching.
    3. Replace non-alphanumeric characters (spaces, punctuation, symbols) with underscores.
    4. Collapse multiple consecutive underscores and strip leading/trailing underscores.
    5. Fall back to a deterministic SHA-256 hash slice if string contains only special characters.

    Args:
        text: Entity name or label.

    Returns:
        Clean, URL-safe slug string.
    """
    if text is None:
        raise ValueError("Cannot canonicalize None text")

    text_str = str(text).strip()
    if not text_str or text_str.lower() in ("nan", "none"):
        raise ValueError(f"Cannot canonicalize invalid/empty text: {text!r}")

    norm = unicodedata.normalize("NFKC", text_str).lower()
    # Replace any character that is not alphanumeric (unicode-aware) with underscore
    slug = re.sub(r"[^\w]+", "_", norm, flags=re.UNICODE).strip("_")

    if not slug:
        # Fallback for strings consisting solely of punctuation/symbols (e.g. '???')
        slug = f"ent_{hashlib.sha256(text_str.encode('utf-8')).hexdigest()[:12]}"

    return slug


def build_entity_uri(entity_type: str, text: str) -> URIRef:
    """Generate a deterministic URI for a metadata entity (Person, Genre, Keyword, etc.).

    Args:
        entity_type: Type prefix ('person', 'genre', 'keyword', 'collection', 'decade').
        text: Name or value of the entity.

    Returns:
        URIRef under the PGR namespace.
    """
    slug = canonicalize_slug(text)
    return PGR[f"{entity_type}_{slug}"]


def build_movie_uri(
    movie_id: int,
    tmdb_id: Optional[int] = None,
    is_tmdb_duplicate: bool = False,
) -> URIRef:
    """Generate a stable URI for a Movie resource.

    Strategy:
    - If TMDB ID is available and unique: pgr:movie_tmdb_{tmdb_id}
    - If TMDB ID is shared by multiple MovieLens movies (e.g. Kaggle mapping duplicate):
      disambiguate with pgr:movie_tmdb_{tmdb_id}_ml_{movie_id} to prevent silent entity merging.
    - If TMDB ID is missing (unmapped MovieLens movie): pgr:movie_ml_{movie_id}

    Args:
        movie_id: Original MovieLens ID (integer).
        tmdb_id: Optional TMDB ID (integer).
        is_tmdb_duplicate: True if multiple MovieLens movies share this tmdb_id.

    Returns:
        URIRef under the PGR namespace.
    """
    if tmdb_id is not None and not is_tmdb_duplicate:
        return PGR[f"movie_tmdb_{int(tmdb_id)}"]
    elif tmdb_id is not None and is_tmdb_duplicate:
        return PGR[f"movie_tmdb_{int(tmdb_id)}_ml_{int(movie_id)}"]
    else:
        return PGR[f"movie_ml_{int(movie_id)}"]
