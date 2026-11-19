"""Data ingestion, ID mapping, splitting, and metadata ingestion modules."""

from .loader import load_ml1m_ratings, load_ml1m_movies
from .id_mapper import IDMapper
from .splitter import split_chronological_per_user
from .tmdb_loader import (
    safe_parse_literal,
    extract_release_decade,
    parse_genres_field,
    parse_collection_field,
    clean_tmdb_metadata,
    parse_credits,
    parse_keywords,
    link_movielens_tmdb,
    compute_coverage_report,
    build_unified_metadata_pipeline,
)

__all__ = [
    "load_ml1m_ratings",
    "load_ml1m_movies",
    "IDMapper",
    "split_chronological_per_user",
    "safe_parse_literal",
    "extract_release_decade",
    "parse_genres_field",
    "parse_collection_field",
    "clean_tmdb_metadata",
    "parse_credits",
    "parse_keywords",
    "link_movielens_tmdb",
    "compute_coverage_report",
    "build_unified_metadata_pipeline",
]

