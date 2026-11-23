"""PlotGraph Features Package (Phase 4A).

Leakage-safe KG-M, KG-L, and continuous emotion feature engineering
for candidate reranking.
"""

from .constants import (
    ALL_COLUMNS,
    ALL_FEATURES,
    ALL_KG_FEATURES,
    BASE_FEATURES,
    CANDIDATE_EMOTION_FEATURES,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_POSITIVE_THRESHOLD,
    EMOTION_COMPATIBILITY_FEATURES,
    EMOTION_FEATURES,
    IDENTIFIER_COLUMNS,
    KGL_CATEGORIES,
    KGL_SEMANTIC_AFFINITY_FEATURES,
    KGL_SEMANTIC_FEATURES,
    KGL_SEMANTIC_OVERLAP_FEATURES,
    KGM_AFFINITY_FEATURES,
    KGM_CATEGORIES,
    KGM_FEATURES,
    KGM_OVERLAP_FEATURES,
    OVERLAP_COUNT_FEATURES,
    PLUTCHIK_EMOTIONS,
    USER_EMOTION_FEATURES,
)
from .feature_builder import build_candidate_features, validate_candidate_features
from .movie_indexer import MovieEntityIndex, MovieMetadataItem
from .summary import generate_feature_summary, save_feature_summary
from .user_profiler import (
    UserProfile,
    UserProfileIndex,
    build_user_profiles,
    compute_age_days,
    compute_recency_weight,
)

__all__ = [
    # Constants & Groups
    "ALL_COLUMNS",
    "ALL_FEATURES",
    "ALL_KG_FEATURES",
    "BASE_FEATURES",
    "CANDIDATE_EMOTION_FEATURES",
    "DEFAULT_HALF_LIFE_DAYS",
    "DEFAULT_POSITIVE_THRESHOLD",
    "EMOTION_COMPATIBILITY_FEATURES",
    "EMOTION_FEATURES",
    "IDENTIFIER_COLUMNS",
    "KGL_CATEGORIES",
    "KGL_SEMANTIC_AFFINITY_FEATURES",
    "KGL_SEMANTIC_FEATURES",
    "KGL_SEMANTIC_OVERLAP_FEATURES",
    "KGM_AFFINITY_FEATURES",
    "KGM_CATEGORIES",
    "KGM_FEATURES",
    "KGM_OVERLAP_FEATURES",
    "OVERLAP_COUNT_FEATURES",
    "PLUTCHIK_EMOTIONS",
    "USER_EMOTION_FEATURES",
    # Movie Indexer
    "MovieEntityIndex",
    "MovieMetadataItem",
    # User Profiler
    "UserProfile",
    "UserProfileIndex",
    "build_user_profiles",
    "compute_age_days",
    "compute_recency_weight",
    # Feature Builder
    "build_candidate_features",
    "validate_candidate_features",
    # Summary
    "generate_feature_summary",
    "save_feature_summary",
]
