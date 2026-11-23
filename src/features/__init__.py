"""PlotGraph Features Package (Phases 4A & 4B).

Leakage-safe KG-M, KG-L semantic, continuous emotion, and text-semantic (TF-IDF + SBERT)
feature engineering for candidate reranking.
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
    PHASE4A_COLUMNS,
    PHASE4A_FEATURES,
    PLUTCHIK_EMOTIONS,
    SBERT_FEATURES,
    TEXT_FEATURES,
    TFIDF_FEATURES,
    USER_EMOTION_FEATURES,
)
from .feature_builder import build_candidate_features, validate_candidate_features
from .movie_indexer import MovieEntityIndex, MovieMetadataItem
from .summary import generate_feature_summary, save_feature_summary
from .text_encoder import FAISSIndexHelper, SBERTMovieEncoder, TFIDFMovieEncoder
from .text_feature_builder import (
    build_text_candidate_features,
    validate_text_candidate_features,
)
from .text_profiler import (
    TextUserProfile,
    TextUserProfileIndex,
    build_text_user_profiles,
)
from .text_summary import generate_text_feature_summary, save_text_feature_summary
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
    "PHASE4A_COLUMNS",
    "PHASE4A_FEATURES",
    "PLUTCHIK_EMOTIONS",
    "SBERT_FEATURES",
    "TEXT_FEATURES",
    "TFIDF_FEATURES",
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
    # Feature Builder (Phase 4A)
    "build_candidate_features",
    "validate_candidate_features",
    # Text Encoders & Profiler (Phase 4B)
    "TFIDFMovieEncoder",
    "SBERTMovieEncoder",
    "FAISSIndexHelper",
    "TextUserProfile",
    "TextUserProfileIndex",
    "build_text_user_profiles",
    "build_text_candidate_features",
    "validate_text_candidate_features",
    # Summaries
    "generate_feature_summary",
    "save_feature_summary",
    "generate_text_feature_summary",
    "save_text_feature_summary",
]
