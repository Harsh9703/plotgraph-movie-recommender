"""Constants, Feature Groups, and Schemas for PlotGraph Phase 4A Feature Engineering.

Defines:
- Plutchik 8 continuous emotions.
- KG-M metadata categories.
- KG-L semantic categories.
- Named feature groups for downstream reranker ablation studies:
  * BASE_FEATURES
  * KGM_AFFINITY_FEATURES
  * KGM_OVERLAP_FEATURES
  * KGM_FEATURES
  * KGL_SEMANTIC_AFFINITY_FEATURES
  * KGL_SEMANTIC_OVERLAP_FEATURES
  * KGL_SEMANTIC_FEATURES
  * EMOTION_COMPATIBILITY_FEATURES
  * CANDIDATE_EMOTION_FEATURES
  * USER_EMOTION_FEATURES
  * EMOTION_FEATURES
  * OVERLAP_COUNT_FEATURES
  * ALL_KG_FEATURES
  * ALL_FEATURES
- Default configuration values.
"""

from typing import List, Tuple

# -----------------------------------------------------------------------------
# Ontological Taxonomy & Categories
# -----------------------------------------------------------------------------

PLUTCHIK_EMOTIONS: Tuple[str, ...] = (
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
)

KGM_CATEGORIES: Tuple[str, ...] = (
    "genre",
    "director",
    "actor",
    "keyword",
    "collection",
    "decade",
)

KGL_CATEGORIES: Tuple[str, ...] = (
    "theme",
    "setting",
    "narrative_device",
    "protagonist_type",
    "source_material",
)

# -----------------------------------------------------------------------------
# Default Parameters
# -----------------------------------------------------------------------------

DEFAULT_HALF_LIFE_DAYS: float = 365.0
DEFAULT_POSITIVE_THRESHOLD: float = 4.0
FROZEN_KG_EMOTION_EDGE_THRESHOLD: float = 0.3

# -----------------------------------------------------------------------------
# Feature Group Definitions (Ordered Lists)
# -----------------------------------------------------------------------------

IDENTIFIER_COLUMNS: List[str] = [
    "user_idx",
    "movie_idx",
]

BASE_FEATURES: List[str] = [
    "candidate_rank",
    "ease_score",
    "user_positive_train_count",
    "candidate_seen_in_positive_train",
    "candidate_has_tmdb_link",
    "candidate_has_overview",
    "candidate_has_kgm_metadata",
    "candidate_has_kgl_semantics",
    "candidate_emotion_available",
    "user_emotion_profile_available",
]

KGM_AFFINITY_FEATURES: List[str] = [
    "kgm_genre_affinity",
    "kgm_director_affinity",
    "kgm_actor_affinity",
    "kgm_keyword_affinity",
    "kgm_collection_affinity",
    "kgm_decade_affinity",
    "kgm_affinity_mean",
    "kgm_affinity_max",
    "kgm_categories_matched",
]

KGM_OVERLAP_FEATURES: List[str] = [
    "shared_genre_count",
    "shared_director_count",
    "shared_actor_count",
    "shared_keyword_count",
    "shared_collection_count",
    "shared_decade_count",
]

KGM_FEATURES: List[str] = KGM_AFFINITY_FEATURES + KGM_OVERLAP_FEATURES

KGL_SEMANTIC_AFFINITY_FEATURES: List[str] = [
    "kgl_theme_affinity",
    "kgl_setting_affinity",
    "kgl_narrative_device_affinity",
    "kgl_protagonist_type_affinity",
    "kgl_source_material_affinity",
    "kgl_semantic_affinity_mean",
    "kgl_semantic_affinity_max",
    "kgl_semantic_categories_matched",
]

KGL_SEMANTIC_OVERLAP_FEATURES: List[str] = [
    "shared_theme_count",
    "shared_setting_count",
    "shared_narrative_device_count",
    "shared_protagonist_type_count",
    "shared_source_material_count",
]

KGL_SEMANTIC_FEATURES: List[str] = (
    KGL_SEMANTIC_AFFINITY_FEATURES + KGL_SEMANTIC_OVERLAP_FEATURES
)

EMOTION_COMPATIBILITY_FEATURES: List[str] = [
    "emotion_cosine_similarity",
    "emotion_dot_product",
    "emotion_l1_distance",
    "emotion_l2_distance",
]

CANDIDATE_EMOTION_FEATURES: List[str] = [
    f"candidate_emotion_{e}" for e in PLUTCHIK_EMOTIONS
]

USER_EMOTION_FEATURES: List[str] = [
    f"user_emotion_{e}" for e in PLUTCHIK_EMOTIONS
]

EMOTION_FEATURES: List[str] = (
    EMOTION_COMPATIBILITY_FEATURES
    + CANDIDATE_EMOTION_FEATURES
    + USER_EMOTION_FEATURES
)

OVERLAP_COUNT_FEATURES: List[str] = (
    KGM_OVERLAP_FEATURES + KGL_SEMANTIC_OVERLAP_FEATURES
)

ALL_KG_FEATURES: List[str] = (
    KGM_FEATURES + KGL_SEMANTIC_FEATURES + EMOTION_FEATURES
)

# -----------------------------------------------------------------------------
# Phase 4B: Text Semantic Feature Groups
# -----------------------------------------------------------------------------

TFIDF_FEATURES: List[str] = [
    "tfidf_cosine_similarity",
    "tfidf_dot_product",
    "tfidf_candidate_available",
    "tfidf_user_profile_available",
]

SBERT_FEATURES: List[str] = [
    "sbert_cosine_similarity",
    "sbert_dot_product",
    "sbert_candidate_available",
    "sbert_user_profile_available",
]

TEXT_FEATURES: List[str] = TFIDF_FEATURES + SBERT_FEATURES

# -----------------------------------------------------------------------------
# Combined Feature Groups
# -----------------------------------------------------------------------------

PHASE4A_FEATURES: List[str] = BASE_FEATURES + ALL_KG_FEATURES
PHASE4A_COLUMNS: List[str] = IDENTIFIER_COLUMNS + PHASE4A_FEATURES

ALL_FEATURES: List[str] = BASE_FEATURES + ALL_KG_FEATURES + TEXT_FEATURES

ALL_COLUMNS: List[str] = IDENTIFIER_COLUMNS + ALL_FEATURES
