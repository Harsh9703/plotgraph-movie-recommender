"""Data structures and schemas for grounded, auditable recommendation explanations."""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional


@dataclass
class EvidencePath:
    """Represents a grounded path connecting a historical movie to a recommended candidate via a KG entity."""
    source_movie_idx: int
    source_movie_title: str
    relation_type: str  # e.g. 'directedBy', 'hasTheme', 'inCollection', 'starsActor', etc.
    target_entity_type: str  # e.g. 'director', 'theme', 'collection', etc.
    target_entity_value: str  # e.g. 'John Lasseter', 'friendship'
    kg_source: str  # 'KG-M' or 'KG-L'
    path_str: str = ""

    def __post_init__(self):
        if not self.path_str:
            self.path_str = (
                f"{self.source_movie_title} -[{self.relation_type}]-> "
                f"'{self.target_entity_value}' <-[{self.relation_type}]- (Candidate)"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureContribution:
    """Additive feature contribution from LightGBM tree decomposition."""
    feature_name: str
    feature_value: float
    contribution: float
    feature_family: str  # 'BASE', 'KGM', 'KGL_SEMANTIC', 'QWEN_EMOTION', 'NRC_EMOTION', 'TEXT'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmotionEvidence:
    """Quantitative evidence grounding continuous Qwen emotion similarity."""
    top_aligned_emotions: List[Dict[str, Any]] = field(default_factory=list)
    candidate_prominent: List[str] = field(default_factory=list)
    cosine_similarity: float = 0.0
    euclidean_distance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TextEvidence:
    """Quantitative evidence for text-semantic compatibility."""
    sbert_cosine_similarity: float = 0.0
    tfidf_cosine_similarity: float = 0.0
    overview_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HistoricalSupportMovie:
    """Historical positive movie that provides ground-truth support for a recommendation."""
    movie_idx: int
    title: str
    user_rating: float
    timestamp: int
    shared_entity_count: int
    shared_entity_summary: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredExplanation:
    """Complete auditable explanation object containing all ground-truth evidence and rendered text."""
    user_idx: int
    movie_idx: int
    title: str
    ease_rank: int
    plotgraph_rank: int
    reranker_score: float
    rank_movement: int
    explanation_type: str  # 'metadata-driven', 'semantic-theme-driven', 'emotion-driven', 'text-semantic-driven', 'hybrid', 'collaborative-fallback'
    explanation_text: str
    top_feature_contributions: List[FeatureContribution] = field(default_factory=list)
    kgm_paths: List[EvidencePath] = field(default_factory=list)
    kgl_paths: List[EvidencePath] = field(default_factory=list)
    emotion_evidence: Optional[EmotionEvidence] = None
    text_evidence: Optional[TextEvidence] = None
    historical_support_movies: List[HistoricalSupportMovie] = field(default_factory=list)
    explanation_components: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_idx": int(self.user_idx),
            "movie_idx": int(self.movie_idx),
            "title": str(self.title),
            "ease_rank": int(self.ease_rank),
            "plotgraph_rank": int(self.plotgraph_rank),
            "reranker_score": float(self.reranker_score),
            "rank_movement": int(self.rank_movement),
            "explanation_type": str(self.explanation_type),
            "explanation_text": str(self.explanation_text),
            "top_feature_contributions": [fc.to_dict() for fc in self.top_feature_contributions],
            "kgm_paths": [p.to_dict() for p in self.kgm_paths],
            "kgl_paths": [p.to_dict() for p in self.kgl_paths],
            "emotion_evidence": self.emotion_evidence.to_dict() if self.emotion_evidence else None,
            "text_evidence": self.text_evidence.to_dict() if self.text_evidence else None,
            "historical_support_movies": [m.to_dict() for m in self.historical_support_movies],
            "explanation_components": list(self.explanation_components),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
