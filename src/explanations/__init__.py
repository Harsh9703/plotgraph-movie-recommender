"""PlotGraph recommendation explanations package."""

from src.explanations.engine import ExplanationEngine, FEATURE_FAMILY_MAP
from src.explanations.path_finder import MovieKnowledgeIndex, RELATION_WEIGHTS
from src.explanations.schemas import (
    EmotionEvidence,
    EvidencePath,
    FeatureContribution,
    HistoricalSupportMovie,
    StructuredExplanation,
    TextEvidence,
)
from src.explanations.validator import ExplanationValidator, ValidationResult

__all__ = [
    "ExplanationEngine",
    "ExplanationValidator",
    "MovieKnowledgeIndex",
    "EvidencePath",
    "FeatureContribution",
    "EmotionEvidence",
    "TextEvidence",
    "HistoricalSupportMovie",
    "StructuredExplanation",
    "ValidationResult",
    "FEATURE_FAMILY_MAP",
    "RELATION_WEIGHTS",
]
