"""PlotGraph Interactive Streamlit Demo package for new/cold-start users."""

from src.demo.artifact_loader import (
    DEMO_PRESETS,
    load_ease_item_matrix,
    load_explanation_knowledge_index,
    load_feature_movie_indexer,
    load_final_reranker,
    load_movie_lookups,
    load_movie_metadata,
    load_nrc_movie_vectors,
    load_text_encoders,
    validate_demo_artifacts,
)
from src.demo.cold_start import build_cold_start_profiles
from src.demo.explanation_adapter import ColdStartExplanationAdapter
from src.demo.recommender import ColdStartRecommendationResult, ColdStartRecommender
from src.demo.research_data import (
    get_system_architecture_summary,
    load_explanation_summary_data,
    load_final_test_results_data,
    load_validation_ablation_data,
)
from src.demo.visualizations import (
    plot_emotion_radar_or_bar,
    plot_final_test_comparison_chart,
    plot_validation_ablations_chart,
)

__all__ = [
    "DEMO_PRESETS",
    "validate_demo_artifacts",
    "load_movie_metadata",
    "load_movie_lookups",
    "load_ease_item_matrix",
    "load_final_reranker",
    "load_feature_movie_indexer",
    "load_explanation_knowledge_index",
    "load_text_encoders",
    "load_nrc_movie_vectors",
    "build_cold_start_profiles",
    "ColdStartRecommender",
    "ColdStartRecommendationResult",
    "ColdStartExplanationAdapter",
    "get_system_architecture_summary",
    "load_validation_ablation_data",
    "load_final_test_results_data",
    "load_explanation_summary_data",
    "plot_emotion_radar_or_bar",
    "plot_validation_ablations_chart",
    "plot_final_test_comparison_chart",
]
