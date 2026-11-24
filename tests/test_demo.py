"""Comprehensive unit and integration tests for Phase 7 Streamlit Demo non-UI logic."""

import numpy as np
import pandas as pd
import pytest

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
from src.demo.recommender import ColdStartRecommender
from src.demo.research_data import (
    get_system_architecture_summary,
    load_explanation_summary_data,
    load_final_test_results_data,
    load_validation_ablation_data,
)
from src.features.constants import KGL_CATEGORIES, KGM_CATEGORIES, PLUTCHIK_EMOTIONS


def test_demo_artifacts_exist():
    """Verify that all required frozen artifacts exist on disk."""
    ok, missing = validate_demo_artifacts()
    assert ok, f"Missing required artifacts: {missing}"


def test_demo_presets_validity():
    """Verify that every title in every demo preset exists in the catalog."""
    title_to_idx, _, _ = load_movie_lookups()
    for preset_name, data in DEMO_PRESETS.items():
        assert len(data["titles"]) >= 3, f"{preset_name} must have at least 3 movies"
        for t in data["titles"]:
            assert t in title_to_idx, f"Preset movie '{t}' not found in catalog!"


def test_ease_matrix_properties():
    """Verify shape, float32 dtype, zero diagonal, and finite values for frozen EASE matrix."""
    B = load_ease_item_matrix()
    assert B.shape == (3883, 3883)
    assert B.dtype == np.float32
    assert np.all(np.isfinite(B))
    assert np.max(np.abs(np.diag(B))) == 0.0


def test_cold_start_profile_equal_weighting():
    """Verify equal weighting (1/N) and structure of cold-start user profiles."""
    movie_index = load_feature_movie_indexer()
    tfidf_encoder, sbert_encoder = load_text_encoders()
    movie_nrc_df = load_nrc_movie_vectors()

    # Toy Story (0), The Matrix (2502), Titanic (1672)
    selected = [0, 2502, 1672]
    u_idx, text_idx, nrc_idx = build_cold_start_profiles(
        selected_movie_indices=selected,
        movie_index=movie_index,
        tfidf_encoder=tfidf_encoder,
        sbert_encoder=sbert_encoder,
        movie_nrc_df=movie_nrc_df,
        demo_user_idx=0,
    )

    prof = u_idx.get(0)
    assert prof is not None
    assert prof.positive_train_count == 3
    assert prof.positive_train_movies == set(selected)

    # Check distributions sum to 1.0 (or 0.0 if empty)
    for cat in KGM_CATEGORIES:
        if prof.kgm_profiles[cat]:
            assert np.isclose(sum(prof.kgm_profiles[cat].values()), 1.0)
    for cat in KGL_CATEGORIES:
        if prof.kgl_profiles[cat]:
            assert np.isclose(sum(prof.kgl_profiles[cat].values()), 1.0)

    # Check Qwen emotion profile
    assert prof.emotion_profile_available
    assert prof.emotion_profile is not None
    assert prof.emotion_profile.shape == (8,)
    assert np.all(prof.emotion_profile >= 0.0)
    assert np.all(prof.emotion_profile <= 1.0)

    # Check Text profile
    t_prof = text_idx.get(0)
    assert t_prof is not None
    assert t_prof.sbert_available
    assert t_prof.sbert_profile is not None
    assert np.isclose(np.linalg.norm(t_prof.sbert_profile), 1.0, atol=1e-4)

    # Check NRC profile
    n_prof = nrc_idx.get(0)
    assert n_prof is not None
    assert n_prof.nrc_available
    assert np.isclose(n_prof.nrc_profile.sum(), 1.0, atol=1e-4)


def test_cold_start_recommender_e2e():
    """Verify candidate scoring, selection exclusion, 89-feature schema, and LightGBM inference."""
    ease_matrix = load_ease_item_matrix()
    reranker_model = load_final_reranker()
    movie_index = load_feature_movie_indexer()
    tfidf_encoder, sbert_encoder = load_text_encoders()
    movie_nrc_df = load_nrc_movie_vectors()
    metadata_df = load_movie_metadata()

    recommender = ColdStartRecommender(
        ease_matrix=ease_matrix,
        reranker_model=reranker_model,
        movie_index=movie_index,
        tfidf_encoder=tfidf_encoder,
        sbert_encoder=sbert_encoder,
        movie_nrc_df=movie_nrc_df,
        movie_metadata_df=metadata_df,
    )

    selected = [0, 2502, 1672]
    res = recommender.recommend(selected_movie_indices=selected, k_candidates=100, k_top_display=10)

    # Check exclusion
    cand_movies = set(res.full_candidates["movie_idx"])
    assert len(cand_movies & set(selected)) == 0, "Selected movies must not appear in candidates!"

    # Check counts
    assert len(res.full_candidates) == 100
    assert len(res.top_recommendations) == 10
    assert len(res.feature_matrix) == 100
    assert res.feature_matrix.shape[1] == 89
    assert list(res.feature_matrix.columns) == reranker_model.booster_.feature_name()

    # Check finite predictions and deterministic rank ordering
    scores = res.full_candidates["reranker_score"].to_numpy()
    assert np.all(np.isfinite(scores))
    assert np.all(np.diff(scores) <= 0.0), "Candidates must be strictly sorted descending by reranker_score!"

    # Check latency measured
    assert res.recommendation_latency_s > 0.0
    assert res.recommendation_latency_s < 5.0, "Recommendation latency should be fast (< 5s)!"


def test_cold_start_explanation_grounding_and_validation():
    """Verify that explanations for cold-start recommendations cite only selected movies and valid KG paths."""
    ease_matrix = load_ease_item_matrix()
    reranker_model = load_final_reranker()
    movie_index = load_feature_movie_indexer()
    knowledge_index = load_explanation_knowledge_index()
    tfidf_encoder, sbert_encoder = load_text_encoders()
    movie_nrc_df = load_nrc_movie_vectors()
    metadata_df = load_movie_metadata()

    recommender = ColdStartRecommender(
        ease_matrix=ease_matrix,
        reranker_model=reranker_model,
        movie_index=movie_index,
        tfidf_encoder=tfidf_encoder,
        sbert_encoder=sbert_encoder,
        movie_nrc_df=movie_nrc_df,
        movie_metadata_df=metadata_df,
    )
    adapter = ColdStartExplanationAdapter(
        knowledge_index=knowledge_index,
        reranker_model=reranker_model,
    )

    selected = [0, 2502, 1672]
    res = recommender.recommend(selected_movie_indices=selected, k_candidates=100, k_top_display=5)

    for i, (_, row) in enumerate(res.top_recommendations.iterrows()):
        feat_row = res.feature_matrix.iloc[i]
        exp = adapter.explain_recommendation(
            candidate_row=row,
            feature_row=feat_row,
            selected_movie_indices=selected,
        )

        assert exp.movie_idx == int(row["movie_idx"])
        assert exp.plotgraph_rank == int(row["plotgraph_rank"])
        assert exp.ease_rank == int(row["ease_rank"])

        # Grounding assertion: all support movies must come from selected movies
        for sm in exp.historical_support_movies:
            assert sm.movie_idx in set(selected), f"Support movie {sm.movie_idx} not in selected set!"

        # Grounding assertion: all paths must connect to one of the selected movies
        for p in exp.kgm_paths:
            assert p.source_movie_idx in set(selected)
        for p in exp.kgl_paths:
            assert p.source_movie_idx in set(selected)

        # Cold-start wording verification
        assert "viewing history" not in exp.explanation_text.lower()
        assert "in your positive history" not in exp.explanation_text.lower()
        for comp in exp.explanation_components:
            assert "viewing history" not in comp.lower()
            assert "in your positive history" not in comp.lower()

        # Graphviz rendering check
        dot = adapter.build_graphviz_diagram(exp)
        assert dot is not None
        assert str(exp.movie_idx) in dot.source or "RECOMMENDATION" in dot.source


def test_research_data_integrity():
    """Verify research metrics data loaded for Tab 3."""
    arch = get_system_architecture_summary()
    assert "candidate_generator" in arch
    assert "knowledge_graphs" in arch
    assert arch["candidate_generator"]["regularization"] == 500.0

    val = load_validation_ablation_data()
    assert "FULL_NRC" in val
    assert "EASE" in val
    assert "EASE_BASELINE" not in val, "EASE_BASELINE must be normalized to EASE without duplication"
    assert len(val) == 12, f"Expected exactly 12 models (EASE + 11 ablations), found {len(val)}"
    assert val["FULL_NRC"]["ndcg_at_10"] >= val["EASE"]["ndcg_at_10"]

    test = load_final_test_results_data()
    assert "end_to_end_results" in test or "end_to_end_population" in test
    e2e = test.get("end_to_end_results", test.get("end_to_end_population", {}))
    assert "ease_ndcg@10" in e2e
    assert "plotgraph_ndcg@10" in e2e
    assert e2e["plotgraph_ndcg@10"] >= e2e["ease_ndcg@10"]
    assert "ease_hr@10" in e2e
    assert e2e["plotgraph_hr@10"] >= e2e["ease_hr@10"]

    exp = load_explanation_summary_data()
    assert exp["total_explanations"] == 60400
    assert exp["validation_metrics"]["validation_pass_rate"] == 1.0
