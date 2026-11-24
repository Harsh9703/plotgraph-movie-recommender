"""Comprehensive unit and integration tests for Phase 6 recommendation explanations."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

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
from src.explanations.validator import ExplanationValidator


class MockKnowledgeIndex(MovieKnowledgeIndex):
    """Mock index for fast, isolated unit tests."""

    def __init__(self):
        self.movie_titles = {
            0: "Toy Story (1995)",
            1: "A Bug's Life (1998)",
            2: "The Lion King (1994)",
            3: "Jurassic Park (1993)",
            4: "Unknown Movie (2000)",
        }
        self.movie_overviews = {
            0: "Woody and Buzz go on an adventure.",
            1: "An ant named Flik looks for warriors to save his colony.",
            2: "A young lion prince flees his kingdom.",
            3: "Dinosaurs break loose in an island theme park.",
            4: "",
        }
        self.movie_ids = {0: 1, 1: 2355, 2: 364, 3: 450, 4: 9999}
        self.idx_to_movie_id = dict(self.movie_ids)
        self.movie_id_to_idx = {v: k for k, v in self.movie_ids.items()}

        self.kgm_entities = {
            0: {
                "directedBy": {"John Lasseter"},
                "inCollection": {"Toy Story Collection"},
                "starsActor": {"Tom Hanks", "Tim Allen"},
                "hasGenre": {"Animation", "Comedy", "Family"},
                "hasKeyword": {"friendship", "toy"},
                "releaseDecade": {"1990s"},
            },
            1: {
                "directedBy": {"John Lasseter"},
                "inCollection": set(),
                "starsActor": {"Dave Foley"},
                "hasGenre": {"Animation", "Comedy", "Family"},
                "hasKeyword": {"insect"},
                "releaseDecade": {"1990s"},
            },
            2: {
                "directedBy": {"Roger Allers", "Rob Minkoff"},
                "inCollection": {"The Lion King Collection"},
                "starsActor": {"Matthew Broderick"},
                "hasGenre": {"Animation", "Drama"},
                "hasKeyword": {"royalty"},
                "releaseDecade": {"1990s"},
            },
            3: {
                "directedBy": {"Steven Spielberg"},
                "inCollection": {"Jurassic Park Collection"},
                "starsActor": {"Sam Neill"},
                "hasGenre": {"Action", "Sci-Fi"},
                "hasKeyword": {"dinosaur"},
                "releaseDecade": {"1990s"},
            },
            4: {
                "directedBy": set(),
                "inCollection": set(),
                "starsActor": set(),
                "hasGenre": set(),
                "hasKeyword": set(),
                "releaseDecade": set(),
            },
        }

        self.kgl_entities = {
            0: {
                "hasTheme": {"friendship", "loyalty"},
                "hasSetting": {"child's bedroom"},
                "hasNarrativeDevice": {"rivalry"},
                "hasProtagonistType": {"toys"},
                "basedOn": set(),
                "evokesEmotion": {"anger", "anticipation"},
            },
            1: {
                "hasTheme": {"loyalty", "perseverance"},
                "hasSetting": {"ant island"},
                "hasNarrativeDevice": set(),
                "hasProtagonistType": {"insects"},
                "basedOn": set(),
                "evokesEmotion": {"anticipation", "joy"},
            },
            2: {
                "hasTheme": {"responsibility", "grief"},
                "hasSetting": {"savannah"},
                "hasNarrativeDevice": {"exile"},
                "hasProtagonistType": {"royalty"},
                "basedOn": {"hamlet"},
                "evokesEmotion": {"sadness", "joy"},
            },
            3: {
                "hasTheme": {"survival", "hubris"},
                "hasSetting": {"tropical island"},
                "hasNarrativeDevice": {"chase"},
                "hasProtagonistType": {"scientists"},
                "basedOn": {"novel"},
                "evokesEmotion": {"fear", "surprise"},
            },
            4: {
                "hasTheme": set(),
                "hasSetting": set(),
                "hasNarrativeDevice": set(),
                "hasProtagonistType": set(),
                "basedOn": set(),
                "evokesEmotion": set(),
            },
        }

        self.emotion_vectors = {
            0: {"joy": 0.0, "trust": 0.0, "fear": 0.0, "surprise": 0.0, "sadness": 0.0, "disgust": 0.0, "anger": 0.4, "anticipation": 0.6},
            1: {"joy": 0.5, "trust": 0.3, "fear": 0.0, "surprise": 0.0, "sadness": 0.0, "disgust": 0.0, "anger": 0.0, "anticipation": 0.5},
            2: {"joy": 0.6, "trust": 0.4, "fear": 0.2, "surprise": 0.1, "sadness": 0.5, "disgust": 0.0, "anger": 0.0, "anticipation": 0.4},
            3: {"joy": 0.0, "trust": 0.0, "fear": 0.8, "surprise": 0.6, "sadness": 0.0, "disgust": 0.0, "anger": 0.2, "anticipation": 0.7},
            4: {"joy": 0.0, "trust": 0.0, "fear": 0.0, "surprise": 0.0, "sadness": 0.0, "disgust": 0.0, "anger": 0.0, "anticipation": 0.0},
        }
        self.prominent_emotions = {
            0: {"anger", "anticipation"},
            1: {"joy", "anticipation"},
            2: {"joy", "sadness", "anticipation"},
            3: {"fear", "surprise", "anticipation"},
            4: set(),
        }
        self.emotion_evidence = {}


def test_feature_family_mapping_completeness():
    """Verify that all 89 LightGBM model features are assigned to a valid feature family."""
    from src.features.constants import (
        BASE_FEATURES,
        KGM_FEATURES,
        KGL_SEMANTIC_FEATURES,
        QWEN_EMOTION_FEATURES,
        NRC_EMOTION_FEATURES,
        TEXT_FEATURES,
    )
    all_features = (
        BASE_FEATURES
        + KGM_FEATURES
        + KGL_SEMANTIC_FEATURES
        + QWEN_EMOTION_FEATURES
        + NRC_EMOTION_FEATURES
        + TEXT_FEATURES
    )
    assert len(all_features) == 89
    for feat in all_features:
        assert feat in FEATURE_FAMILY_MAP
        assert FEATURE_FAMILY_MAP[feat] in {"BASE", "KGM", "KGL_SEMANTIC", "QWEN_EMOTION", "NRC_EMOTION", "TEXT"}


def test_kgm_and_kgl_path_extraction():
    """Verify exact path extraction between two movies in the knowledge graph."""
    mock_idx = MockKnowledgeIndex()
    # Path between Toy Story (0) and A Bug's Life (1)
    kgm_paths, kgl_paths = mock_idx.find_shared_paths(source_idx=0, target_idx=1)

    # They share director John Lasseter, genres Animation, Comedy, Family, and decade 1990s
    kgm_rel_entities = {(p.relation_type, p.target_entity_value) for p in kgm_paths}
    assert ("directedBy", "John Lasseter") in kgm_rel_entities
    assert ("hasGenre", "Animation") in kgm_rel_entities
    assert ("hasGenre", "Comedy") in kgm_rel_entities
    assert ("hasGenre", "Family") in kgm_rel_entities
    assert ("releaseDecade", "1990s") in kgm_rel_entities

    # They share theme 'loyalty' and prominent emotion 'anticipation'
    kgl_rel_entities = {(p.relation_type, p.target_entity_value) for p in kgl_paths}
    assert ("hasTheme", "loyalty") in kgl_rel_entities
    assert ("evokesEmotion", "anticipation") in kgl_rel_entities


def test_historical_support_ranking():
    """Verify that historical movies with specific overlapping entities rank higher than generic matches."""
    mock_idx = MockKnowledgeIndex()
    # User liked Toy Story (0) and Jurassic Park (3)
    user_history = [(0, 5.0, 100000), (3, 4.0, 90000)]
    # Target candidate is A Bug's Life (1)
    support_movies = mock_idx.find_top_support_movies(candidate_idx=1, user_positive_history=user_history)

    assert len(support_movies) == 2
    # Toy Story (same director, themes, genres) must rank #1 over Jurassic Park (only decade 1990s)
    assert support_movies[0].movie_idx == 0
    assert support_movies[0].title == "Toy Story (1995)"
    assert support_movies[0].shared_entity_count > support_movies[1].shared_entity_count


def test_explanation_engine_rendering_hybrid():
    """Verify deterministic template rendering for a hybrid explanation."""
    mock_idx = MockKnowledgeIndex()
    engine = ExplanationEngine(knowledge_index=mock_idx, min_emotion_threshold=0.2)

    user_history = [(0, 5.0, 100000)]
    feature_row = pd.Series({
        "candidate_emotion_available": 1,
        "user_emotion_profile_available": 1,
        "emotion_cosine_similarity": 0.85,
        "emotion_l2_distance": 0.25,
        "candidate_emotion_anticipation": 0.5,
        "user_emotion_anticipation": 0.6,
        "sbert_cosine_similarity": 0.65,
        "tfidf_cosine_similarity": 0.40,
        "ease_score": 0.95,
        "candidate_rank": 5,
    })

    names = ["kgm_director_affinity", "kgl_theme_affinity", "emotion_cosine_similarity"]
    vals = np.array([1.0, 1.0, 0.85])
    contribs = np.array([0.05, 0.04, 0.03])

    exp = engine.generate_explanation(
        user_idx=10,
        movie_idx=1,  # A Bug's Life
        ease_rank=5,
        plotgraph_rank=2,
        reranker_score=0.15,
        feature_row=feature_row,
        feature_names=names,
        feature_values=vals,
        contrib_values=contribs,
        user_positive_history=user_history,
    )

    assert exp.movie_idx == 1
    assert exp.plotgraph_rank == 2
    assert exp.rank_movement == 3
    assert exp.explanation_type == "hybrid"
    assert "Directed by John Lasseter" in exp.explanation_text
    assert "Shares narrative theme 'loyalty'" in exp.explanation_text
    assert "Matches the emotional profile" in exp.explanation_text


def test_explanation_fallback_behavior():
    """Verify graceful fallback to collaborative filtering when no KG entities or emotions match."""
    mock_idx = MockKnowledgeIndex()
    engine = ExplanationEngine(knowledge_index=mock_idx)

    # Empty history or unlinked movie
    user_history = [(0, 5.0, 100000)]
    feature_row = pd.Series({
        "candidate_emotion_available": 0,
        "user_emotion_profile_available": 0,
        "emotion_cosine_similarity": 0.0,
        "emotion_l2_distance": 1.0,
        "sbert_cosine_similarity": 0.1,
        "tfidf_cosine_similarity": 0.05,
        "ease_score": 0.80,
        "candidate_rank": 8,
    })

    names = ["ease_score", "candidate_rank"]
    vals = np.array([0.80, 8.0])
    contribs = np.array([0.06, 0.02])

    exp = engine.generate_explanation(
        user_idx=20,
        movie_idx=4,  # Unknown Movie with zero KG entities
        ease_rank=8,
        plotgraph_rank=8,
        reranker_score=0.08,
        feature_row=feature_row,
        feature_names=names,
        feature_values=vals,
        contrib_values=contribs,
        user_positive_history=user_history,
    )

    assert exp.explanation_type == "collaborative-fallback"
    assert "collaborative viewing patterns" in exp.explanation_text
    assert "ranked #8" in exp.explanation_text


def test_validator_rejects_fabricated_support_movie():
    """Validator must reject any explanation citing a movie not in the user's positive history."""
    mock_idx = MockKnowledgeIndex()
    validator = ExplanationValidator(knowledge_index=mock_idx)

    fake_support = [
        HistoricalSupportMovie(
            movie_idx=999,  # Not in user history
            title="Fake Movie",
            user_rating=5.0,
            timestamp=10000,
            shared_entity_count=1,
            shared_entity_summary=["director: Someone"],
        )
    ]

    exp = StructuredExplanation(
        user_idx=1,
        movie_idx=0,
        title="Toy Story (1995)",
        ease_rank=1,
        plotgraph_rank=1,
        reranker_score=0.2,
        rank_movement=0,
        explanation_type="metadata-driven",
        explanation_text="Some explanation",
        historical_support_movies=fake_support,
    )

    user_history_set = {0, 1}  # 999 is NOT here
    res = validator.validate(exp, user_pos_history_set=user_history_set)
    assert not res.is_valid
    assert any("Fabricated support movie" in e for e in res.errors)


def test_validator_rejects_test_movie_leakage():
    """Validator must immediately fail if any support movie belongs to the TEST set."""
    mock_idx = MockKnowledgeIndex()
    validator = ExplanationValidator(knowledge_index=mock_idx)

    support = [
        HistoricalSupportMovie(
            movie_idx=2,
            title="The Lion King (1994)",
            user_rating=5.0,
            timestamp=10000,
            shared_entity_count=1,
        )
    ]

    exp = StructuredExplanation(
        user_idx=1,
        movie_idx=0,
        title="Toy Story (1995)",
        ease_rank=1,
        plotgraph_rank=1,
        reranker_score=0.2,
        rank_movement=0,
        explanation_type="metadata-driven",
        explanation_text="Some explanation",
        historical_support_movies=support,
    )

    user_history_set = {2}
    test_set = {2}  # Movie 2 is in TEST!
    res = validator.validate(exp, user_pos_history_set=user_history_set, test_movies_set=test_set)
    assert not res.is_valid
    assert any("TEST leakage violation" in e for e in res.errors)


def test_validator_rejects_fabricated_kg_entity():
    """Validator must reject any path citing an entity that doesn't exist for the movie."""
    mock_idx = MockKnowledgeIndex()
    validator = ExplanationValidator(knowledge_index=mock_idx)

    fake_path = [
        EvidencePath(
            source_movie_idx=0,
            source_movie_title="Toy Story (1995)",
            relation_type="directedBy",
            target_entity_type="director",
            target_entity_value="Quentin Tarantino",  # False claim: Toy Story was NOT directed by Tarantino!
            kg_source="KG-M",
        )
    ]

    exp = StructuredExplanation(
        user_idx=1,
        movie_idx=1,  # A Bug's Life
        title="A Bug's Life (1998)",
        ease_rank=1,
        plotgraph_rank=1,
        reranker_score=0.2,
        rank_movement=0,
        explanation_type="metadata-driven",
        explanation_text="Some explanation",
        kgm_paths=fake_path,
        historical_support_movies=[],
    )

    res = validator.validate(exp, user_pos_history_set=set())
    assert not res.is_valid
    assert any("KG-M entity 'Quentin Tarantino' not found" in e for e in res.errors)


def test_real_artifact_integration():
    """Integration test loading real frozen Phase 5 artifacts and checking knowledge index construction."""
    metadata_path = Path("data/processed/ml-1m/metadata/movies_with_tmdb.parquet")
    semantic_path = Path("data/processed/ml-1m/kg_l/full_extraction/semantic_validated.parquet")
    emotion_path = Path("data/processed/ml-1m/kg_l/full_extraction/emotion_vectors.parquet")

    if not (metadata_path.exists() and semantic_path.exists() and emotion_path.exists()):
        pytest.skip("Real data artifacts not available.")

    index = MovieKnowledgeIndex(
        metadata_path=metadata_path,
        semantic_path=semantic_path,
        emotion_vectors_path=emotion_path,
    )

    # Check that Toy Story (movie_idx 0) is indexed with expected properties
    assert 0 in index.movie_titles
    assert "Toy Story" in index.movie_titles[0]
    assert "John Lasseter" in index.kgm_entities[0]["directedBy"]
    assert "Animation" in index.kgm_entities[0]["hasGenre"]
    assert "friendship" in index.kgl_entities[0]["hasTheme"]
    assert index.emotion_vectors[0]["anger"] == 0.4
    assert index.emotion_vectors[0]["anticipation"] == 0.6


def test_batch_explanation_generation_and_validation():
    """Verify batch explanation generation and 100% pass rate in automated validation."""
    mock_idx = MockKnowledgeIndex()
    engine = ExplanationEngine(knowledge_index=mock_idx)
    validator = ExplanationValidator(knowledge_index=mock_idx)

    user_history = [(0, 5.0, 100000), (2, 4.0, 95000)]
    user_pos_set = {0, 2}
    test_set = {3}

    names = ["ease_score", "candidate_rank"]
    vals = np.array([0.9, 1.0])
    contribs = np.array([0.05, 0.02])

    explanations = []
    validation_results = []

    # Generate explanations for 4 candidate movies
    for midx in [0, 1, 2, 4]:
        feature_row = pd.Series({
            "candidate_emotion_available": 1 if midx in [0, 1, 2] else 0,
            "user_emotion_profile_available": 1,
            "emotion_cosine_similarity": 0.8,
            "sbert_cosine_similarity": 0.5,
            "tfidf_cosine_similarity": 0.3,
            "ease_score": 0.9,
            "candidate_rank": 1,
        })
        exp = engine.generate_explanation(
            user_idx=1,
            movie_idx=midx,
            ease_rank=1,
            plotgraph_rank=1,
            reranker_score=0.1,
            feature_row=feature_row,
            feature_names=names,
            feature_values=vals,
            contrib_values=contribs,
            user_positive_history=user_history,
        )
        val_res = validator.validate(
            explanation=exp,
            user_pos_history_set=user_pos_set,
            test_movies_set=test_set,
        )
        explanations.append(exp)
        validation_results.append(val_res)

    assert len(explanations) == 4
    for res in validation_results:
        assert res.is_valid, f"Validation failed: {res.errors}"

