"""Artifact loader and cache management for the PlotGraph Streamlit Demo."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import joblib
import numpy as np
import pandas as pd

try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except (ImportError, Exception):
    def cache_resource(func):
        return func
    def cache_data(func):
        return func

from src.explanations.path_finder import MovieKnowledgeIndex
from src.features.movie_indexer import MovieEntityIndex
from src.features.text_encoder import SBERTMovieEncoder, TFIDFMovieEncoder


# Canonical file paths for Phase 7 Demo
METADATA_PATH = Path("data/processed/ml-1m/metadata/movies_with_tmdb.parquet")
SEMANTIC_PATH = Path("data/processed/ml-1m/kg_l/full_extraction/semantic_validated.parquet")
EMOTION_PATH = Path("data/processed/ml-1m/kg_l/full_extraction/emotion_vectors.parquet")
EMOTION_EVIDENCE_PATH = Path("data/processed/ml-1m/kg_l/full_extraction/emotion_validated_evidence.jsonl")
NRC_VECTORS_PATH = Path("data/processed/ml-1m/features/nrc/movie_nrc_vectors.parquet")
TFIDF_DIR = Path("data/processed/ml-1m/features/tfidf")
SBERT_DIR = Path("data/processed/ml-1m/features/text_embeddings")
EASE_MATRIX_PATH = Path("data/processed/ml-1m/final_eval/ease_item_matrix.npy")
RERANKER_PATH = Path("data/processed/ml-1m/final_eval/final_reranker.joblib")

VALIDATION_RESULTS_PATH = Path("data/processed/ml-1m/reranker/ablation_results.json")
FINAL_TEST_RESULTS_PATH = Path("data/processed/ml-1m/final_eval/results/final_test_results.json")
BOOTSTRAP_TEST_RESULTS_PATH = Path("data/processed/ml-1m/final_eval/results/bootstrap_test_results.json")
EXPLANATION_SUMMARY_PATH = Path("data/processed/ml-1m/explanations/explanation_summary.json")

DEMO_PRESETS: Dict[str, Dict[str, Any]] = {
    "Preset A — Mainstream Mixed": {
        "description": "Popular family, sci-fi, and romance blockbusters from the late 1990s.",
        "titles": [
            "Toy Story (1995)",
            "Matrix, The (1999)",
            "Titanic (1997)",
        ],
    },
    "Preset B — Sci-Fi / Adventure": {
        "description": "Action, time-travel, and futuristic adventure classics.",
        "titles": [
            "Star Wars: Episode IV - A New Hope (1977)",
            "Jurassic Park (1993)",
            "Terminator 2: Judgment Day (1991)",
            "Back to the Future (1985)",
        ],
    },
    "Preset C — Crime / Drama": {
        "description": "Acclaimed character-driven dramas and crime masterpieces.",
        "titles": [
            "Godfather, The (1972)",
            "Pulp Fiction (1994)",
            "GoodFellas (1990)",
            "Shawshank Redemption, The (1994)",
        ],
    },
}


def validate_demo_artifacts() -> Tuple[bool, List[str]]:
    """Verify that all required frozen artifacts exist on disk before launching demo."""
    required_paths = [
        METADATA_PATH,
        SEMANTIC_PATH,
        EMOTION_PATH,
        NRC_VECTORS_PATH,
        TFIDF_DIR / "movie_tfidf_matrix.npz",
        TFIDF_DIR / "tfidf_vectorizer.joblib",
        SBERT_DIR / "movie_sbert_embeddings.npy",
        EASE_MATRIX_PATH,
        RERANKER_PATH,
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    return len(missing) == 0, missing


@cache_data
def load_movie_metadata() -> pd.DataFrame:
    """Load catalog metadata table."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
    return pd.read_parquet(METADATA_PATH)


@cache_data
def load_movie_lookups() -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """Build movie title lookups for searchable Streamlit multiselect."""
    df = load_movie_metadata()
    title_to_idx: Dict[str, int] = {}
    idx_to_title: Dict[int, str] = {}
    titles_list: List[str] = []

    for _, row in df.iterrows():
        midx = int(row["movie_idx"])
        title = str(row["ml_title"])
        title_to_idx[title] = midx
        idx_to_title[midx] = title
        titles_list.append(title)

    titles_list.sort()
    return title_to_idx, idx_to_title, titles_list


@cache_resource
def load_ease_item_matrix() -> np.ndarray:
    """Load the frozen EASE item-item weight matrix B."""
    if not EASE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Frozen EASE matrix not found: {EASE_MATRIX_PATH}. "
            "Run 'python scripts/prepare_demo_artifacts.py' to generate it."
        )
    matrix = np.load(EASE_MATRIX_PATH)
    assert matrix.shape == (3883, 3883), f"Invalid EASE matrix shape: {matrix.shape}"
    return matrix


@cache_resource
def load_final_reranker() -> Any:
    """Load the frozen LightGBM reranker model."""
    if not RERANKER_PATH.exists():
        raise FileNotFoundError(f"Final reranker model not found: {RERANKER_PATH}")
    return joblib.load(RERANKER_PATH)


@cache_resource
def load_feature_movie_indexer() -> MovieEntityIndex:
    """Load MovieEntityIndex used for building candidate features."""
    meta = load_movie_metadata()
    sem = pd.read_parquet(SEMANTIC_PATH)
    emo = pd.read_parquet(EMOTION_PATH)
    return MovieEntityIndex.from_dataframes(metadata_df=meta, semantic_df=sem, emotion_df=emo)


@cache_resource
def load_explanation_knowledge_index() -> MovieKnowledgeIndex:
    """Load MovieKnowledgeIndex used for path extraction and explanation grounding."""
    return MovieKnowledgeIndex(
        metadata_path=METADATA_PATH,
        semantic_path=SEMANTIC_PATH,
        emotion_vectors_path=EMOTION_PATH,
        emotion_evidence_path=EMOTION_EVIDENCE_PATH if EMOTION_EVIDENCE_PATH.exists() else None,
    )


@cache_resource
def load_text_encoders() -> Tuple[TFIDFMovieEncoder, SBERTMovieEncoder]:
    """Load pre-fitted TF-IDF vectorizer and SBERT dense embeddings."""
    tfidf = TFIDFMovieEncoder.load(TFIDF_DIR)
    sbert = SBERTMovieEncoder.load(SBERT_DIR)
    return tfidf, sbert


@cache_data
def load_nrc_movie_vectors() -> pd.DataFrame:
    """Load pre-computed 8D NRC vectors."""
    if not NRC_VECTORS_PATH.exists():
        raise FileNotFoundError(f"NRC vectors not found: {NRC_VECTORS_PATH}")
    return pd.read_parquet(NRC_VECTORS_PATH)
