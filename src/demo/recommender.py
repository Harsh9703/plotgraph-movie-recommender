"""Cold-start candidate generation, feature assembly, and LightGBM reranking."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
import pandas as pd

from src.demo.cold_start import build_cold_start_profiles
from src.features.feature_builder import build_candidate_features
from src.features.movie_indexer import MovieEntityIndex
from src.features.nrc_feature_builder import build_nrc_candidate_features
from src.features.nrc_profiler import NRCUserProfileIndex
from src.features.text_encoder import SBERTMovieEncoder, TFIDFMovieEncoder
from src.features.text_feature_builder import build_text_candidate_features
from src.features.text_profiler import TextUserProfileIndex
from src.features.user_profiler import UserProfileIndex


@dataclass
class ColdStartRecommendationResult:
    """Holds full recommendation and candidate ranking results for a cold-start user."""

    selected_movie_indices: List[int]
    top_recommendations: pd.DataFrame  # Top K candidates with ranks, scores, titles, movements
    full_candidates: pd.DataFrame      # All 100 candidate rows
    feature_matrix: pd.DataFrame       # Exact 89 model features aligned with full_candidates
    user_profiles: UserProfileIndex
    text_profiles: TextUserProfileIndex
    nrc_profiles: NRCUserProfileIndex
    recommendation_latency_s: float
    feature_cols: List[str]


class ColdStartRecommender:
    """End-to-end recommender for new / cold-start users using frozen PlotGraph components."""

    def __init__(
        self,
        ease_matrix: np.ndarray,
        reranker_model: Any,
        movie_index: MovieEntityIndex,
        tfidf_encoder: TFIDFMovieEncoder,
        sbert_encoder: SBERTMovieEncoder,
        movie_nrc_df: pd.DataFrame,
        movie_metadata_df: pd.DataFrame,
    ):
        self.ease_matrix = ease_matrix
        self.reranker_model = reranker_model
        self.movie_index = movie_index
        self.tfidf_encoder = tfidf_encoder
        self.sbert_encoder = sbert_encoder
        self.movie_nrc_df = movie_nrc_df
        self.metadata_df = movie_metadata_df

        self.feature_cols = self.reranker_model.booster_.feature_name()
        assert len(self.feature_cols) == 89, f"Expected 89 features, got {len(self.feature_cols)}"

        # Precompute movie_idx to title map
        self.idx_to_title = dict(zip(self.metadata_df["movie_idx"], self.metadata_df["ml_title"]))

    def recommend(
        self,
        selected_movie_indices: List[int],
        k_candidates: int = 100,
        k_top_display: int = 10,
        demo_user_idx: int = 0,
    ) -> ColdStartRecommendationResult:
        """Run candidate generation, feature assembly, and reranking for a cold-start user.

        Latency is measured strictly for the interactive computation without
        including static artifact loading.
        """
        if len(selected_movie_indices) < 1:
            raise ValueError("Must provide at least one selected movie.")

        t_start = time.perf_counter()

        # 1. EASE candidate scoring from item-item matrix B
        selected_set = set(selected_movie_indices)
        scores = self.ease_matrix[selected_movie_indices, :].sum(axis=0).astype(np.float32)

        # Exclude selected movies strictly
        scores[list(selected_set)] = -np.inf

        # Rank candidates
        top_cand_indices = np.argsort(-scores, kind="stable")[:k_candidates]

        # Verify no selected items leaked into candidates
        cand_set = set(top_cand_indices)
        assert len(cand_set & selected_set) == 0, "Selected items must be excluded from candidates!"

        cand_df = pd.DataFrame({
            "user_idx": [demo_user_idx] * len(top_cand_indices),
            "movie_idx": top_cand_indices,
            "rank": np.arange(1, len(top_cand_indices) + 1, dtype=np.int32),
            "score": scores[top_cand_indices],
        })

        # 2. Build cold-start preference profiles
        user_profiles, text_profiles, nrc_profiles = build_cold_start_profiles(
            selected_movie_indices=selected_movie_indices,
            movie_index=self.movie_index,
            tfidf_encoder=self.tfidf_encoder,
            sbert_encoder=self.sbert_encoder,
            movie_nrc_df=self.movie_nrc_df,
            demo_user_idx=demo_user_idx,
        )

        # 3. Construct exact frozen feature table (Phase 4A -> 4B -> 4C)
        p4a = build_candidate_features(
            candidate_df=cand_df,
            user_profiles=user_profiles,
            movie_index=self.movie_index,
            validate=False,
        )
        p4b = build_text_candidate_features(
            phase4a_df=p4a,
            text_user_profiles=text_profiles,
            tfidf_encoder=self.tfidf_encoder,
            sbert_encoder=self.sbert_encoder,
            validate=False,
        )
        p4c = build_nrc_candidate_features(
            phase4b_df=p4b,
            nrc_user_profiles=nrc_profiles,
            movie_nrc_df=self.movie_nrc_df,
            validate=False,
        )

        # Explicit schema assertion
        missing_cols = set(self.feature_cols) - set(p4c.columns)
        if missing_cols:
            raise KeyError(f"Feature schema mismatch! Missing columns: {missing_cols}")

        X_cand = p4c[self.feature_cols]

        # 4. Frozen LightGBM Reranker inference
        reranker_preds = self.reranker_model.predict(X_cand)
        cand_df["reranker_score"] = reranker_preds.astype(np.float64)

        # 5. Deterministic sorting and rank assignment
        cand_df = cand_df.sort_values(by="reranker_score", ascending=False, kind="stable")
        cand_df["plotgraph_rank"] = np.arange(1, len(cand_df) + 1, dtype=np.int32)
        cand_df["ease_rank"] = cand_df["rank"]
        cand_df["rank_movement"] = cand_df["ease_rank"] - cand_df["plotgraph_rank"]
        cand_df["title"] = cand_df["movie_idx"].map(lambda midx: self.idx_to_title.get(midx, f"Movie {midx}"))

        # Re-index feature DataFrame to match candidate sorting
        X_cand_sorted = X_cand.iloc[cand_df.index].reset_index(drop=True)
        cand_df = cand_df.reset_index(drop=True)

        latency_s = time.perf_counter() - t_start

        return ColdStartRecommendationResult(
            selected_movie_indices=selected_movie_indices,
            top_recommendations=cand_df.head(k_top_display).copy(),
            full_candidates=cand_df,
            feature_matrix=X_cand_sorted,
            user_profiles=user_profiles,
            text_profiles=text_profiles,
            nrc_profiles=nrc_profiles,
            recommendation_latency_s=latency_s,
            feature_cols=self.feature_cols,
        )
