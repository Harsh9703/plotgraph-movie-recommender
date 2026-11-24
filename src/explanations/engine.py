"""Explanation engine for generating grounded, auditable recommendation explanations.

Decomposes frozen LightGBM tree predictions into exact additive feature contributions,
connects influential features to grounded KG-M / KG-L / emotion evidence, and renders
concise, non-hallucinatory deterministic natural-language explanations.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.explanations.path_finder import MovieKnowledgeIndex
from src.explanations.schemas import (
    EmotionEvidence,
    EvidencePath,
    FeatureContribution,
    HistoricalSupportMovie,
    StructuredExplanation,
    TextEvidence,
)


FEATURE_FAMILY_MAP = {
    # BASE
    "candidate_rank": "BASE",
    "ease_score": "BASE",
    "user_positive_train_count": "BASE",
    "candidate_seen_in_positive_train": "BASE",
    "candidate_has_tmdb_link": "BASE",
    "candidate_has_overview": "BASE",
    "candidate_has_kgm_metadata": "BASE",
    "candidate_has_kgl_semantics": "BASE",
    "candidate_emotion_available": "BASE",
    "user_emotion_profile_available": "BASE",
    # KGM
    "kgm_genre_affinity": "KGM",
    "kgm_director_affinity": "KGM",
    "kgm_actor_affinity": "KGM",
    "kgm_keyword_affinity": "KGM",
    "kgm_collection_affinity": "KGM",
    "kgm_decade_affinity": "KGM",
    "kgm_affinity_mean": "KGM",
    "kgm_affinity_max": "KGM",
    "kgm_categories_matched": "KGM",
    "shared_genre_count": "KGM",
    "shared_director_count": "KGM",
    "shared_actor_count": "KGM",
    "shared_keyword_count": "KGM",
    "shared_collection_count": "KGM",
    "shared_decade_count": "KGM",
    # KGL_SEMANTIC
    "kgl_theme_affinity": "KGL_SEMANTIC",
    "kgl_setting_affinity": "KGL_SEMANTIC",
    "kgl_narrative_device_affinity": "KGL_SEMANTIC",
    "kgl_protagonist_type_affinity": "KGL_SEMANTIC",
    "kgl_source_material_affinity": "KGL_SEMANTIC",
    "kgl_semantic_affinity_mean": "KGL_SEMANTIC",
    "kgl_semantic_affinity_max": "KGL_SEMANTIC",
    "kgl_semantic_categories_matched": "KGL_SEMANTIC",
    "shared_theme_count": "KGL_SEMANTIC",
    "shared_setting_count": "KGL_SEMANTIC",
    "shared_narrative_device_count": "KGL_SEMANTIC",
    "shared_protagonist_type_count": "KGL_SEMANTIC",
    "shared_source_material_count": "KGL_SEMANTIC",
    # QWEN_EMOTION
    "emotion_cosine_similarity": "QWEN_EMOTION",
    "emotion_dot_product": "QWEN_EMOTION",
    "emotion_l1_distance": "QWEN_EMOTION",
    "emotion_l2_distance": "QWEN_EMOTION",
    "candidate_emotion_joy": "QWEN_EMOTION",
    "candidate_emotion_trust": "QWEN_EMOTION",
    "candidate_emotion_fear": "QWEN_EMOTION",
    "candidate_emotion_surprise": "QWEN_EMOTION",
    "candidate_emotion_sadness": "QWEN_EMOTION",
    "candidate_emotion_disgust": "QWEN_EMOTION",
    "candidate_emotion_anger": "QWEN_EMOTION",
    "candidate_emotion_anticipation": "QWEN_EMOTION",
    "user_emotion_joy": "QWEN_EMOTION",
    "user_emotion_trust": "QWEN_EMOTION",
    "user_emotion_fear": "QWEN_EMOTION",
    "user_emotion_surprise": "QWEN_EMOTION",
    "user_emotion_sadness": "QWEN_EMOTION",
    "user_emotion_disgust": "QWEN_EMOTION",
    "user_emotion_anger": "QWEN_EMOTION",
    "user_emotion_anticipation": "QWEN_EMOTION",
    # TEXT
    "tfidf_cosine_similarity": "TEXT",
    "tfidf_dot_product": "TEXT",
    "tfidf_candidate_available": "TEXT",
    "tfidf_user_profile_available": "TEXT",
    "sbert_cosine_similarity": "TEXT",
    "sbert_dot_product": "TEXT",
    "sbert_candidate_available": "TEXT",
    "sbert_user_profile_available": "TEXT",
    # NRC_EMOTION
    "nrc_emotion_cosine_similarity": "NRC_EMOTION",
    "nrc_emotion_dot_product": "NRC_EMOTION",
    "nrc_emotion_l1_distance": "NRC_EMOTION",
    "nrc_emotion_l2_distance": "NRC_EMOTION",
    "candidate_nrc_joy": "NRC_EMOTION",
    "candidate_nrc_trust": "NRC_EMOTION",
    "candidate_nrc_fear": "NRC_EMOTION",
    "candidate_nrc_surprise": "NRC_EMOTION",
    "candidate_nrc_sadness": "NRC_EMOTION",
    "candidate_nrc_disgust": "NRC_EMOTION",
    "candidate_nrc_anger": "NRC_EMOTION",
    "candidate_nrc_anticipation": "NRC_EMOTION",
    "user_nrc_joy": "NRC_EMOTION",
    "user_nrc_trust": "NRC_EMOTION",
    "user_nrc_fear": "NRC_EMOTION",
    "user_nrc_surprise": "NRC_EMOTION",
    "user_nrc_sadness": "NRC_EMOTION",
    "user_nrc_disgust": "NRC_EMOTION",
    "user_nrc_anger": "NRC_EMOTION",
    "user_nrc_anticipation": "NRC_EMOTION",
    "candidate_nrc_available": "NRC_EMOTION",
    "user_nrc_profile_available": "NRC_EMOTION",
    "candidate_nrc_matched_token_count": "NRC_EMOTION",
}

PLUTCHIK_EMOTIONS = [
    "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"
]


class ExplanationEngine:
    """Generates structured, grounded explanations for PlotGraph top-k recommendations."""

    def __init__(
        self,
        knowledge_index: MovieKnowledgeIndex,
        min_emotion_threshold: float = 0.2,
    ):
        self.knowledge_index = knowledge_index
        self.min_emotion_threshold = min_emotion_threshold

    def extract_feature_contributions(
        self,
        feature_names: List[str],
        feature_values: np.ndarray,
        contrib_values: np.ndarray,
        top_n: int = 5,
    ) -> List[FeatureContribution]:
        """Extract top positive feature contributions from LightGBM prediction decomposition."""
        contributions: List[FeatureContribution] = []
        for name, val, contrib in zip(feature_names, feature_values, contrib_values):
            family = FEATURE_FAMILY_MAP.get(name, "OTHER")
            contributions.append(
                FeatureContribution(
                    feature_name=name,
                    feature_value=float(val),
                    contribution=float(contrib),
                    feature_family=family,
                )
            )

        # Sort descending by contribution
        contributions.sort(key=lambda x: x.contribution, reverse=True)
        return contributions[:top_n]

    def extract_emotion_evidence(
        self,
        candidate_idx: int,
        feature_row: pd.Series,
    ) -> Optional[EmotionEvidence]:
        """Extract quantitative emotion evidence grounding continuous Qwen profile similarity."""
        cand_avail = bool(feature_row.get("candidate_emotion_available", 0))
        user_avail = bool(feature_row.get("user_emotion_profile_available", 0))

        if not cand_avail or not user_avail:
            return None

        cosine_sim = float(feature_row.get("emotion_cosine_similarity", 0.0))
        l2_dist = float(feature_row.get("emotion_l2_distance", 0.0))

        cand_vec = self.knowledge_index.emotion_vectors.get(candidate_idx, {})
        cand_prominent = sorted(list(self.knowledge_index.prominent_emotions.get(candidate_idx, set())))

        # Identify aligned emotions where both user and candidate scores are meaningful
        aligned: List[Dict[str, Any]] = []
        for emo in PLUTCHIK_EMOTIONS:
            cand_score = float(feature_row.get(f"candidate_emotion_{emo}", 0.0))
            user_score = float(feature_row.get(f"user_emotion_{emo}", 0.0))

            if cand_score >= self.min_emotion_threshold and user_score >= self.min_emotion_threshold:
                # Alignment score = product of scores
                alignment_val = cand_score * user_score
                aligned.append({
                    "emotion": emo,
                    "candidate_score": round(cand_score, 3),
                    "user_score": round(user_score, 3),
                    "alignment": round(alignment_val, 4),
                })

        aligned.sort(key=lambda x: x["alignment"], reverse=True)

        return EmotionEvidence(
            top_aligned_emotions=aligned,
            candidate_prominent=cand_prominent,
            cosine_similarity=round(cosine_sim, 4),
            euclidean_distance=round(l2_dist, 4),
        )

    def extract_text_evidence(
        self,
        candidate_idx: int,
        feature_row: pd.Series,
    ) -> Optional[TextEvidence]:
        """Extract text-semantic similarity evidence."""
        sbert_sim = float(feature_row.get("sbert_cosine_similarity", 0.0))
        tfidf_sim = float(feature_row.get("tfidf_cosine_similarity", 0.0))
        overview = self.knowledge_index.movie_overviews.get(candidate_idx, "")
        snippet = (overview[:140] + "...") if len(overview) > 140 else overview

        return TextEvidence(
            sbert_cosine_similarity=round(sbert_sim, 4),
            tfidf_cosine_similarity=round(tfidf_sim, 4),
            overview_snippet=snippet,
        )

    def generate_explanation(
        self,
        user_idx: int,
        movie_idx: int,
        ease_rank: int,
        plotgraph_rank: int,
        reranker_score: float,
        feature_row: pd.Series,
        feature_names: List[str],
        feature_values: np.ndarray,
        contrib_values: np.ndarray,
        user_positive_history: List[Tuple[int, float, int]],  # (movie_idx, rating, timestamp)
    ) -> StructuredExplanation:
        """Construct a grounded, auditable StructuredExplanation for a single recommendation."""
        title = self.knowledge_index.movie_titles.get(movie_idx, f"Movie {movie_idx}")
        rank_movement = ease_rank - plotgraph_rank

        # 1. Feature contributions
        top_contribs = self.extract_feature_contributions(
            feature_names=feature_names,
            feature_values=feature_values,
            contrib_values=contrib_values,
            top_n=5,
        )

        # 2. Historical support movies (top 1-3)
        support_movies = self.knowledge_index.find_top_support_movies(
            candidate_idx=movie_idx,
            user_positive_history=user_positive_history,
            max_movies=3,
        )

        # 3. Grounded paths from best support movie
        all_kgm_paths: List[EvidencePath] = []
        all_kgl_paths: List[EvidencePath] = []
        for sm in support_movies:
            kp, lp = self.knowledge_index.find_shared_paths(sm.movie_idx, movie_idx)
            all_kgm_paths.extend(kp)
            all_kgl_paths.extend(lp)

        # Deduplicate paths by (relation_type, target_entity_value)
        seen_kgm: Set[Tuple[str, str]] = set()
        dedup_kgm: List[EvidencePath] = []
        for p in all_kgm_paths:
            key = (p.relation_type, p.target_entity_value)
            if key not in seen_kgm:
                seen_kgm.add(key)
                dedup_kgm.append(p)

        seen_kgl: Set[Tuple[str, str]] = set()
        dedup_kgl: List[EvidencePath] = []
        for p in all_kgl_paths:
            key = (p.relation_type, p.target_entity_value)
            if key not in seen_kgl:
                seen_kgl.add(key)
                dedup_kgl.append(p)

        # 4. Emotion & Text evidence
        emotion_ev = self.extract_emotion_evidence(movie_idx, feature_row)
        text_ev = self.extract_text_evidence(movie_idx, feature_row)

        # 5. Determine explanation type & components using deterministic priority
        explanation_type, components, text = self._render_deterministic_explanation(
            candidate_title=title,
            support_movies=support_movies,
            kgm_paths=dedup_kgm,
            kgl_paths=dedup_kgl,
            emotion_ev=emotion_ev,
            text_ev=text_ev,
            top_contribs=top_contribs,
            ease_rank=ease_rank,
            plotgraph_rank=plotgraph_rank,
            rank_movement=rank_movement,
        )

        return StructuredExplanation(
            user_idx=user_idx,
            movie_idx=movie_idx,
            title=title,
            ease_rank=ease_rank,
            plotgraph_rank=plotgraph_rank,
            reranker_score=reranker_score,
            rank_movement=rank_movement,
            explanation_type=explanation_type,
            explanation_text=text,
            top_feature_contributions=top_contribs,
            kgm_paths=dedup_kgm,
            kgl_paths=dedup_kgl,
            emotion_evidence=emotion_ev,
            text_evidence=text_ev,
            historical_support_movies=support_movies,
            explanation_components=components,
        )

    def _render_deterministic_explanation(
        self,
        candidate_title: str,
        support_movies: List[HistoricalSupportMovie],
        kgm_paths: List[EvidencePath],
        kgl_paths: List[EvidencePath],
        emotion_ev: Optional[EmotionEvidence],
        text_ev: Optional[TextEvidence],
        top_contribs: List[FeatureContribution],
        ease_rank: int,
        plotgraph_rank: int,
        rank_movement: int,
    ) -> Tuple[str, List[str], str]:
        """Render deterministic, grounded natural language without hallucination."""
        components: List[str] = []
        has_kgm = len(kgm_paths) > 0
        has_kgl = len(kgl_paths) > 0
        has_emotion = emotion_ev is not None and len(emotion_ev.top_aligned_emotions) > 0

        # Check for specific strong relation types
        directors = [p for p in kgm_paths if p.relation_type == "directedBy"]
        collections = [p for p in kgm_paths if p.relation_type == "inCollection"]
        themes = [p for p in kgl_paths if p.relation_type == "hasTheme"]
        devices = [p for p in kgl_paths if p.relation_type == "hasNarrativeDevice"]
        settings = [p for p in kgl_paths if p.relation_type == "hasSetting"]
        actors = [p for p in kgm_paths if p.relation_type == "starsActor"]
        genres = [p for p in kgm_paths if p.relation_type == "hasGenre"]

        primary_sm_title = support_movies[0].title if support_movies else None

        # 1. Collection match (strongest specific link)
        if collections and primary_sm_title:
            coll_name = collections[0].target_entity_value
            components.append(
                f"Belongs to the '{coll_name}', continuing the franchise from {collections[0].source_movie_title} in your positive history."
            )

        # 2. Director match
        if directors and len(components) < 2:
            director_name = directors[0].target_entity_value
            components.append(
                f"Directed by {director_name}, who also directed {directors[0].source_movie_title} from your viewing history."
            )

        # 3. Theme match
        if themes and len(components) < 2:
            theme_names = [t.target_entity_value for t in themes[:2]]
            theme_str = "', '".join(theme_names)
            components.append(
                f"Shares narrative theme '{theme_str}' with {themes[0].source_movie_title} from your positive viewing history."
            )

        # 4. Setting / Narrative Device match
        if (devices or settings) and len(components) < 2:
            detail_parts = []
            if devices:
                detail_parts.append(f"narrative device '{devices[0].target_entity_value}'")
            if settings:
                detail_parts.append(f"setting '{settings[0].target_entity_value}'")
            detail_str = " and ".join(detail_parts)
            src_m = devices[0].source_movie_title if devices else settings[0].source_movie_title
            components.append(f"Features {detail_str}, matching elements from {src_m} in your history.")

        # 5. Actor match
        if actors and len(components) < 2:
            actor_names = [a.target_entity_value for a in actors[:2]]
            actor_str = ", ".join(actor_names)
            components.append(
                f"Stars {actor_str}, who also appeared in {actors[0].source_movie_title} in your history."
            )

        # 6. Genre match
        if genres and len(components) < 2 and (not themes and not directors and not collections):
            genre_names = [g.target_entity_value for g in genres[:2]]
            genre_str = ", ".join(genre_names)
            components.append(
                f"Connected to {genre_str} films in your viewing history such as {genres[0].source_movie_title}."
            )

        # 7. Emotion compatibility
        if has_emotion and len(components) < 3:
            top_emotions = [e["emotion"] for e in emotion_ev.top_aligned_emotions[:2]]
            emo_str = " and ".join(top_emotions)
            components.append(
                f"Matches the emotional profile of movies in your viewing history, particularly with aligned {emo_str}."
            )

        # 8. Text-semantic compatibility (if no KG paths and no emotion)
        if not components and text_ev and text_ev.sbert_cosine_similarity > 0.4:
            components.append(
                "Plot overview demonstrates high semantic compatibility with movies in your positive viewing history."
            )

        # 9. Collaborative fallback (if still no components)
        if not components:
            components.append(
                f"Recommended based on collaborative viewing patterns across users with similar tastes (ranked #{ease_rank} in collaborative filtering)."
            )

        # Classify explanation type
        has_kgl_comp = bool(themes or devices or settings)
        has_kgm_comp = bool(directors or collections or actors or genres)
        has_emo_comp = any("emotional profile" in c for c in components)
        has_text_comp = any("semantic compatibility" in c for c in components)
        has_collab_comp = any("collaborative" in c for c in components)

        active_facets = sum([has_kgl_comp, has_kgm_comp, has_emo_comp])
        if active_facets >= 2:
            exp_type = "hybrid"
        elif has_kgl_comp:
            exp_type = "semantic-theme-driven"
        elif has_kgm_comp:
            exp_type = "metadata-driven"
        elif has_emo_comp:
            exp_type = "emotion-driven"
        elif has_text_comp:
            exp_type = "text-semantic-driven"
        else:
            exp_type = "collaborative-fallback"

        # Join components into concise final text
        full_text = " ".join(components)
        return exp_type, components, full_text
