"""Adapter connecting cold-start recommendations to Phase 6 grounded explanations and Graphviz visualization."""

from typing import Any, Dict, List, Optional, Set, Tuple
import graphviz
import numpy as np
import pandas as pd

from src.explanations.engine import ExplanationEngine, FEATURE_FAMILY_MAP
from src.explanations.path_finder import MovieKnowledgeIndex
from src.explanations.schemas import StructuredExplanation
from src.explanations.validator import ExplanationValidator


def _adapt_cold_start_text(text: str) -> str:
    """Adjust viewing history phrases to cold-start interactive user phrasing."""
    replacements = [
        ("movies in your positive viewing history", "your selected movies"),
        ("movies in your viewing history", "your selected movies"),
        ("from your positive viewing history", "from your selected movies"),
        ("from your viewing history", "from your selected movies"),
        ("in your positive viewing history", "in your selected movies"),
        ("in your viewing history", "in your selected movies"),
        ("in your positive history", "in your selected movies"),
        ("in your history", "in your selected movies"),
        (
            "Matches the emotional profile of movies in your viewing history",
            "Matches the emotional profile of your selected movies",
        ),
        (
            "matches the emotional profile of movies in your viewing history",
            "matches the emotional profile of your selected movies",
        ),
    ]
    adapted = text
    for old, new in replacements:
        adapted = adapted.replace(old, new)
    return adapted


class ColdStartExplanationAdapter:
    """Generates and validates grounded explanations for cold-start recommendations."""

    def __init__(
        self,
        knowledge_index: MovieKnowledgeIndex,
        reranker_model: Any,
        min_emotion_threshold: float = 0.2,
    ):
        self.knowledge_index = knowledge_index
        self.reranker_model = reranker_model
        self.engine = ExplanationEngine(
            knowledge_index=self.knowledge_index,
            min_emotion_threshold=min_emotion_threshold,
        )
        self.validator = ExplanationValidator(knowledge_index=self.knowledge_index)
        self.feature_names = self.reranker_model.booster_.feature_name()

    def explain_recommendation(
        self,
        candidate_row: pd.Series,
        feature_row: pd.Series,
        selected_movie_indices: List[int],
        demo_user_idx: int = 0,
    ) -> StructuredExplanation:
        """Generate a validated StructuredExplanation for a single cold-start recommendation."""
        midx = int(candidate_row["movie_idx"])
        ease_rank = int(candidate_row["ease_rank"])
        plotgraph_rank = int(candidate_row["plotgraph_rank"])
        reranker_score = float(candidate_row["reranker_score"])

        # History is defined strictly by the selected movies
        user_history = [(m, 5.0, 1000) for m in selected_movie_indices]
        selected_set = set(selected_movie_indices)

        # Compute additive LightGBM tree contributions for this row
        X_vec = feature_row[self.feature_names].to_numpy().reshape(1, -1)
        contribs_matrix = self.reranker_model.predict(X_vec, pred_contrib=True)
        feat_contribs = contribs_matrix[0, :-1]  # (89,)

        exp = self.engine.generate_explanation(
            user_idx=demo_user_idx,
            movie_idx=midx,
            ease_rank=ease_rank,
            plotgraph_rank=plotgraph_rank,
            reranker_score=reranker_score,
            feature_row=feature_row,
            feature_names=self.feature_names,
            feature_values=X_vec[0],
            contrib_values=feat_contribs,
            user_positive_history=user_history,
        )

        # Audit explanation against ground truth
        val_res = self.validator.validate(
            explanation=exp,
            user_pos_history_set=selected_set,
            test_movies_set=None,
            feature_row=feature_row,
        )

        if not val_res.is_valid:
            # Fall back to safe collaborative explanation if any detail fails validation
            exp.explanation_text = (
                f"Recommended based on collaborative viewing patterns across users with similar tastes "
                f"(ranked #{ease_rank} in collaborative filtering)."
            )
            exp.explanation_type = "collaborative-fallback"
            exp.explanation_components = [exp.explanation_text]
        else:
            # Adapt wording for interactive cold-start user
            exp.explanation_components = [
                _adapt_cold_start_text(c) for c in exp.explanation_components
            ]
            exp.explanation_text = _adapt_cold_start_text(exp.explanation_text)

        return exp

    def build_graphviz_diagram(
        self,
        explanation: StructuredExplanation,
    ) -> graphviz.Digraph:
        """Render a clean, presentation-ready multi-hop Knowledge Graph diagram for an explanation."""
        dot = graphviz.Digraph(
            comment=f"Explanation for {explanation.title}",
            format="svg",
        )
        dot.attr(rankdir="LR", bgcolor="transparent", fontsize="11", fontname="Helvetica")
        dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10")
        dot.attr("edge", fontname="Helvetica", fontsize="9", color="#666666")

        # Candidate Movie Node
        cand_id = f"cand_{explanation.movie_idx}"
        cand_label = f"RECOMMENDATION\n{explanation.title}\n(PlotGraph #{explanation.plotgraph_rank})"
        dot.node(
            cand_id,
            label=cand_label,
            fillcolor="#E3F2FD",
            color="#1565C0",
            penwidth="2.0",
        )

        # Supporting Selected Movie Nodes
        support_nodes: Dict[int, str] = {}
        for sm in explanation.historical_support_movies:
            sm_id = f"hist_{sm.movie_idx}"
            sm_label = f"SELECTED MOVIE\n{sm.title}"
            dot.node(
                sm_id,
                label=sm_label,
                fillcolor="#E8F5E9",
                color="#2E7D32",
                penwidth="1.5",
            )
            support_nodes[sm.movie_idx] = sm_id

        # KG-M Paths (Blue / Slate)
        for i, path in enumerate(explanation.kgm_paths[:4]):
            ent_id = f"kgm_{path.target_entity_type}_{i}"
            ent_label = f"KG-M [{path.target_entity_type.upper()}]\n{path.target_entity_value}"
            dot.node(
                ent_id,
                label=ent_label,
                fillcolor="#FFF3E0",
                color="#E65100",
                shape="ellipse",
            )
            # Edge from support movie to entity
            src_node = support_nodes.get(path.source_movie_idx, f"hist_{path.source_movie_idx}")
            dot.edge(src_node, ent_id, label=path.relation_type, color="#E65100")
            # Edge from entity to candidate movie
            dot.edge(ent_id, cand_id, label=path.relation_type, color="#E65100")

        # KG-L Paths (Purple / Magenta)
        for i, path in enumerate(explanation.kgl_paths[:4]):
            ent_id = f"kgl_{path.target_entity_type}_{i}"
            ent_label = f"KG-L [{path.target_entity_type.upper()}]\n{path.target_entity_value}"
            dot.node(
                ent_id,
                label=ent_label,
                fillcolor="#F3E5F5",
                color="#7B1FA2",
                shape="ellipse",
            )
            src_node = support_nodes.get(path.source_movie_idx, f"hist_{path.source_movie_idx}")
            dot.edge(src_node, ent_id, label=path.relation_type, color="#7B1FA2")
            dot.edge(ent_id, cand_id, label=path.relation_type, color="#7B1FA2")

        # Fallback if no paths exist
        if not explanation.kgm_paths and not explanation.kgl_paths:
            cf_id = "cf_node"
            dot.node(
                cf_id,
                label="COLLABORATIVE FILTERING\nEASE Co-occurrence\nPatterns",
                fillcolor="#ECEFF1",
                color="#455A64",
                shape="ellipse",
            )
            for sm_id in support_nodes.values():
                dot.edge(sm_id, cf_id, label="implicit co-viewing")
            dot.edge(cf_id, cand_id, label=f"EASE rank #{explanation.ease_rank}")

        return dot
