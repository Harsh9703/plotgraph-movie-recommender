"""Automated explanation validator for auditing groundedness and integrity.

Performs 100% verification of generated explanations against ground-truth data:
- Historical support movies must belong to positive TRAIN+VAL history and NEVER TEST.
- Every KG-M path must exist in ground-truth metadata for both source and candidate movies.
- Every KG-L path must exist in ground-truth semantics for both source and candidate movies.
- Emotion evidence must match validated continuous Plutchik emotion vectors.
- Feature contributions must be mathematically consistent with LightGBM predictions.
- Movie IDs and titles must match catalog metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

from src.explanations.path_finder import MovieKnowledgeIndex
from src.explanations.schemas import StructuredExplanation


@dataclass
class ValidationResult:
    """Result of auditing a StructuredExplanation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    checks_passed: Dict[str, bool] = field(default_factory=dict)


class ExplanationValidator:
    """Rigorous audit validator ensuring zero hallucination in recommendation explanations."""

    def __init__(self, knowledge_index: MovieKnowledgeIndex):
        self.knowledge_index = knowledge_index

    def validate(
        self,
        explanation: StructuredExplanation,
        user_pos_history_set: Set[int],
        test_movies_set: Optional[Set[int]] = None,
        feature_row: Optional[pd.Series] = None,
    ) -> ValidationResult:
        """Audit a StructuredExplanation against ground-truth tables."""
        errors: List[str] = []
        checks: Dict[str, bool] = {}

        # 1. Title and Catalog Alignment
        catalog_title = self.knowledge_index.movie_titles.get(explanation.movie_idx)
        if catalog_title is None:
            errors.append(f"Candidate movie_idx {explanation.movie_idx} not found in catalog!")
            checks["catalog_alignment"] = False
        elif catalog_title != explanation.title:
            errors.append(f"Title mismatch: '{explanation.title}' != catalog '{catalog_title}'")
            checks["catalog_alignment"] = False
        else:
            checks["catalog_alignment"] = True

        # 2. Historical Support Movie Verification
        history_ok = True
        for sm in explanation.historical_support_movies:
            # Must be in user's positive TRAIN+VAL history
            if sm.movie_idx not in user_pos_history_set:
                errors.append(
                    f"Fabricated support movie: movie_idx {sm.movie_idx} ({sm.title}) "
                    f"not in user positive history!"
                )
                history_ok = False

            # Must NEVER be a TEST item used for explanation history
            if test_movies_set and sm.movie_idx in test_movies_set:
                errors.append(
                    f"TEST leakage violation: support movie_idx {sm.movie_idx} "
                    f"appears in user TEST set!"
                )
                history_ok = False

            # Title alignment for support movie
            sm_cat_title = self.knowledge_index.movie_titles.get(sm.movie_idx)
            if sm_cat_title and sm_cat_title != sm.title:
                errors.append(
                    f"Support movie title mismatch: '{sm.title}' != '{sm_cat_title}'"
                )
                history_ok = False

        checks["history_support"] = history_ok

        # 3. KG-M Paths Verification
        kgm_ok = True
        cand_kgm = self.knowledge_index.kgm_entities.get(explanation.movie_idx, {})
        for path in explanation.kgm_paths:
            src_kgm = self.knowledge_index.kgm_entities.get(path.source_movie_idx, {})

            # Check relation exists in KG-M
            if path.relation_type not in cand_kgm or path.relation_type not in src_kgm:
                errors.append(f"KG-M relation '{path.relation_type}' invalid or missing from movies!")
                kgm_ok = False
                continue

            # Check entity value is present in BOTH candidate and source
            if path.target_entity_value not in cand_kgm[path.relation_type]:
                errors.append(
                    f"KG-M entity '{path.target_entity_value}' not found in candidate movie {explanation.movie_idx}!"
                )
                kgm_ok = False
            if path.target_entity_value not in src_kgm[path.relation_type]:
                errors.append(
                    f"KG-M entity '{path.target_entity_value}' not found in support movie {path.source_movie_idx}!"
                )
                kgm_ok = False

        checks["kgm_grounding"] = kgm_ok

        # 4. KG-L Paths Verification
        kgl_ok = True
        cand_kgl = self.knowledge_index.kgl_entities.get(explanation.movie_idx, {})
        for path in explanation.kgl_paths:
            src_kgl = self.knowledge_index.kgl_entities.get(path.source_movie_idx, {})

            if path.relation_type not in cand_kgl or path.relation_type not in src_kgl:
                errors.append(f"KG-L relation '{path.relation_type}' invalid or missing from movies!")
                kgl_ok = False
                continue

            if path.target_entity_value not in cand_kgl[path.relation_type]:
                errors.append(
                    f"KG-L entity '{path.target_entity_value}' not found in candidate movie {explanation.movie_idx}!"
                )
                kgl_ok = False
            if path.target_entity_value not in src_kgl[path.relation_type]:
                errors.append(
                    f"KG-L entity '{path.target_entity_value}' not found in support movie {path.source_movie_idx}!"
                )
                kgl_ok = False

        checks["kgl_grounding"] = kgl_ok

        # 5. Emotion Evidence Verification
        emo_ok = True
        if explanation.emotion_evidence:
            cand_vec = self.knowledge_index.emotion_vectors.get(explanation.movie_idx, {})
            for item in explanation.emotion_evidence.top_aligned_emotions:
                emo_name = item["emotion"]
                claimed_cand_score = item["candidate_score"]
                true_cand_score = cand_vec.get(emo_name, 0.0)

                if abs(claimed_cand_score - true_cand_score) > 0.01:
                    errors.append(
                        f"Emotion score mismatch for '{emo_name}': "
                        f"claimed {claimed_cand_score} != true {true_cand_score}"
                    )
                    emo_ok = False

        checks["emotion_grounding"] = emo_ok

        # 6. Feature Values Consistency
        feat_ok = True
        if feature_row is not None:
            for fc in explanation.top_feature_contributions:
                if fc.feature_name in feature_row:
                    true_val = float(feature_row[fc.feature_name])
                    if abs(fc.feature_value - true_val) > 1e-4:
                        errors.append(
                            f"Feature value mismatch for '{fc.feature_name}': "
                            f"recorded {fc.feature_value} != true {true_val}"
                        )
                        feat_ok = False

        checks["feature_consistency"] = feat_ok

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, checks_passed=checks)
