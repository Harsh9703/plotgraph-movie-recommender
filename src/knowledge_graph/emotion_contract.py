"""Phase 3E: Dedicated Evidence-Grounded Emotion Extraction Contract.

Defines:
1. Strict Pydantic extraction schema for evidence-grounded Plutchik emotions:
   - Each of the 8 Plutchik emotions requires:
     - evidence: list[str] (concise plot circumstances traceable to overview)
     - score: float in [0.0, 1.0]
   - Nonzero score requires at least one evidence item.
   - Zero score requires empty evidence list [].
2. Versioned emotion prompts (v1.0.0) with independent 2-step evaluation:
   - Step 1: Identify supporting plot evidence from overview.
   - Step 2: Assign calibrated intensity in [0, 1].
3. Parsing, recovery, and validation functions.
4. Derivation of standard 8-dimensional emotion vectors for downstream KG-L construction (score > 0.3; pilot-revised from >0.5).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .kg_l_contract import (
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    DEFAULT_MODEL_ID,
    PLUTCHIK_EMOTIONS,
    ExtractionParseError,
    ExtractionValidationError,
    PlutchikEmotions,
    extract_json_string,
)

logger = logging.getLogger(__name__)

EMOTION_PROMPT_VERSION_V1_0_0: str = "v1.0.0"
EMOTION_SCHEMA_VERSION_V1_0_0: str = "v1.0.0"
DEFAULT_EMOTION_PROMPT_VERSION: str = EMOTION_PROMPT_VERSION_V1_0_0
DEFAULT_EMOTION_SCHEMA_VERSION: str = EMOTION_SCHEMA_VERSION_V1_0_0


# ==============================================================================
# 1. STRICT EVIDENCE-GROUNDED EMOTION SCHEMA
# ==============================================================================

class EmotionEvidenceScore(BaseModel):
    """Evidence list and continuous intensity score for a single Plutchik emotion."""
    model_config = ConfigDict(extra="forbid")

    evidence: List[str] = Field(
        default_factory=list,
        description="Concise plot events/circumstances supporting this emotion",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated Plutchik intensity score in [0.0, 1.0]",
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def clean_evidence_list(cls, v: Any) -> List[str]:
        """Validate string elements, discard empty strings, and deduplicate deterministically."""
        if not isinstance(v, list):
            raise ValueError(f"Evidence must be a list of strings, got {type(v).__name__}")

        cleaned: List[str] = []
        seen: Set[str] = set()

        for item in v:
            if not isinstance(item, str):
                raise ValueError(f"All evidence items must be strings, got {type(item).__name__}: {item!r}")

            trimmed = re.sub(r"\s+", " ", item.strip())
            if not trimmed or trimmed.lower() in ("nan", "none"):
                continue

            lower_key = trimmed.lower()
            if lower_key not in seen:
                seen.add(lower_key)
                cleaned.append(trimmed)

        return cleaned

    @model_validator(mode="after")
    def validate_evidence_score_consistency(self) -> EmotionEvidenceScore:
        """Enforce strict bidirectional consistency between evidence and score:

        - score > 0.0 requires at least one evidence item.
        - score == 0.0 requires empty evidence list [].
        """
        if self.score > 0.0 and len(self.evidence) == 0:
            raise ValueError(
                f"Nonzero emotion score ({self.score}) requires at least one supporting evidence item."
            )
        if self.score == 0.0 and len(self.evidence) > 0:
            raise ValueError(
                f"Emotion score is 0.0, so evidence list must be empty [], got: {self.evidence}"
            )
        return self


class PlutchikEmotionEvidenceMap(BaseModel):
    """Strict mapping containing exactly the eight Plutchik basic emotions."""
    model_config = ConfigDict(extra="forbid")

    joy: EmotionEvidenceScore
    trust: EmotionEvidenceScore
    fear: EmotionEvidenceScore
    surprise: EmotionEvidenceScore
    sadness: EmotionEvidenceScore
    disgust: EmotionEvidenceScore
    anger: EmotionEvidenceScore
    anticipation: EmotionEvidenceScore

    def to_vector(self) -> PlutchikEmotions:
        """Extract continuous float scores as a PlutchikEmotions instance."""
        return PlutchikEmotions(**{emo: getattr(self, emo).score for emo in PLUTCHIK_EMOTIONS})

    def to_dict(self) -> Dict[str, float]:
        """Extract continuous float scores for all eight Plutchik emotions as a dictionary."""
        return {emo: getattr(self, emo).score for emo in PLUTCHIK_EMOTIONS}

    def to_evidence_dict(self) -> Dict[str, List[str]]:
        """Extract evidence lists for all eight Plutchik emotions."""
        return {emo: getattr(self, emo).evidence for emo in PLUTCHIK_EMOTIONS}


class MovieEmotionExtraction(BaseModel):
    """Strict validated schema for one movie's dedicated evidence-grounded emotion extraction."""
    model_config = ConfigDict(extra="forbid")

    movie_idx: int = Field(..., description="Internal 0-indexed movie index")
    movie_id: int = Field(..., description="Original MovieLens movie ID")
    tmdb_id: Optional[int] = Field(default=None, description="TMDB movie ID if available")
    title: str = Field(..., min_length=1, description="Movie title")

    # Exactly 8 Plutchik emotions with evidence and scores
    emotions: PlutchikEmotionEvidenceMap = Field(
        ...,
        description="Evidence and calibrated intensity for all eight Plutchik emotions",
    )

    # Metadata for provenance & auditing
    model_id: Optional[str] = Field(default=DEFAULT_MODEL_ID, description="Extraction model identifier")
    prompt_version: Optional[str] = Field(default=DEFAULT_EMOTION_PROMPT_VERSION, description="Emotion prompt version")
    schema_version: Optional[str] = Field(default=DEFAULT_EMOTION_SCHEMA_VERSION, description="Emotion schema version")

    def get_emotion_vector(self) -> PlutchikEmotions:
        """Derive standard PlutchikEmotions instance containing the eight continuous scores."""
        return self.emotions.to_vector()

    def get_prominent_emotions(self, threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD) -> List[str]:
        """Return list of emotion names whose intensity is strictly above the edge threshold (score > threshold)."""
        return [emo for emo in PLUTCHIK_EMOTIONS if getattr(self.emotions, emo).score > threshold]

    def get_prominent_emotions_dict(self, threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD) -> Dict[str, float]:
        """Return dict of prominent emotion names and scores strictly above threshold."""
        return {emo: getattr(self.emotions, emo).score for emo in PLUTCHIK_EMOTIONS if getattr(self.emotions, emo).score > threshold}

    def to_vector_record(self, threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD) -> Dict[str, Any]:
        """Format flat summary record suitable for emotion_vectors.jsonl."""
        rec: Dict[str, Any] = {
            "movie_idx": self.movie_idx,
            "movie_id": self.movie_id,
            "tmdb_id": self.tmdb_id,
            "title": self.title,
        }
        for emo in PLUTCHIK_EMOTIONS:
            rec[emo] = getattr(self.emotions, emo).score

        rec["prominent_emotions"] = self.get_prominent_emotions(threshold=threshold)
        rec["model_id"] = self.model_id
        rec["prompt_version"] = self.prompt_version
        rec["schema_version"] = self.schema_version
        return rec


# ==============================================================================
# 2. VERSIONED EMOTION PROMPT TEMPLATES (v1.0.0)
# ==============================================================================

EMOTION_SYSTEM_PROMPT_V1 = """You are an expert movie knowledge graph emotion analysis assistant for the PlotGraph research project.
Your task is to analyze a movie plot overview and determine the narrative evidence and intensity for each of Plutchik's eight basic emotions.

PLUTCHIK EMOTIONS EVALUATED:
- joy
- trust
- fear
- surprise
- sadness
- disgust
- anger
- anticipation

TWO-STEP EVALUATION PROCESS FOR EACH EMOTION:
For EACH of the eight emotions, independently perform:
Step 1. Identify Plot Evidence:
  Examine the supplied Plot Overview to identify concrete narrative events, situations, character relationships, conflicts, or circumstances that would cause an audience to experience this emotion.
  - If there is meaningful narrative evidence, provide concise summary items (1-10 words per item) in the "evidence" array.
  - If there is NO meaningful supporting narrative evidence, the "evidence" array MUST be empty: [].
Step 2. Assign Intensity Score:
  Assign a continuous intensity score between 0.0 and 1.0 based solely on the strength of the identified evidence:
  - 0.0 = essentially absent / no meaningful supporting narrative evidence (evidence must be [])
  - 0.1–0.3 = weak intensity
  - 0.4–0.6 = moderate intensity
  - 0.7–0.8 = strong intensity
  - 0.9–1.0 = dominant or extremely intense

CRITICAL GROUNDING & CALIBRATION RULES:
1. Authoritative Evidence: Base evidence and scores ONLY on the supplied Plot Overview. Do NOT use outside knowledge about the movie, actors, director, franchise, genre labels, or production history.
2. No Verbatim Requirement: Evidence should be concise, factual descriptions of plot circumstances; exact quotes are not required, but concepts must be explicitly traceable to the text.
3. No Literal Word Requirement: Explicit emotion words (e.g. "he felt sad") are NOT required. Described plot events and narrative situations evoke emotions.
4. Consistency Rule: If score > 0.0, you MUST provide at least one supporting evidence item in "evidence". If score is 0.0, "evidence" MUST be empty [].
5. Independent Scoring: Evaluate each emotion independently. Multiple emotions can be simultaneously strong (e.g. both fear and sadness can be 0.8). Scores do NOT sum to 1.0. Do NOT suppress one emotion because another is strong.
6. No Artificial Inflation: Do NOT force an emotion to be nonzero without supporting evidence. Do NOT artificially push scores above 0.5.
7. Output Format: Output ONLY a single valid JSON object matching the requested schema. No preamble, no explanation, no markdown text outside the JSON.

ILLUSTRATIVE REASONING GUIDELINES (Examples, NOT Mandatory Rules):
- threat, physical danger, pursuit, confinement, horror -> may support fear
- death, grief, separation, tragic loss, heartbreak -> may support sadness
- betrayal, cruelty, injustice, violent conflict, oppression -> may support anger and/or disgust
- friendship, affection, romance, humor, reunion, triumph, achievement -> may support joy and/or trust
- sudden uncertainty, revelation, plot twist, unexpected change -> may support surprise
- unresolved danger, mission goals, approaching conflict, suspense -> may support anticipation
"""

EMOTION_USER_PROMPT_TEMPLATE_V1 = """Movie Identifiers:
- Title: {title}
- MovieLens ID: {movie_id}
- Movie Index: {movie_idx}
- TMDB ID: {tmdb_id}

Plot Overview:
\"\"\"{overview}\"\"\"

Analyze the plot overview and extract the evidence-grounded Plutchik emotion evaluations strictly following this JSON schema:
{{
  "movie_idx": {movie_idx},
  "movie_id": {movie_id},
  "tmdb_id": {tmdb_id_json},
  "title": "{title_escaped}",
  "emotions": {{
    "joy": {{
      "evidence": ["<evidence_item_1>", "<evidence_item_2>"],
      "score": 0.0
    }},
    "trust": {{
      "evidence": [],
      "score": 0.0
    }},
    "fear": {{
      "evidence": [],
      "score": 0.0
    }},
    "surprise": {{
      "evidence": [],
      "score": 0.0
    }},
    "sadness": {{
      "evidence": [],
      "score": 0.0
    }},
    "disgust": {{
      "evidence": [],
      "score": 0.0
    }},
    "anger": {{
      "evidence": [],
      "score": 0.0
    }},
    "anticipation": {{
      "evidence": [],
      "score": 0.0
    }}
  }}
}}
"""


def format_emotion_prompt(
    title: str,
    overview: str,
    movie_id: int,
    movie_idx: int = 0,
    tmdb_id: Optional[int] = None,
    prompt_version: str = DEFAULT_EMOTION_PROMPT_VERSION,
) -> Tuple[str, str]:
    """Format system and user prompts for dedicated evidence-grounded emotion extraction.

    Args:
        title: MovieLens movie title.
        overview: Plot overview text.
        movie_id: Original MovieLens movie ID.
        movie_idx: Internal 0-indexed movie index (default: 0).
        tmdb_id: TMDB ID if available.
        prompt_version: Emotion prompt version identifier.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    if prompt_version not in ("v1.0.0", "v1.0"):
        logger.warning(
            f"Unknown emotion prompt version '{prompt_version}', defaulting to {DEFAULT_EMOTION_PROMPT_VERSION}"
        )

    system_prompt = EMOTION_SYSTEM_PROMPT_V1.strip()
    title_escaped = title.replace('"', '\\"')
    tmdb_id_json = str(tmdb_id) if tmdb_id is not None else "null"

    user_prompt = EMOTION_USER_PROMPT_TEMPLATE_V1.format(
        title=title,
        movie_id=movie_id,
        movie_idx=movie_idx,
        tmdb_id=tmdb_id if tmdb_id is not None else "N/A",
        tmdb_id_json=tmdb_id_json,
        title_escaped=title_escaped,
        overview=overview.strip(),
    ).strip()

    return system_prompt, user_prompt


# ==============================================================================
# 3. PARSING & VALIDATION
# ==============================================================================

def parse_and_validate_emotion_extraction(
    response_text: Optional[str] = None,
    expected_ids: Optional[Dict[str, Any]] = None,
    raw_text: Optional[str] = None,
    expected_movie_id: Optional[int] = None,
    expected_movie_idx: Optional[int] = None,
    expected_tmdb_id: Optional[int] = None,
) -> MovieEmotionExtraction:
    """Parse raw LLM response text into a validated MovieEmotionExtraction object.

    Args:
        response_text: Model response string (raw or markdown-wrapped JSON).
        expected_ids: Optional dict containing expected 'movie_id' and 'movie_idx' for verification.
        raw_text: Alias for response_text.
        expected_movie_id: Optional expected MovieLens movie ID.
        expected_movie_idx: Optional expected internal movie index.
        expected_tmdb_id: Optional expected TMDB movie ID.

    Returns:
        Validated MovieEmotionExtraction instance.

    Raises:
        ExtractionParseError: If JSON syntax is invalid or unrecoverable.
        ExtractionValidationError: If JSON violates the strict emotion extraction schema.
    """
    text_to_parse = response_text if response_text is not None else raw_text
    if text_to_parse is None:
        raise ExtractionParseError("No response text provided to parse")

    ids_to_check = dict(expected_ids) if expected_ids else {}
    if expected_movie_id is not None:
        ids_to_check["movie_id"] = expected_movie_id
    if expected_movie_idx is not None:
        ids_to_check["movie_idx"] = expected_movie_idx
    if expected_tmdb_id is not None:
        ids_to_check["tmdb_id"] = expected_tmdb_id

    json_str = extract_json_string(text_to_parse)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as err:
        raise ExtractionParseError(f"Malformed JSON syntax: {err}") from err

    if not isinstance(data, dict):
        raise ExtractionValidationError(f"Expected JSON object, got {type(data).__name__}")

    try:
        extraction = MovieEmotionExtraction.model_validate(data)
    except ValidationError as err:
        raise ExtractionValidationError(f"Emotion schema validation failed: {err}") from err

    # Verify ID alignment if expected_ids provided
    if ids_to_check:
        if "movie_id" in ids_to_check and extraction.movie_id != ids_to_check["movie_id"]:
            raise ExtractionValidationError(
                f"movie_id mismatch: expected {ids_to_check['movie_id']}, got {extraction.movie_id}"
            )
        if "movie_idx" in ids_to_check and extraction.movie_idx != ids_to_check["movie_idx"]:
            raise ExtractionValidationError(
                f"movie_idx mismatch: expected {ids_to_check['movie_idx']}, got {extraction.movie_idx}"
            )
        if "tmdb_id" in ids_to_check and extraction.tmdb_id != ids_to_check["tmdb_id"]:
            raise ExtractionValidationError(
                f"tmdb_id mismatch: expected {ids_to_check['tmdb_id']}, got {extraction.tmdb_id}"
            )

    return extraction
