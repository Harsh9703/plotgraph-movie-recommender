"""KG-L Extraction Contract: Fixed Ontology, Extraction Schema, Prompts, and Validation.

Phase 3C establishes the contract and validation infrastructure for future LLM plot extraction
(Qwen2.5-7B-Instruct) without requiring model execution or GPU inference in this phase.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
import yaml

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. FIXED KG-L ONTOLOGY CONSTANTS
# ==============================================================================

# Exact semantic relation names allowed in KG-L (connecting Movie -> Semantic Entity)
KGL_SEMANTIC_RELATIONS: Tuple[str, ...] = (
    "hasTheme",
    "hasSetting",
    "hasNarrativeDevice",
    "hasProtagonistType",
    "basedOn",
)

# Complete fixed relation set including emotion relation
KGL_ALL_RELATIONS: Tuple[str, ...] = (
    "hasTheme",
    "hasSetting",
    "hasNarrativeDevice",
    "hasProtagonistType",
    "basedOn",
    "evokesEmotion",
)

# Exact eight Plutchik basic emotion dimensions
PLUTCHIK_EMOTIONS: Tuple[str, ...] = (
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
)

# Intensity threshold for creating evokesEmotion edges in the future KG-L (score > threshold).
# Pilot-driven methodology revision: revised from originally proposed 0.5 to 0.3 based on empirical pilot evaluation.
# Calibration: 0.0 = absent, 0.1-0.3 = weak, 0.4-0.6 = moderate, 0.7-0.8 = strong, 0.9-1.0 = dominant.
# A threshold of score > 0.3 captures moderate-or-stronger evidence-grounded emotions, achieving 76.7% movie coverage
# on pilot data (vs 23.3% at >0.5) while conservatively excluding weak signals (<=0.3).
# Note: affects only discrete RDF edge creation and prominent-emotion labels; raw continuous scores
# across all 8 dimensions are always preserved.
DEFAULT_EMOTION_EDGE_THRESHOLD: float = 0.3

PROMPT_VERSION_V1_0_0: str = "v1.0.0"
PROMPT_VERSION_V1_1_0: str = "v1.1.0"
DEFAULT_PROMPT_VERSION: str = "v1.1.0"
DEFAULT_SCHEMA_VERSION: str = "v1.0.0"
DEFAULT_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"


# ==============================================================================
# 2. ENTITY CANONICALIZATION
# ==============================================================================

def canonicalize_semantic_entity(text: str) -> str:
    """Deterministically normalize an extracted semantic entity string.

    Rules:
    - Unicode NFKC normalization.
    - Trim leading and trailing whitespace.
    - Collapse multiple whitespace characters into a single space.
    - Lowercase for canonical deduplication.
    - Does NOT perform fuzzy semantic merging.

    Args:
        text: Extracted entity string.

    Returns:
        Canonical entity string.
    """
    if text is None:
        raise ValueError("Cannot canonicalize None entity text")

    text_str = str(text).strip()
    if not text_str or text_str.lower() in ("nan", "none"):
        raise ValueError(f"Cannot canonicalize invalid/empty entity text: {text!r}")

    norm = unicodedata.normalize("NFKC", text_str)
    collapsed = re.sub(r"\s+", " ", norm).strip().lower()
    return collapsed


# ==============================================================================
# 3. STRICT PYDANTIC EXTRACTION SCHEMA
# ==============================================================================

class PlutchikEmotions(BaseModel):
    """Continuous Plutchik emotion intensity vector in [0.0, 1.0].

    Strictly forbids extra fields, missing emotions, and values outside [0, 1].
    """
    model_config = ConfigDict(extra="forbid")

    joy: float = Field(..., ge=0.0, le=1.0, description="Plutchik joy intensity in [0, 1]")
    trust: float = Field(..., ge=0.0, le=1.0, description="Plutchik trust intensity in [0, 1]")
    fear: float = Field(..., ge=0.0, le=1.0, description="Plutchik fear intensity in [0, 1]")
    surprise: float = Field(..., ge=0.0, le=1.0, description="Plutchik surprise intensity in [0, 1]")
    sadness: float = Field(..., ge=0.0, le=1.0, description="Plutchik sadness intensity in [0, 1]")
    disgust: float = Field(..., ge=0.0, le=1.0, description="Plutchik disgust intensity in [0, 1]")
    anger: float = Field(..., ge=0.0, le=1.0, description="Plutchik anger intensity in [0, 1]")
    anticipation: float = Field(..., ge=0.0, le=1.0, description="Plutchik anticipation intensity in [0, 1]")

    def to_dict(self) -> Dict[str, float]:
        return {emo: getattr(self, emo) for emo in PLUTCHIK_EMOTIONS}

    def get_prominent_emotions(self, threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD) -> Dict[str, float]:
        """Return subset of emotions whose intensity is strictly above (greater than) the edge threshold."""
        return {emo: getattr(self, emo) for emo in PLUTCHIK_EMOTIONS if getattr(self, emo) > threshold}


class MoviePlotExtraction(BaseModel):
    """Strict validated schema for one movie's LLM plot extraction."""
    model_config = ConfigDict(extra="forbid")

    movie_idx: int = Field(..., description="Internal 0-indexed movie index")
    movie_id: int = Field(..., description="Original MovieLens movie ID")
    tmdb_id: Optional[int] = Field(default=None, description="TMDB movie ID if available")
    title: str = Field(..., min_length=1, description="Movie title")

    # Fixed semantic categories
    themes: List[str] = Field(default_factory=list, description="Central recurring themes")
    settings: List[str] = Field(default_factory=list, description="Plot settings / environments / eras")
    narrative_devices: List[str] = Field(default_factory=list, description="Storytelling / narrative devices")
    protagonist_types: List[str] = Field(default_factory=list, description="Protagonist roles or archetypes")
    based_on: List[str] = Field(default_factory=list, description="Explicit source material")

    # Emotions
    emotions: PlutchikEmotions = Field(..., description="Plutchik 8-emotion intensity scores")

    # Metadata for provenance & auditing
    model_id: Optional[str] = Field(default=DEFAULT_MODEL_ID, description="Extraction model identifier")
    prompt_version: Optional[str] = Field(default=DEFAULT_PROMPT_VERSION, description="Prompt version used")
    schema_version: Optional[str] = Field(default=DEFAULT_SCHEMA_VERSION, description="Schema version used")

    @field_validator(
        "themes",
        "settings",
        "narrative_devices",
        "protagonist_types",
        "based_on",
        mode="before",
    )
    @classmethod
    def clean_and_deduplicate_entities(cls, v: Any) -> List[str]:
        """Validate string elements, discard empty strings, and deduplicate deterministically."""
        if not isinstance(v, list):
            raise ValueError(f"Entity category must be a list of strings, got {type(v).__name__}")

        cleaned: List[str] = []
        seen_canonical: Set[str] = set()

        for item in v:
            if not isinstance(item, str):
                raise ValueError(f"All entities must be strings, got {type(item).__name__}: {item!r}")

            trimmed = item.strip()
            if not trimmed or trimmed.lower() in ("nan", "none"):
                continue  # Discard blank strings

            canon = canonicalize_semantic_entity(trimmed)
            if canon not in seen_canonical:
                seen_canonical.add(canon)
                cleaned.append(trimmed)

        return cleaned


# ==============================================================================
# 4. PARSER EXCEPTIONS & RECOVERY LOGIC
# ==============================================================================

class ExtractionParseError(Exception):
    """Raised when JSON cannot be parsed from the model response."""
    pass


class ExtractionValidationError(Exception):
    """Raised when extracted JSON fails Pydantic schema validation."""
    pass


def extract_json_string(response_text: str) -> str:
    """Extract a valid JSON object string from model response text.

    Supports:
    - Direct raw JSON string
    - Markdown fenced blocks (```json ... ``` or ``` ... ```)
    - Embedded JSON objects with surrounding text

    Args:
        response_text: Raw output string from LLM.

    Returns:
        Isolated JSON string.

    Raises:
        ExtractionParseError: If no JSON object can be located.
    """
    if not response_text or not isinstance(response_text, str):
        raise ExtractionParseError("Empty or non-string response text")

    stripped = response_text.strip()

    # 1. Direct valid JSON check
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    # 2. Markdown fenced block
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # 3. Outer braces in mixed prose
    brace_match = re.search(r"(\{.*\})", stripped, re.DOTALL)
    if brace_match:
        return brace_match.group(1).strip()

    raise ExtractionParseError("No JSON object structure found in response text")


def parse_and_validate_extraction(
    response_text: str,
    expected_ids: Optional[Dict[str, Any]] = None,
) -> MoviePlotExtraction:
    """Parse raw LLM response text into a validated MoviePlotExtraction object.

    Args:
        response_text: Model response string (raw or markdown-wrapped JSON).
        expected_ids: Optional dict containing expected 'movie_id' and 'movie_idx' for verification.

    Returns:
        Validated MoviePlotExtraction instance.

    Raises:
        ExtractionParseError: If JSON syntax is invalid or unrecoverable.
        ExtractionValidationError: If JSON violates the strict extraction schema.
    """
    json_str = extract_json_string(response_text)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as err:
        raise ExtractionParseError(f"Malformed JSON syntax: {err}") from err

    if not isinstance(data, dict):
        raise ExtractionValidationError(f"Expected JSON object, got {type(data).__name__}")

    try:
        extraction = MoviePlotExtraction.model_validate(data)
    except ValidationError as err:
        raise ExtractionValidationError(f"Schema validation failed: {err}") from err

    # Verify ID alignment if expected_ids provided
    if expected_ids:
        if "movie_id" in expected_ids and extraction.movie_id != expected_ids["movie_id"]:
            raise ExtractionValidationError(
                f"movie_id mismatch: expected {expected_ids['movie_id']}, got {extraction.movie_id}"
            )
        if "movie_idx" in expected_ids and extraction.movie_idx != expected_ids["movie_idx"]:
            raise ExtractionValidationError(
                f"movie_idx mismatch: expected {expected_ids['movie_idx']}, got {extraction.movie_idx}"
            )

    return extraction


# ==============================================================================
# 5. EXTRACTION PROMPT TEMPLATES (VERSIONED)
# ==============================================================================

EXTRACTION_SYSTEM_PROMPT_V1 = """You are an expert movie knowledge graph extraction assistant for the PlotGraph research project.
Your task is to analyze a movie plot overview and extract structured semantic entities and Plutchik emotion intensities.

CRITICAL RULES:
1. Grounding: Rely ONLY on facts and concepts explicitly stated or directly described in the provided Plot Overview.
2. No Hallucination: Do NOT use outside knowledge about the movie, actors, history, or franchise.
3. No Missing Replacements: If a category is not supported by the overview, return an empty array []. Do NOT invent facts.
4. Output Format: Output ONLY a single valid JSON object. No preamble, no explanation, no markdown commentary.
5. Entity Conciseness: Entities must be concise noun phrases (1-4 words). Do NOT output full sentences.
6. Emotions: Score all eight Plutchik basic emotions between 0.0 and 1.0 based on what the plot narrative EVOKES for an audience (0.0 = absent/minimal, 1.0 = dominant).

ONTOLOGY DEFINITIONS:
- themes (hasTheme): Abstract recurring ideas or subjects central to the plot (e.g. "friendship", "revenge", "identity").
- settings (hasSetting): Key locations, worlds, historical eras, or social contexts (e.g. "victorian london", "deep space", "suburbia").
- narrative_devices (hasNarrativeDevice): Storytelling mechanisms evident in the plot (e.g. "time travel", "unreliable narrator", "mistaken identity").
- protagonist_types (hasProtagonistType): Concise character roles or archetypes (e.g. "detective", "high school student", "cyborg").
- based_on (basedOn): Source material explicitly mentioned in the overview (e.g. "novel", "true story", "comic book"). Do NOT assume source material from outside memory.
- emotions (evokesEmotion): Exact 8 Plutchik dimensions (joy, trust, fear, surprise, sadness, disgust, anger, anticipation).
"""

EXTRACTION_USER_PROMPT_TEMPLATE_V1 = """Movie Identifiers:
- Title: {title}
- MovieLens ID: {movie_id}
- Movie Index: {movie_idx}
- TMDB ID: {tmdb_id}

Plot Overview:
\"\"\"{overview}\"\"\"

Extract the structured knowledge graph entities and Plutchik emotion scores strictly following this JSON schema:
{{
  "movie_idx": {movie_idx},
  "movie_id": {movie_id},
  "tmdb_id": {tmdb_id_json},
  "title": "{title_escaped}",
  "themes": ["<theme1>", "<theme2>"],
  "settings": ["<setting1>"],
  "narrative_devices": ["<device1>"],
  "protagonist_types": ["<archetype1>"],
  "based_on": ["<source_if_explicit>"],
  "emotions": {{
    "joy": 0.0,
    "trust": 0.0,
    "fear": 0.0,
    "surprise": 0.0,
    "sadness": 0.0,
    "disgust": 0.0,
    "anger": 0.0,
    "anticipation": 0.0
  }}
}}
"""

EXTRACTION_SYSTEM_PROMPT_V1_1 = """You are an expert movie knowledge graph extraction assistant for the PlotGraph research project.
Your task is to analyze a movie plot overview and extract structured semantic entities and Plutchik emotion intensities.

CRITICAL RULES:
1. Strict Grounding: Rely ONLY on facts, concepts, and narrative events explicitly stated or directly described in the provided Plot Overview. Do NOT use outside knowledge about the movie, franchise, source material, characters, actors, or production history.
2. Output Format: Output ONLY a single valid JSON object strictly matching the specified JSON schema. No preamble, no explanation, no markdown commentary.
3. Entity Conciseness: Entities must be concise noun phrases (1-4 words). Do NOT output full sentences.
4. Prefer Empty Arrays Over Speculation: If a semantic category is not clearly supported by the overview, return an empty array []. An empty array [] is always preferable to inventing weak, speculative, or unsupported entries.
5. Protagonist Types Grounding:
   - Extract character roles or archetypes directly supported by the overview.
   - Do NOT infer occupation, age, identity, or franchise knowledge from outside knowledge.
   - Prefer reusable concise roles (e.g. soldier, student, parent, criminal, scientist, journalist, athlete) ONLY when the overview explicitly supports them.
   - An empty array [] is preferable to an unsupported role.
6. Narrative Device Definitions and Exclusions:
   - A narrative device is a storytelling or plot mechanism evident from the overview, such as: time travel, mistaken identity, unreliable narrator, nonlinear narrative, flashback, parallel timelines, frame narrative.
   - It must NOT simply be: an ordinary plot event, a theme, a protagonist type, a conflict, or a character trait.
   - If no genuine narrative device is clearly supported, output []. An empty array [] is preferable to inventing a weak narrative device.
7. Emotion Calibration and Inference:
   - Score all eight Plutchik basic emotions between 0.0 and 1.0 based on what the plot narrative EVOKES for an audience.
   - Do NOT require literal emotion words in the overview. Infer likely audience emotions from plot events that are explicitly described.
   - Calibration guidance:
     * 0.0 = essentially absent / no meaningful evidence
     * 0.1–0.3 = weak
     * 0.4–0.6 = moderate
     * 0.7–0.8 = strong
     * 0.9–1.0 = dominant or extremely intense
   - Zero means essentially no evidence that the narrative would evoke that emotion, not merely that the emotion word itself is absent.
   - Multiple emotions may be nonzero simultaneously.
   - Do not artificially force every emotion to be nonzero.
   - Do not artificially force an emotion above 0.5 merely to create a KG edge.
   - Scores must reflect evidence in the supplied overview.
   - Calibration guidance provides reference benchmarks, not deterministic scoring rules.
   - Examples of evidence that MAY support emotions (these are illustrative examples, not mandatory mappings):
     * threat, danger, pursuit, horror → fear
     * death, grief, separation, loss → sadness
     * betrayal, injustice, violent conflict → anger and/or disgust
     * friendship, romance, reunion, humor, success → joy and/or trust
     * uncertainty, revelations, escalating conflict → surprise and/or anticipation

ONTOLOGY DEFINITIONS:
- themes (hasTheme): Abstract recurring ideas or subjects central to the plot (e.g. "friendship", "revenge", "identity", "survival").
- settings (hasSetting): Key locations, worlds, historical eras, or social contexts (e.g. "victorian london", "deep space", "suburbia").
- narrative_devices (hasNarrativeDevice): Storytelling or plot mechanisms (e.g. "time travel", "mistaken identity", "unreliable narrator", "nonlinear narrative", "flashback", "frame narrative"). NOT ordinary plot events, character traits, or conflicts.
- protagonist_types (hasProtagonistType): Concise character roles or archetypes directly supported by the overview (e.g. "soldier", "student", "parent"). NOT inferred from outside knowledge.
- based_on (basedOn): Source material explicitly mentioned in the overview (e.g. "novel", "true story", "comic book"). Do NOT assume source material from outside memory.
- emotions (evokesEmotion): Exact 8 Plutchik continuous dimensions (joy, trust, fear, surprise, sadness, disgust, anger, anticipation) scored in [0.0, 1.0].
"""

EXTRACTION_USER_PROMPT_TEMPLATE_V1_1 = """Movie Identifiers:
- Title: {title}
- MovieLens ID: {movie_id}
- Movie Index: {movie_idx}
- TMDB ID: {tmdb_id}

Plot Overview:
\"\"\"{overview}\"\"\"

Extract the structured knowledge graph entities and calibrated Plutchik emotion scores strictly following this JSON schema:
{{
  "movie_idx": {movie_idx},
  "movie_id": {movie_id},
  "tmdb_id": {tmdb_id_json},
  "title": "{title_escaped}",
  "themes": ["<theme1>", "<theme2>"],
  "settings": ["<setting1>"],
  "narrative_devices": ["<device_if_present>"],
  "protagonist_types": ["<archetype_if_supported>"],
  "based_on": ["<source_if_explicit>"],
  "emotions": {{
    "joy": 0.0,
    "trust": 0.0,
    "fear": 0.0,
    "surprise": 0.0,
    "sadness": 0.0,
    "disgust": 0.0,
    "anger": 0.0,
    "anticipation": 0.0
  }}
}}
"""


def format_extraction_prompt(
    title: str,
    overview: str,
    movie_id: int,
    movie_idx: int,
    tmdb_id: Optional[int] = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> Tuple[str, str]:
    """Format the system prompt and user prompt for Qwen2.5-7B-Instruct extraction.

    Args:
        title: MovieLens movie title.
        overview: Plot overview text.
        movie_id: Original MovieLens movie ID.
        movie_idx: Internal 0-indexed movie index.
        tmdb_id: TMDB ID if available.
        prompt_version: Version identifier ('v1.0.0' or 'v1.1.0').

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    if prompt_version in ("v1.0.0", "v1.0"):
        system_prompt = EXTRACTION_SYSTEM_PROMPT_V1.strip()
        user_template = EXTRACTION_USER_PROMPT_TEMPLATE_V1
    elif prompt_version in ("v1.1.0", "v1.1"):
        system_prompt = EXTRACTION_SYSTEM_PROMPT_V1_1.strip()
        user_template = EXTRACTION_USER_PROMPT_TEMPLATE_V1_1
    else:
        logger.warning(
            f"Unknown prompt version '{prompt_version}', defaulting to {DEFAULT_PROMPT_VERSION}"
        )
        system_prompt = EXTRACTION_SYSTEM_PROMPT_V1_1.strip()
        user_template = EXTRACTION_USER_PROMPT_TEMPLATE_V1_1

    title_escaped = title.replace('"', '\\"')
    tmdb_id_json = str(tmdb_id) if tmdb_id is not None else "null"

    user_prompt = user_template.format(
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
# 6. DETERMINISTIC SAMPLE MOVIE EXPORT
# ==============================================================================

def select_sample_movies(
    df: pd.DataFrame,
    sample_size: int = 30,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Select a deterministic, reproducible sample of metadata-rich movies for extraction testing.

    Selection Criteria:
    - Must have valid TMDB link (has_tmdb_link == True).
    - Must have non-empty plot overview (has_overview == True).
    - Sampled reproducibly using fixed random seed.
    - Sorted by internal movie_idx.

    Args:
        df: Unified metadata DataFrame (from movies_with_tmdb.parquet).
        sample_size: Number of movies to select (default 30).
        random_seed: Random state for reproducibility.

    Returns:
        Sample DataFrame with selected movies.
    """
    # Filter to eligible movies with verified TMDB mapping and non-empty overview
    eligible = df[df["has_tmdb_link"] & df["has_overview"]].copy()

    if len(eligible) < sample_size:
        raise ValueError(f"Not enough eligible movies ({len(eligible)}) for sample_size={sample_size}")

    sample = (
        eligible.sample(n=sample_size, random_state=random_seed)
        .sort_values(by="movie_idx")
        .reset_index(drop=True)
    )

    # Attach auditing metadata
    sample["sample_seed"] = random_seed
    sample["sample_size"] = sample_size
    return sample


def export_sample_movies(
    config_path_or_dict: Union[str, Path, Dict[str, Any]],
) -> pd.DataFrame:
    """Load Phase 3A metadata, select deterministic sample, and export to parquet.

    Args:
        config_path_or_dict: Path to kg_l.yaml or parsed config dict.

    Returns:
        Exported sample DataFrame.
    """
    if isinstance(config_path_or_dict, (str, Path)):
        with open(config_path_or_dict, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = config_path_or_dict

    paths = config["paths"]
    sampling = config.get("sampling", {})
    sample_size = sampling.get("sample_size", 30)
    seed = sampling.get("random_seed", 42)

    input_parquet = Path(paths["input_metadata_parquet"])
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input metadata parquet not found at {input_parquet}")

    df = pd.read_parquet(input_parquet)
    logger.info(f"Loaded {len(df)} movies from {input_parquet}.")

    sample_df = select_sample_movies(df, sample_size=sample_size, random_seed=seed)

    output_parquet = Path(paths["sample_movies_parquet"])
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(output_parquet, engine="pyarrow", index=False)
    logger.info(f"Saved {len(sample_df)} sample movies to {output_parquet}")

    # Save summary JSON
    summary_path = Path(paths.get("sample_summary_json", output_parquet.parent / "sample_summary.json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "sample_size": len(sample_df),
        "random_seed": seed,
        "source_catalog_size": len(df),
        "eligible_movies": int((df["has_tmdb_link"] & df["has_overview"]).sum()),
        "columns": sample_df.columns.tolist(),
        "movies": [
            {
                "movie_idx": int(row["movie_idx"]),
                "movie_id": int(row["movie_id"]),
                "tmdb_id": int(row["tmdb_id"]) if pd.notna(row["tmdb_id"]) else None,
                "title": str(row["ml_title"]),
                "release_decade": str(row["release_decade"]) if pd.notna(row["release_decade"]) else None,
                "overview_len": len(str(row["overview"])),
            }
            for _, row in sample_df.iterrows()
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved sample summary to {summary_path}")

    return sample_df


# ==============================================================================
# 7. SYNTHETIC MOCK FIXTURES FOR TESTING
# ==============================================================================

def get_synthetic_fixtures() -> Dict[str, Any]:
    """Return synthetic mock extraction fixtures for unit testing.

    Explicitly labeled SYNTHETIC; not real LLM extraction results.
    """
    valid_obj = {
        "movie_idx": 0,
        "movie_id": 1,
        "tmdb_id": 862,
        "title": "Toy Story (1995)",
        "themes": ["friendship", "jealousy", "loyalty"],
        "settings": ["suburban home", "child's bedroom"],
        "narrative_devices": ["rivalry to friendship"],
        "protagonist_types": ["sentient toy", "cowboy doll"],
        "based_on": [],
        "emotions": {
            "joy": 0.85,
            "trust": 0.80,
            "fear": 0.20,
            "surprise": 0.40,
            "sadness": 0.30,
            "disgust": 0.05,
            "anger": 0.25,
            "anticipation": 0.75,
        },
        "model_id": DEFAULT_MODEL_ID,
        "prompt_version": DEFAULT_PROMPT_VERSION,
        "schema_version": DEFAULT_SCHEMA_VERSION,
    }

    empty_categories_obj = {
        "movie_idx": 140,
        "movie_id": 142,
        "tmdb_id": None,
        "title": "Minimal Movie (1990)",
        "themes": [],
        "settings": [],
        "narrative_devices": [],
        "protagonist_types": [],
        "based_on": [],
        "emotions": {
            "joy": 0.1,
            "trust": 0.2,
            "fear": 0.3,
            "surprise": 0.1,
            "sadness": 0.5,
            "disgust": 0.1,
            "anger": 0.2,
            "anticipation": 0.2,
        },
    }

    duplicate_entities_obj = {
        **valid_obj,
        "themes": ["friendship", "Friendship", "FRIENDSHIP", "jealousy", "   "],
    }

    unicode_entities_obj = {
        **valid_obj,
        "themes": ["cinéma vérité", "déjà vu", "übermensch"],
        "settings": ["champs-élysées", "tōkyō"],
    }

    return {
        "valid": valid_obj,
        "empty_categories": empty_categories_obj,
        "duplicate_entities": duplicate_entities_obj,
        "unicode_entities": unicode_entities_obj,
        "valid_fenced_json": f"```json\n{json.dumps(valid_obj, indent=2)}\n```",
        "valid_mixed_prose": f"Here is the extraction:\n```json\n{json.dumps(valid_obj)}\n```\nHope this helps!",
    }
