"""Unit and integration tests for Phase 3C: KG-L extraction contract, schema, prompt, and sample export."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest

from src.knowledge_graph.kg_l_contract import (
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SCHEMA_VERSION,
    KGL_ALL_RELATIONS,
    KGL_SEMANTIC_RELATIONS,
    PLUTCHIK_EMOTIONS,
    ExtractionParseError,
    ExtractionValidationError,
    MoviePlotExtraction,
    PlutchikEmotions,
    canonicalize_semantic_entity,
    extract_json_string,
    format_extraction_prompt,
    get_synthetic_fixtures,
    parse_and_validate_extraction,
    select_sample_movies,
)


def test_fixed_ontology_relations_and_emotions():
    """Verify exact allowed relations and Plutchik emotion dimensions."""
    expected_semantic = (
        "hasTheme",
        "hasSetting",
        "hasNarrativeDevice",
        "hasProtagonistType",
        "basedOn",
    )
    assert KGL_SEMANTIC_RELATIONS == expected_semantic

    expected_all = (
        "hasTheme",
        "hasSetting",
        "hasNarrativeDevice",
        "hasProtagonistType",
        "basedOn",
        "evokesEmotion",
    )
    assert KGL_ALL_RELATIONS == expected_all

    expected_emotions = (
        "joy",
        "trust",
        "fear",
        "surprise",
        "sadness",
        "disgust",
        "anger",
        "anticipation",
    )
    assert PLUTCHIK_EMOTIONS == expected_emotions
    assert len(PLUTCHIK_EMOTIONS) == 8


def test_canonicalize_semantic_entity():
    """Verify deterministic canonicalization for semantic entities."""
    assert canonicalize_semantic_entity("  Friendship  ") == "friendship"
    assert canonicalize_semantic_entity("Space   Travel") == "space travel"
    assert canonicalize_semantic_entity("Victorian\nLondon") == "victorian london"

    # Unicode NFKC normalization
    assert canonicalize_semantic_entity("cinéma vérité") == "cinéma vérité"
    assert canonicalize_semantic_entity("Übermensch") == "übermensch"

    # Edge cases
    with pytest.raises(ValueError):
        canonicalize_semantic_entity("")
    with pytest.raises(ValueError):
        canonicalize_semantic_entity("   ")
    with pytest.raises(ValueError):
        canonicalize_semantic_entity(None)


def test_valid_schema_acceptance():
    """Verify that a compliant extraction object passes Pydantic validation."""
    fixtures = get_synthetic_fixtures()
    valid_data = fixtures["valid"]

    extraction = MoviePlotExtraction.model_validate(valid_data)
    assert extraction.movie_idx == 0
    assert extraction.movie_id == 1
    assert extraction.tmdb_id == 862
    assert extraction.title == "Toy Story (1995)"
    assert extraction.themes == ["friendship", "jealousy", "loyalty"]
    assert extraction.emotions.joy == 0.85
    assert extraction.emotions.trust == 0.80

    # Prominent emotions thresholding check (score > DEFAULT_EMOTION_EDGE_THRESHOLD)
    prominent = extraction.emotions.get_prominent_emotions()
    assert "joy" in prominent
    assert "trust" in prominent
    assert "anticipation" in prominent
    assert "disgust" not in prominent


def test_emotion_edge_threshold_boundary_behavior():
    """Verify that evokesEmotion edge creation strictly requires score > threshold (not >=).

    Boundary conditions at revised threshold 0.3 (pilot-driven methodology revision from 0.5):
    - score = 0.29 -> no edge
    - score = 0.30 -> no edge
    - score = 0.31 -> edge
    - score = 0.40 -> edge

    All continuous scores, including 0.30, 0.50, must be preserved in the schema.
    """
    assert DEFAULT_EMOTION_EDGE_THRESHOLD == 0.3

    emotions = PlutchikEmotions(
        joy=0.31,        # 0.31 > 0.30 is True  -> edge
        trust=0.30,      # 0.30 > 0.30 is False -> no edge
        fear=0.29,       # 0.29 > 0.30 is False -> no edge
        surprise=0.0,    # 0.00 > 0.30 is False -> no edge
        sadness=0.30,    # 0.30 > 0.30 is False -> no edge
        disgust=0.299,   # 0.299 > 0.30 is False -> no edge
        anger=0.40,      # 0.40 > 0.30 is True  -> edge
        anticipation=1.0 # 1.00 > 0.30 is True  -> edge
    )

    # 1. Schema preservation: exact raw scores must be preserved without distortion
    assert emotions.trust == 0.30
    assert emotions.sadness == 0.30
    assert emotions.to_dict()["trust"] == 0.30
    assert emotions.to_dict()["fear"] == 0.29
    assert emotions.to_dict()["joy"] == 0.31
    assert emotions.to_dict()["anger"] == 0.40
    assert emotions.to_dict()["anticipation"] == 1.0

    # 2. Prominent emotions edge selection (default threshold = 0.3)
    prominent = emotions.get_prominent_emotions()

    # 0.29 and 0.299 -> no edge
    assert "fear" not in prominent
    assert "disgust" not in prominent

    # 0.30 -> no edge (strictly > 0.3 required)
    assert "trust" not in prominent
    assert "sadness" not in prominent

    # 0.31 -> edge
    assert "joy" in prominent
    assert prominent["joy"] == 0.31

    # 0.40 -> edge
    assert "anger" in prominent
    assert prominent["anger"] == 0.40

    # 1.0 -> edge
    assert "anticipation" in prominent
    assert prominent["anticipation"] == 1.0
    assert len(prominent) == 3

    # Explicit threshold=0.3 gives identical result
    assert emotions.get_prominent_emotions(threshold=0.3) == prominent


def test_empty_semantic_categories_allowed():
    """Verify that empty arrays for semantic categories without evidence are accepted."""
    fixtures = get_synthetic_fixtures()
    extraction = MoviePlotExtraction.model_validate(fixtures["empty_categories"])
    assert extraction.themes == []
    assert extraction.settings == []
    assert extraction.narrative_devices == []
    assert extraction.protagonist_types == []
    assert extraction.based_on == []
    assert extraction.emotions.sadness == 0.5


def test_duplicate_and_blank_entity_cleaning():
    """Verify that duplicate semantic entities are deduplicated and blanks are removed."""
    fixtures = get_synthetic_fixtures()
    extraction = MoviePlotExtraction.model_validate(fixtures["duplicate_entities"])
    # Original list: ["friendship", "Friendship", "FRIENDSHIP", "jealousy", "   "]
    assert extraction.themes == ["friendship", "jealousy"]


def test_non_string_entity_rejected():
    """Verify that non-string entities raise validation errors."""
    fixtures = get_synthetic_fixtures()
    bad_data = fixtures["valid"].copy()
    bad_data["themes"] = ["friendship", 12345]

    with pytest.raises(ValidationError):
        MoviePlotExtraction.model_validate(bad_data)


def test_missing_emotions_rejected():
    """Verify that omitting any of the 8 Plutchik emotions fails validation."""
    fixtures = get_synthetic_fixtures()
    for emo in PLUTCHIK_EMOTIONS:
        incomplete = fixtures["valid"].copy()
        incomplete_emotions = incomplete["emotions"].copy()
        del incomplete_emotions[emo]
        incomplete["emotions"] = incomplete_emotions

        with pytest.raises(ValidationError):
            MoviePlotExtraction.model_validate(incomplete)


def test_unknown_emotion_rejected():
    """Verify that unknown/hallucinated emotion names are rejected."""
    fixtures = get_synthetic_fixtures()
    bad_data = fixtures["valid"].copy()
    bad_emotions = bad_data["emotions"].copy()
    bad_emotions["nostalgia"] = 0.7  # Not in Plutchik 8
    bad_data["emotions"] = bad_emotions

    with pytest.raises(ValidationError):
        MoviePlotExtraction.model_validate(bad_data)


def test_out_of_range_emotion_scores_rejected():
    """Verify emotion intensity scores outside [0.0, 1.0] are rejected."""
    fixtures = get_synthetic_fixtures()

    # Greater than 1.0
    too_high = fixtures["valid"].copy()
    too_high_emo = too_high["emotions"].copy()
    too_high_emo["joy"] = 1.05
    too_high["emotions"] = too_high_emo
    with pytest.raises(ValidationError):
        MoviePlotExtraction.model_validate(too_high)

    # Less than 0.0
    too_low = fixtures["valid"].copy()
    too_low_emo = too_low["emotions"].copy()
    too_low_emo["anger"] = -0.1
    too_low["emotions"] = too_low_emo
    with pytest.raises(ValidationError):
        MoviePlotExtraction.model_validate(too_low)


def test_unknown_schema_fields_rejected():
    """Verify that unknown root fields or relation types are rejected (extra='forbid')."""
    fixtures = get_synthetic_fixtures()
    bad_data = fixtures["valid"].copy()
    bad_data["genre"] = "Animation"  # Belongs to KG-M, forbidden in KG-L extraction
    bad_data["random_key"] = "test"

    with pytest.raises(ValidationError):
        MoviePlotExtraction.model_validate(bad_data)


def test_extract_json_string_and_fenced_recovery():
    """Verify extraction of raw JSON and markdown-fenced ```json blocks."""
    fixtures = get_synthetic_fixtures()

    # Raw JSON
    raw_json = json.dumps(fixtures["valid"])
    assert extract_json_string(raw_json) == raw_json

    # Fenced JSON
    fenced = fixtures["valid_fenced_json"]
    extracted = extract_json_string(fenced)
    parsed = json.loads(extracted)
    assert parsed["title"] == "Toy Story (1995)"

    # Mixed prose
    mixed = fixtures["valid_mixed_prose"]
    extracted_mixed = extract_json_string(mixed)
    parsed_mixed = json.loads(extracted_mixed)
    assert parsed_mixed["title"] == "Toy Story (1995)"

    # Unrecoverable response
    with pytest.raises(ExtractionParseError):
        extract_json_string("Sorry, I am unable to analyze this movie plot.")


def test_parse_and_validate_extraction_end_to_end():
    """Verify end-to-end response parsing, schema validation, and expected ID alignment."""
    fixtures = get_synthetic_fixtures()

    # Full success
    res = parse_and_validate_extraction(
        fixtures["valid_fenced_json"],
        expected_ids={"movie_id": 1, "movie_idx": 0},
    )
    assert res.movie_id == 1
    assert res.movie_idx == 0

    # ID mismatch rejection
    with pytest.raises(ExtractionValidationError, match="movie_id mismatch"):
        parse_and_validate_extraction(
            fixtures["valid_fenced_json"],
            expected_ids={"movie_id": 999, "movie_idx": 0},
        )


def test_prompt_formatting():
    """Verify system and user prompt generation and formatting for legacy v1.0.0."""
    sys_prompt, user_prompt = format_extraction_prompt(
        title="Inception (2010)",
        overview="A thief who steals corporate secrets through the use of dream-sharing technology.",
        movie_id=79132,
        movie_idx=3500,
        tmdb_id=27205,
        prompt_version="v1.0.0",
    )

    # Check key grounding instructions in system prompt
    assert "CRITICAL RULES" in sys_prompt
    assert "No Hallucination" in sys_prompt
    assert "hasTheme" in sys_prompt
    assert "hasSetting" in sys_prompt
    assert "evokesEmotion" in sys_prompt
    assert "joy, trust, fear" in sys_prompt

    # Check user prompt formatting
    assert "Title: Inception (2010)" in user_prompt
    assert "MovieLens ID: 79132" in user_prompt
    assert "Movie Index: 3500" in user_prompt
    assert "TMDB ID: 27205" in user_prompt
    assert "dream-sharing technology" in user_prompt
    assert '"themes": ["<theme1>", "<theme2>"]' in user_prompt


def test_prompt_version_v1_1_formatting():
    """Verify refined v1.1.0 prompt formatting, calibration, exclusions, and grounding."""
    from src.knowledge_graph.kg_l_contract import (
        DEFAULT_PROMPT_VERSION,
        DEFAULT_SCHEMA_VERSION,
        PROMPT_VERSION_V1_0_0,
        PROMPT_VERSION_V1_1_0,
    )

    # Verify version constants
    assert DEFAULT_PROMPT_VERSION == "v1.1.0"
    assert PROMPT_VERSION_V1_1_0 == "v1.1.0"
    assert PROMPT_VERSION_V1_0_0 == "v1.0.0"
    assert DEFAULT_SCHEMA_VERSION == "v1.0.0"

    # Default call should format with v1.1.0
    sys_prompt, user_prompt = format_extraction_prompt(
        title="Batman Forever (1995)",
        overview="Batman must battle Two-Face and The Riddler with the help of a circus acrobat who becomes Robin.",
        movie_id=153,
        movie_idx=151,
        tmdb_id=414,
        prompt_version="v1.1.0",
    )

    # 1. Calibration guidance
    assert "0.0 = essentially absent / no meaningful evidence" in sys_prompt
    assert "0.1–0.3 = weak" in sys_prompt
    assert "0.4–0.6 = moderate" in sys_prompt
    assert "0.7–0.8 = strong" in sys_prompt
    assert "0.9–1.0 = dominant or extremely intense" in sys_prompt

    # 2. Emotion interpretation rules
    assert "Do NOT require literal emotion words in the overview" in sys_prompt
    assert "Infer likely audience emotions from plot events that are explicitly described" in sys_prompt
    assert "Zero means essentially no evidence that the narrative would evoke that emotion" in sys_prompt
    assert "Multiple emotions may be nonzero simultaneously" in sys_prompt
    assert "Do not artificially force an emotion above 0.5 merely to create a KG edge" in sys_prompt
    assert "Scores must reflect evidence in the supplied overview" in sys_prompt

    # 3. Emotion evidence examples
    assert "threat, danger, pursuit, horror → fear" in sys_prompt
    assert "death, grief, separation, loss → sadness" in sys_prompt
    assert "betrayal, injustice, violent conflict → anger and/or disgust" in sys_prompt
    assert "friendship, romance, reunion, humor, success → joy and/or trust" in sys_prompt
    assert "uncertainty, revelations, escalating conflict → surprise and/or anticipation" in sys_prompt
    assert "these are illustrative examples, not mandatory mappings" in sys_prompt

    # 4. Narrative device definition and exclusions
    assert "time travel, mistaken identity, unreliable narrator, nonlinear narrative, flashback, parallel timelines, frame narrative" in sys_prompt
    assert "It must NOT simply be: an ordinary plot event, a theme, a protagonist type, a conflict, or a character trait" in sys_prompt
    assert "If no genuine narrative device is clearly supported, output []" in sys_prompt
    assert "An empty array [] is preferable to inventing a weak narrative device" in sys_prompt

    # 5. Protagonist types grounding
    assert "Extract character roles or archetypes directly supported by the overview" in sys_prompt
    assert "Do NOT infer occupation, age, identity, or franchise knowledge from outside knowledge" in sys_prompt
    assert "Prefer reusable concise roles (e.g. soldier, student, parent, criminal, scientist, journalist, athlete) ONLY when the overview explicitly supports them" in sys_prompt
    assert "An empty array [] is preferable to an unsupported role" in sys_prompt

    # 6. Strict Grounding
    assert "Rely ONLY on facts, concepts, and narrative events explicitly stated or directly described" in sys_prompt
    assert "Do NOT use outside knowledge about the movie, franchise, source material, characters, actors, or production history" in sys_prompt

    # 7. User prompt formatting
    assert "Title: Batman Forever (1995)" in user_prompt
    assert "MovieLens ID: 153" in user_prompt
    assert "Movie Index: 151" in user_prompt
    assert "TMDB ID: 414" in user_prompt
    assert "circus acrobat who becomes Robin" in user_prompt
    assert '"narrative_devices": ["<device_if_present>"]' in user_prompt
    assert '"protagonist_types": ["<archetype_if_supported>"]' in user_prompt


def test_sample_selection_determinism_and_validity():
    """Verify sample selection determinism and data validity against real Phase 3A artifact."""
    parquet_path = Path("data/processed/ml-1m/metadata/movies_with_tmdb.parquet")
    if not parquet_path.exists():
        pytest.skip("Phase 3A metadata parquet not generated yet")

    df = pd.read_parquet(parquet_path)

    # Run selection twice with same seed
    sample1 = select_sample_movies(df, sample_size=30, random_seed=42)
    sample2 = select_sample_movies(df, sample_size=30, random_seed=42)

    # Assert exact equality
    pd.testing.assert_frame_equal(sample1, sample2)

    # Assert criteria
    assert len(sample1) == 30
    assert sample1["has_tmdb_link"].all()
    assert sample1["has_overview"].all()
    assert (sample1["overview"].str.len() > 10).all()

    # Assert traceability back to source catalog
    source_ids = set(df["movie_id"])
    source_idxs = set(df["movie_idx"])
    for _, row in sample1.iterrows():
        assert int(row["movie_id"]) in source_ids
        assert int(row["movie_idx"]) in source_idxs
