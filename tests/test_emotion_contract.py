"""Unit tests for Phase 3E: Dedicated Evidence-Grounded Emotion Extraction Contract."""

import json
from pydantic import ValidationError
import pytest

from src.knowledge_graph.emotion_contract import (
    DEFAULT_EMOTION_PROMPT_VERSION,
    DEFAULT_EMOTION_SCHEMA_VERSION,
    EMOTION_PROMPT_VERSION_V1_0_0,
    EMOTION_SCHEMA_VERSION_V1_0_0,
    EmotionEvidenceScore,
    MovieEmotionExtraction,
    PlutchikEmotionEvidenceMap,
    format_emotion_prompt,
    parse_and_validate_emotion_extraction,
)
from src.knowledge_graph.kg_l_contract import (
    ExtractionParseError,
    ExtractionValidationError,
    PLUTCHIK_EMOTIONS,
    PlutchikEmotions,
)


def _make_valid_emotion_dict():
    """Helper to generate a valid 8-emotion dictionary with grounded evidence."""
    return {
        "joy": {
            "evidence": ["Woody and Buzz celebrate their reunion with Andy."],
            "score": 0.85,
        },
        "trust": {
            "evidence": ["The toys demonstrate loyalty and solidarity."],
            "score": 0.70,
        },
        "fear": {
            "evidence": ["Toys dread being blown up in Sid's bedroom."],
            "score": 0.60,
        },
        "surprise": {
            "evidence": ["Buzz realizes he cannot actually fly and is a toy."],
            "score": 0.55,
        },
        "sadness": {
            "evidence": ["Woody feels abandoned and replaced."],
            "score": 0.45,
        },
        "disgust": {
            "evidence": [],
            "score": 0.0,
        },
        "anger": {
            "evidence": [],
            "score": 0.0,
        },
        "anticipation": {
            "evidence": ["The climactic chase to catch up with the moving van."],
            "score": 0.80,
        },
    }


def test_emotion_evidence_score_valid():
    """Verify EmotionEvidenceScore validation with valid evidence and scores."""
    # Positive score with evidence
    item1 = EmotionEvidenceScore(evidence=["The character celebrates a victory"], score=0.75)
    assert item1.score == 0.75
    assert item1.evidence == ["The character celebrates a victory"]

    # Boundary score = 1.0 with evidence
    item2 = EmotionEvidenceScore(evidence=["Supreme triumph and euphoria"], score=1.0)
    assert item2.score == 1.0

    # Zero score with empty evidence
    item3 = EmotionEvidenceScore(evidence=[], score=0.0)
    assert item3.score == 0.0
    assert item3.evidence == []

    # Whitespace in evidence list items is stripped
    item4 = EmotionEvidenceScore(evidence=["  Spaced evidence text  "], score=0.5)
    assert item4.evidence == ["Spaced evidence text"]


def test_emotion_evidence_score_consistency_rules():
    """Verify strict consistency: positive score requires evidence, zero score requires empty evidence."""
    # Positive score without evidence -> rejected
    with pytest.raises(ValidationError) as exc_info:
        EmotionEvidenceScore(evidence=[], score=0.5)
    assert "requires at least one supporting evidence" in str(exc_info.value)

    # Positive score with only whitespace evidence -> rejected after stripping
    with pytest.raises(ValidationError) as exc_info:
        EmotionEvidenceScore(evidence=["   ", ""], score=0.5)
    assert "requires at least one supporting evidence" in str(exc_info.value)

    # Zero score with evidence -> rejected
    with pytest.raises(ValidationError) as exc_info:
        EmotionEvidenceScore(evidence=["Some evidence"], score=0.0)
    assert "evidence list must be empty []" in str(exc_info.value)


def test_emotion_evidence_score_bounds():
    """Verify score bounds [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        EmotionEvidenceScore(evidence=["Too low"], score=-0.01)

    with pytest.raises(ValidationError):
        EmotionEvidenceScore(evidence=["Too high"], score=1.01)


def test_plutchik_emotion_evidence_map():
    """Verify PlutchikEmotionEvidenceMap accepts exact 8 emotions and forbids extras."""
    valid_data = _make_valid_emotion_dict()
    emap = PlutchikEmotionEvidenceMap.model_validate(valid_data)

    assert emap.joy.score == 0.85
    assert len(emap.joy.evidence) == 1
    assert emap.disgust.score == 0.0
    assert emap.disgust.evidence == []

    # Derived vector
    vec = emap.to_vector()
    assert isinstance(vec, PlutchikEmotions)
    assert vec.joy == 0.85
    assert vec.disgust == 0.0

    # Evidence dict
    ev_dict = emap.to_evidence_dict()
    assert len(ev_dict) == 8
    assert ev_dict["joy"] == ["Woody and Buzz celebrate their reunion with Andy."]
    assert ev_dict["disgust"] == []

    # Unknown emotion rejected (extra="forbid")
    invalid_data = valid_data.copy()
    invalid_data["nostalgia"] = {"evidence": ["old times"], "score": 0.5}
    with pytest.raises(ValidationError) as exc_info:
        PlutchikEmotionEvidenceMap.model_validate(invalid_data)
    assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()

    # Missing emotion rejected
    incomplete_data = {k: v for k, v in valid_data.items() if k != "anger"}
    with pytest.raises(ValidationError):
        PlutchikEmotionEvidenceMap.model_validate(incomplete_data)


def test_movie_emotion_extraction_schema_and_methods():
    """Verify MovieEmotionExtraction schema validation, edge thresholding, and vector record conversion."""
    data = {
        "movie_idx": 0,
        "movie_id": 1,
        "tmdb_id": 862,
        "title": "Toy Story (1995)",
        "emotions": _make_valid_emotion_dict(),
        "prompt_version": DEFAULT_EMOTION_PROMPT_VERSION,
        "schema_version": DEFAULT_EMOTION_SCHEMA_VERSION,
    }

    extraction = MovieEmotionExtraction.model_validate(data)
    assert extraction.movie_id == 1
    assert extraction.title == "Toy Story (1995)"

    # Prominent emotions threshold test: strictly score > 0.5
    # joy: 0.85 (>0.5), trust: 0.70 (>0.5), fear: 0.60 (>0.5), surprise: 0.55 (>0.5),
    # sadness: 0.45 (<=0.5), disgust: 0.0 (<=0.5), anger: 0.0 (<=0.5), anticipation: 0.80 (>0.5)
    prominent = extraction.get_prominent_emotions(threshold=0.5)
    assert prominent == ["joy", "trust", "fear", "surprise", "anticipation"]
    assert "sadness" not in prominent

    # Boundary test for threshold > 0.5
    custom_dict = _make_valid_emotion_dict()
    custom_dict["sadness"] = {"evidence": ["Slight sorrow"], "score": 0.50}
    custom_dict["fear"] = {"evidence": ["Dread"], "score": 0.51}
    custom_dict["surprise"] = {"evidence": ["Gasp"], "score": 0.49}
    custom_dict["joy"] = {"evidence": [], "score": 0.0}
    custom_dict["trust"] = {"evidence": [], "score": 0.0}
    custom_dict["anticipation"] = {"evidence": [], "score": 0.0}

    extraction_boundary = MovieEmotionExtraction.model_validate({
        "movie_idx": 0,
        "movie_id": 1,
        "tmdb_id": 862,
        "title": "Toy Story (1995)",
        "emotions": custom_dict,
        "prompt_version": DEFAULT_EMOTION_PROMPT_VERSION,
        "schema_version": DEFAULT_EMOTION_SCHEMA_VERSION,
    })
    boundary_prominent = extraction_boundary.get_prominent_emotions(threshold=0.5)
    assert "fear" in boundary_prominent  # 0.51 > 0.50
    assert "sadness" not in boundary_prominent  # 0.50 is NOT > 0.50
    assert "surprise" not in boundary_prominent  # 0.49 is NOT > 0.50

    # Vector record export
    vec_rec = extraction.to_vector_record(threshold=0.5)
    assert vec_rec["movie_id"] == 1
    assert vec_rec["movie_idx"] == 0
    assert vec_rec["title"] == "Toy Story (1995)"
    assert vec_rec["joy"] == 0.85
    assert vec_rec["prominent_emotions"] == ["joy", "trust", "fear", "surprise", "anticipation"]
    assert vec_rec["prompt_version"] == DEFAULT_EMOTION_PROMPT_VERSION
    assert vec_rec["schema_version"] == DEFAULT_EMOTION_SCHEMA_VERSION

    # Unknown top-level fields rejected
    bad_data = data.copy()
    bad_data["themes"] = ["friendship"]
    with pytest.raises(ValidationError):
        MovieEmotionExtraction.model_validate(bad_data)


def test_format_emotion_prompt():
    """Verify emotion prompt formatting contains required steps, calibration text, and versioning."""
    system_prompt, user_prompt = format_emotion_prompt(
        title="Jumanji (1995)",
        overview="When two kids find a magical board game, they release a man trapped in it for decades.",
        movie_id=2,
    )

    # Check system prompt
    assert "Step 1. Identify Plot Evidence" in system_prompt
    assert "Step 2. Assign Intensity Score" in system_prompt
    for emo in ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"):
        assert emo in system_prompt.lower()
    assert "score > 0.0" in system_prompt
    assert 'score is 0.0, "evidence" MUST be empty []' in system_prompt

    # Check user prompt
    assert "Jumanji (1995)" in user_prompt
    assert "MovieLens ID: 2" in user_prompt
    assert "magical board game" in user_prompt
    assert "emotions" in user_prompt
    assert "joy" in user_prompt


def test_parse_and_validate_emotion_extraction_success():
    """Verify successful parsing and validation from raw JSON and markdown-fenced JSON."""
    payload = {
        "movie_idx": 1,
        "movie_id": 2,
        "tmdb_id": 8844,
        "title": "Jumanji (1995)",
        "emotions": _make_valid_emotion_dict(),
        "prompt_version": DEFAULT_EMOTION_PROMPT_VERSION,
        "schema_version": DEFAULT_EMOTION_SCHEMA_VERSION,
    }

    raw_json = json.dumps(payload)
    parsed = parse_and_validate_emotion_extraction(
        raw_text=raw_json,
        expected_movie_id=2,
        expected_movie_idx=1,
        expected_tmdb_id=8844,
    )
    assert parsed.movie_id == 2
    assert parsed.title == "Jumanji (1995)"
    assert parsed.emotions.joy.score == 0.85

    # Markdown fenced
    fenced_text = f"Here is the emotion extraction:\n```json\n{raw_json}\n```\nHope this helps!"
    parsed_fenced = parse_and_validate_emotion_extraction(
        raw_text=fenced_text,
        expected_movie_id=2,
        expected_movie_idx=1,
        expected_tmdb_id=8844,
    )
    assert parsed_fenced.movie_id == 2


def test_parse_and_validate_emotion_extraction_failures():
    """Verify that syntax errors, id mismatches, and schema violations raise appropriate exceptions."""
    payload = {
        "movie_idx": 1,
        "movie_id": 2,
        "tmdb_id": 8844,
        "title": "Jumanji (1995)",
        "emotions": _make_valid_emotion_dict(),
        "prompt_version": DEFAULT_EMOTION_PROMPT_VERSION,
        "schema_version": DEFAULT_EMOTION_SCHEMA_VERSION,
    }

    # Malformed JSON
    with pytest.raises(ExtractionParseError):
        parse_and_validate_emotion_extraction(
            raw_text="This is not JSON at all.",
            expected_movie_id=2,
        )

    # Mismatched movie_id
    with pytest.raises(ExtractionValidationError) as exc_info:
        parse_and_validate_emotion_extraction(
            raw_text=json.dumps(payload),
            expected_movie_id=999,
        )
    assert "movie_id mismatch" in str(exc_info.value)

    # Inconsistent evidence (score > 0 with empty evidence)
    bad_payload = payload.copy()
    bad_emotions = _make_valid_emotion_dict()
    bad_emotions["joy"] = {"evidence": [], "score": 0.8}
    bad_payload["emotions"] = bad_emotions

    with pytest.raises(ExtractionValidationError) as exc_info:
        parse_and_validate_emotion_extraction(
            raw_text=json.dumps(bad_payload),
            expected_movie_id=2,
        )
    assert "requires at least one supporting evidence" in str(exc_info.value)
