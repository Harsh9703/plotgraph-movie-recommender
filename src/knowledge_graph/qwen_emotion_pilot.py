"""GPU Inference Infrastructure for Phase 3E Dedicated Emotion Extraction Pilot.

Provides a reproducible, checkpointed, and strictly bounded inference pipeline for
extracting evidence-grounded Plutchik emotion evaluations using Qwen2.5-7B-Instruct
on ASU SOL GPU compute via vLLM, with local mocking support for development and testing.

Strict Safety Rules:
- Enforces an absolute maximum limit of 30 movies.
- Refuses to run against the full catalog.
- Evaluates the exact same deterministic 30-movie sample from sample_movies.parquet.
- Writes to isolated generated-data directory (emotion_pilot_v1/), never overwriting
  semantic pilot directories (pilot/, pilot_v1_1/).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import yaml

from .emotion_contract import (
    DEFAULT_EMOTION_PROMPT_VERSION,
    DEFAULT_EMOTION_SCHEMA_VERSION,
    MovieEmotionExtraction,
    format_emotion_prompt,
    parse_and_validate_emotion_extraction,
)
from .kg_l_contract import (
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    DEFAULT_MODEL_ID,
    PLUTCHIK_EMOTIONS,
    ExtractionParseError,
    ExtractionValidationError,
)
from .qwen_pilot import (
    MAX_PILOT_MOVIES,
    BaseInferenceBackend,
    VLLMQwenBackend,
    load_failure_indices,
    load_pilot_movies,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. BANNER & LOGGING
# ==============================================================================

def print_emotion_pilot_banner(
    movie_count: int,
    model_id: str,
    backend: str,
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> None:
    """Print visual banner confirming Phase 3E Emotion Pilot is active."""
    banner = (
        "\n"
        "======================================================================\n"
        "*** PHASE 3E DEDICATED EMOTION PILOT (Qwen2.5-7B-Instruct) ***\n"
        f"Sample File : {input_path}\n"
        f"Movie Count : {movie_count} (Hard Safety Limit: <= {MAX_PILOT_MOVIES})\n"
        f"Model ID    : {model_id}\n"
        f"Backend     : {backend}\n"
        f"Output Dir  : {output_dir}\n"
        "======================================================================\n"
    )
    logger.info(banner)
    print(banner, flush=True)


# ==============================================================================
# 2. MOCK EMOTION BACKEND (FOR LOCAL DEVELOPMENT & UNIT TESTS)
# ==============================================================================

class MockQwenEmotionBackend(BaseInferenceBackend):
    """Deterministic mock Qwen backend for evidence-grounded emotion extraction."""

    def __init__(
        self,
        failure_modes: Optional[Dict[int, str]] = None,
        failure_mode: Optional[str] = None,
        model_id: str = DEFAULT_MODEL_ID,
        prompt_version: str = DEFAULT_EMOTION_PROMPT_VERSION,
    ):
        """Initialize mock emotion backend.

        Args:
            failure_modes: Mapping of movie_idx to simulated failure modes:
                - 'malformed_json': Non-JSON string.
                - 'schema_validation_error': Schema violation (e.g. invalid score > 1.0).
                - 'id_mismatch': Wrong movie_id/movie_idx.
                - 'retry_then_succeed': Fails on attempt 1, succeeds on attempt 2.
                - 'backend_error': Raises unexpected RuntimeError.
                - 'score_without_evidence': score > 0 with evidence=[].
                - 'evidence_with_zero_score': score == 0 with evidence=["something"].
            failure_mode: Optional default failure mode applied to all movies.
            model_id: Model identifier.
            prompt_version: Emotion prompt version.
        """
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.failure_modes = failure_modes or {}
        self.default_failure_mode = failure_mode
        self.call_counts: Dict[int, int] = {}

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        """Generate synthetic, evidence-grounded JSON response or simulated failure."""
        idx_match = re.search(r"- Movie Index:\s*(\d+)", user_prompt)
        id_match = re.search(r"- MovieLens ID:\s*(\d+)", user_prompt)
        title_match = re.search(r"- Title:\s*(.*)", user_prompt)
        tmdb_match = re.search(r"- TMDB ID:\s*(\d+)", user_prompt)
        overview_match = re.search(r'Plot Overview:\s*"""(.*?)"""', user_prompt, re.DOTALL)

        movie_idx = kwargs.get("movie_idx") if kwargs.get("movie_idx") is not None else (int(idx_match.group(1)) if idx_match else 0)
        movie_id = kwargs.get("movie_id") if kwargs.get("movie_id") is not None else (int(id_match.group(1)) if id_match else 1)
        title = kwargs.get("title") or (title_match.group(1).strip() if title_match else "Mock Movie")
        tmdb_id = kwargs.get("tmdb_id") if "tmdb_id" in kwargs else (int(tmdb_match.group(1)) if tmdb_match else None)
        overview = kwargs.get("overview") or (overview_match.group(1).strip() if overview_match else "")

        self.call_counts[movie_idx] = self.call_counts.get(movie_idx, 0) + 1
        call_num = self.call_counts[movie_idx]

        fmode = self.failure_modes.get(movie_idx, self.default_failure_mode)

        # 1. Simulated backend error
        if fmode == "backend_error":
            raise RuntimeError(f"Simulated CUDA backend error for movie {movie_idx}")

        # 2. Simulated retry then succeed
        if fmode in ("retry_then_succeed", "retry_schema_then_succeed"):
            if call_num == 1:
                if fmode == "retry_schema_then_succeed":
                    # Exact schema violation from real Phase 3E pilot: score == 0.0 with evidence
                    real_failure_data = {
                        "movie_idx": movie_idx,
                        "movie_id": movie_id,
                        "tmdb_id": tmdb_id,
                        "title": title,
                        "emotions": {
                            emo: (
                                {"evidence": ["discovered by a fashion photographer"], "score": 0.0}
                                if emo == "trust"
                                else (
                                    {"evidence": ["dangerous conflict described in plot"], "score": 0.65}
                                    if emo == "fear"
                                    else {"evidence": [], "score": 0.0}
                                )
                            )
                            for emo in PLUTCHIK_EMOTIONS
                        },
                        "model_id": self.model_id,
                        "prompt_version": self.prompt_version,
                        "schema_version": DEFAULT_EMOTION_SCHEMA_VERSION,
                    }
                    return json.dumps(real_failure_data)
                return "Malformed JSON attempt 1: { incomplete json..."

        # 3. Simulated malformed JSON
        if fmode == "malformed_json":
            return "Non-JSON response: {\"unclosed\": [\"evidence\""

        # 4. Simulated schema error (score > 1.0)
        if fmode == "schema_validation_error":
            bad_data = {
                "movie_idx": movie_idx,
                "movie_id": movie_id,
                "tmdb_id": tmdb_id,
                "title": title,
                "emotions": {emo: {"evidence": ["some evidence"], "score": 1.5} for emo in PLUTCHIK_EMOTIONS},
            }
            return json.dumps(bad_data)

        # 5. Simulated ID mismatch
        if fmode == "id_mismatch":
            mismatch_data = {
                "movie_idx": movie_idx + 9999,
                "movie_id": movie_id + 9999,
                "tmdb_id": tmdb_id,
                "title": title,
                "emotions": {emo: {"evidence": [], "score": 0.0} for emo in PLUTCHIK_EMOTIONS},
            }
            return json.dumps(mismatch_data)

        # 6. Simulated consistency errors
        if fmode == "score_without_evidence":
            bad_consistency = {
                "movie_idx": movie_idx,
                "movie_id": movie_id,
                "tmdb_id": tmdb_id,
                "title": title,
                "emotions": {
                    emo: (
                        {"evidence": [], "score": 0.8}
                        if emo == "fear"
                        else {"evidence": [], "score": 0.0}
                    )
                    for emo in PLUTCHIK_EMOTIONS
                },
            }
            return json.dumps(bad_consistency)

        if fmode == "evidence_with_zero_score":
            bad_zero = {
                "movie_idx": movie_idx,
                "movie_id": movie_id,
                "tmdb_id": tmdb_id,
                "title": title,
                "emotions": {
                    emo: (
                        {"evidence": ["danger present"], "score": 0.0}
                        if emo == "fear"
                        else {"evidence": [], "score": 0.0}
                    )
                    for emo in PLUTCHIK_EMOTIONS
                },
            }
            return json.dumps(bad_zero)

        # Standard deterministic generation grounded in overview keywords
        overview_lower = overview.lower()
        title_lower = title.lower()
        comb_text = f"{title_lower} {overview_lower}"

        emotions_payload: Dict[str, Dict[str, Any]] = {}

        # Fear
        if any(w in comb_text for w in ("threat", "danger", "horror", "kill", "pursuit", "terror", "night", "batman", "war")):
            emotions_payload["fear"] = {
                "evidence": ["dangerous conflict and physical threats described in plot"],
                "score": 0.65,
            }
        else:
            emotions_payload["fear"] = {"evidence": [], "score": 0.0}

        # Sadness
        if any(w in comb_text for w in ("death", "grief", "loss", "tragic", "separation", "die")):
            emotions_payload["sadness"] = {
                "evidence": ["tragic loss or personal grief evident in overview"],
                "score": 0.55,
            }
        else:
            emotions_payload["sadness"] = {"evidence": [], "score": 0.0}

        # Anger
        if any(w in comb_text for w in ("revenge", "injustice", "betrayal", "villain", "fight", "battle")):
            emotions_payload["anger"] = {
                "evidence": ["hostile confrontation and desire for vengeance"],
                "score": 0.55,
            }
        else:
            emotions_payload["anger"] = {"evidence": [], "score": 0.0}

        # Disgust
        if any(w in comb_text for w in ("cruel", "corruption", "grotesque", "crime", "menace")):
            emotions_payload["disgust"] = {
                "evidence": ["depraved actions or oppressive criminal environment"],
                "score": 0.45,
            }
        else:
            emotions_payload["disgust"] = {"evidence": [], "score": 0.0}

        # Joy
        if any(w in comb_text for w in ("friend", "humor", "romance", "funny", "happy", "success", "toy", "triumph")):
            emotions_payload["joy"] = {
                "evidence": ["lighthearted camaraderie, humor, or affection"],
                "score": 0.60,
            }
        else:
            emotions_payload["joy"] = {"evidence": [], "score": 0.0}

        # Trust
        if any(w in comb_text for w in ("help", "partner", "alliance", "loyal", "together", "robin")):
            emotions_payload["trust"] = {
                "evidence": ["mutual loyalty, partnership, or cooperative alliance"],
                "score": 0.55,
            }
        else:
            emotions_payload["trust"] = {"evidence": [], "score": 0.0}

        # Surprise
        if any(w in comb_text for w in ("unexpected", "secret", "twist", "reveal", "discover")):
            emotions_payload["surprise"] = {
                "evidence": ["unexpected plot discovery or dramatic revelation"],
                "score": 0.50,
            }
        else:
            emotions_payload["surprise"] = {"evidence": [], "score": 0.0}

        # Anticipation (almost all movie plots contain an impending goal or struggle)
        emotions_payload["anticipation"] = {
            "evidence": ["unfolding mission, struggle, and upcoming confrontation"],
            "score": 0.70,
        }

        valid_data = {
            "movie_idx": movie_idx,
            "movie_id": movie_id,
            "tmdb_id": tmdb_id,
            "title": title,
            "emotions": emotions_payload,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "schema_version": DEFAULT_EMOTION_SCHEMA_VERSION,
        }

        return f"```json\n{json.dumps(valid_data, indent=2)}\n```"


# ==============================================================================
# 3. CHECKPOINTING & REVIEW CSV WRITERS
# ==============================================================================

def load_validated_emotion_extractions(validated_path: Path) -> Dict[int, MovieEmotionExtraction]:
    """Load validated emotion extractions from disk into a dictionary keyed by movie_idx."""
    extractions: Dict[int, MovieEmotionExtraction] = {}
    if not validated_path.exists():
        return extractions

    with open(validated_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ext = MovieEmotionExtraction.model_validate(record)
                extractions[ext.movie_idx] = ext
            except Exception as err:
                logger.warning(f"Error loading emotion extraction: {err}")
    return extractions


def export_emotion_review_csv(
    df_sample: pd.DataFrame,
    validated_extractions: Dict[int, MovieEmotionExtraction],
    output_csv_path: Path,
) -> pd.DataFrame:
    """Export human-readable CSV with plot overview, evidence items, and scores for each emotion.

    Columns:
    - movie_idx, movie_id, title, overview
    - for each of 8 emotions:
      - {emotion}_evidence (semicolon-separated)
      - {emotion}_score (float)
    - prominent_emotions (semicolon-separated where score > 0.3)
    - status (SUCCESS / FAILED)
    """
    rows = []
    sorted_sample = df_sample.sort_values("movie_idx")

    for _, row in sorted_sample.iterrows():
        movie_idx = int(row["movie_idx"])
        movie_id = int(row["movie_id"])
        title = str(row.get("ml_title") or row.get("title") or "")
        overview = str(row.get("overview") or "")

        row_dict: Dict[str, Any] = {
            "movie_idx": movie_idx,
            "movie_id": movie_id,
            "title": title,
            "overview": overview,
        }

        if movie_idx in validated_extractions:
            rec = validated_extractions[movie_idx]
            for emo in PLUTCHIK_EMOTIONS:
                item = getattr(rec.emotions, emo)
                row_dict[f"{emo}_evidence"] = "; ".join(item.evidence)
                row_dict[f"{emo}_score"] = item.score

            prominent = rec.get_prominent_emotions(threshold=DEFAULT_EMOTION_EDGE_THRESHOLD)
            row_dict["prominent_emotions"] = "; ".join(prominent)
            row_dict["status"] = "SUCCESS"
        else:
            for emo in PLUTCHIK_EMOTIONS:
                row_dict[f"{emo}_evidence"] = ""
                row_dict[f"{emo}_score"] = 0.0
            row_dict["prominent_emotions"] = ""
            row_dict["status"] = "FAILED"

        rows.append(row_dict)

    df_review = pd.DataFrame(rows)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_review.to_csv(output_csv_path, index=False)
    logger.info(f"Saved emotion review CSV ({len(df_review)} rows) to {output_csv_path}")
    return df_review


# ==============================================================================
# 4. CONFIGURATION & RETRY RESOLUTION
# ==============================================================================

def validate_and_resolve_max_retries(
    explicit_retries: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> int:
    """Validate and resolve max_retries for the emotion pilot.

    Priority:
    1. explicit_retries if provided (not None)
    2. cfg['emotion_llm']['max_retries'] if present and not None
    3. cfg['llm']['max_retries'] if present and not None
    4. Default: 1

    Validation Rules:
    - Must be an integer >= 0.
    - If missing or null, safely defaults to 1.
    - Rejects booleans (e.g. True/False), strings, floats, negative values, and non-numeric objects.

    Returns:
        Validated integer retry count >= 0.

    Raises:
        ValueError: If value is not an integer or is negative.
    """
    raw_val: Any = None

    if explicit_retries is not None:
        raw_val = explicit_retries
    elif cfg is not None:
        emotion_cfg = cfg.get("emotion_llm")
        if isinstance(emotion_cfg, dict) and "max_retries" in emotion_cfg and emotion_cfg["max_retries"] is not None:
            raw_val = emotion_cfg["max_retries"]
        else:
            general_cfg = cfg.get("llm")
            if isinstance(general_cfg, dict) and "max_retries" in general_cfg and general_cfg["max_retries"] is not None:
                raw_val = general_cfg["max_retries"]

    # Safe default when missing or null
    if raw_val is None:
        return 1

    # Python bool inherits from int (isinstance(True, int) is True), so check explicitly
    if isinstance(raw_val, bool):
        raise ValueError(f"max_retries must be an integer >= 0, got boolean: {raw_val!r}")

    # Must be integer type
    if not isinstance(raw_val, int):
        raise ValueError(
            f"max_retries must be an integer, got {type(raw_val).__name__}: {raw_val!r}"
        )

    # Must be non-negative
    if raw_val < 0:
        raise ValueError(f"max_retries must be an integer >= 0, got negative value: {raw_val}")

    return raw_val


# ==============================================================================
# 5. CORE EMOTION PILOT ORCHESTRATOR
# ==============================================================================

def run_qwen_emotion_pilot(
    config_path: Union[str, Path] = "configs/kg_l.yaml",
    backend_type: str = "mock",
    input_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    prompt_version: Optional[str] = None,
    max_movies: int = MAX_PILOT_MOVIES,
    mock_failures: Optional[Dict[int, str]] = None,
    backend: Optional[BaseInferenceBackend] = None,
    input_sample_path: Optional[Union[str, Path]] = None,
    max_retries: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute the Phase 3E dedicated Qwen2.5-7B-Instruct emotion pilot pipeline.

    Args:
        config_path: Path to configs/kg_l.yaml.
        backend_type: 'mock' for CPU/testing or 'vllm' for GPU compute on ASU SOL.
        input_path: Optional path to sample parquet (default: from config).
        output_dir: Optional output directory (default: data/processed/ml-1m/kg_l/emotion_pilot_v1).
        prompt_version: Optional emotion prompt version (default: v1.0.0).
        max_movies: Maximum movies to process (hard limit <= 30).
        mock_failures: Optional failure injection dictionary for MockQwenEmotionBackend.
        backend: Optional initialized backend instance (overrides backend_type).
        input_sample_path: Alias for input_path.
        max_retries: Optional override for max retries per movie.

    Returns:
        Dictionary of pilot run summary metrics.
    """
    pilot_start_time = time.time()

    # Load configuration
    cfg_file = Path(config_path)
    cfg: Dict[str, Any] = {}
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    paths_cfg = cfg.get("paths", {})
    emotion_llm_cfg = cfg.get("emotion_llm", {})
    general_llm_cfg = cfg.get("llm", {})

    # Validate and resolve max_retries before inference begins
    max_retries = validate_and_resolve_max_retries(explicit_retries=max_retries, cfg=cfg)

    model_id = str(emotion_llm_cfg.get("model_id") or general_llm_cfg.get("model_id", DEFAULT_MODEL_ID))
    resolved_prompt_version = str(
        prompt_version
        or emotion_llm_cfg.get("emotion_prompt_version")
        or DEFAULT_EMOTION_PROMPT_VERSION
    )
    schema_version = str(
        emotion_llm_cfg.get("emotion_schema_version")
        or DEFAULT_EMOTION_SCHEMA_VERSION
    )
    temperature = float(emotion_llm_cfg.get("temperature", 0.0))
    max_tokens = int(emotion_llm_cfg.get("max_tokens", 1536))
    tensor_parallel_size = int(emotion_llm_cfg.get("tensor_parallel_size", 1))
    gpu_memory_utilization = float(emotion_llm_cfg.get("gpu_memory_utilization", 0.90))

    # Resolve output directory (isolated from semantic pilot dirs)
    raw_input_path = input_sample_path or input_path
    resolved_input = Path(raw_input_path or paths_cfg.get("sample_movies_parquet", "data/processed/ml-1m/kg_l/sample_movies.parquet"))
    resolved_output_dir = Path(output_dir or paths_cfg.get("emotion_pilot_output_dir", "data/processed/ml-1m/kg_l/emotion_pilot_v1"))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    raw_outputs_path = resolved_output_dir / "raw_outputs.jsonl"
    validated_evidence_path = resolved_output_dir / "validated_emotion_evidence.jsonl"
    emotion_vectors_path = resolved_output_dir / "emotion_vectors.jsonl"
    failures_path = resolved_output_dir / "failures.jsonl"
    summary_path = resolved_output_dir / "pilot_summary.json"
    review_path = resolved_output_dir / "emotion_review.csv"

    # Enforce strict safety guard
    if max_movies > MAX_PILOT_MOVIES:
        raise ValueError(
            f"Safety guard violation: max_movies ({max_movies}) exceeds hard limit of {MAX_PILOT_MOVIES}."
        )

    df_sample = load_pilot_movies(resolved_input, max_allowed=MAX_PILOT_MOVIES)
    if len(df_sample) > max_movies:
        df_sample = df_sample.head(max_movies).copy()
    total_movies = len(df_sample)

    effective_backend_name = backend.__class__.__name__ if backend is not None else backend_type
    # Print prominent banner
    print_emotion_pilot_banner(total_movies, model_id, effective_backend_name, resolved_input, resolved_output_dir)

    # Initialize backend
    if backend is not None:
        active_backend: BaseInferenceBackend = backend
    elif backend_type == "mock":
        active_backend = MockQwenEmotionBackend(
            failure_modes=mock_failures,
            model_id=model_id,
            prompt_version=resolved_prompt_version,
        )
    elif backend_type == "vllm":
        active_backend = VLLMQwenBackend(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type!r}. Supported: 'mock', 'vllm'")

    # Checkpoint resumption: read validated_emotion_evidence.jsonl
    existing_validated = load_validated_emotion_extractions(validated_evidence_path)
    completed_indices: Set[int] = set(existing_validated.keys())
    already_validated_count = sum(1 for idx in df_sample["movie_idx"] if int(idx) in completed_indices)

    if already_validated_count > 0:
        logger.info(
            f"Checkpoint found: {already_validated_count}/{total_movies} movies already validated. Skipping."
        )

    total_retries_used = 0
    newly_validated_count = 0
    newly_failed_count = 0

    # Process movies sequentially
    for row in df_sample.itertuples():
        movie_idx = int(row.movie_idx)
        movie_id = int(row.movie_id)
        title = str(getattr(row, "ml_title", None) or getattr(row, "title", None) or "")
        overview = str(getattr(row, "overview", "") or "")
        tmdb_id_val = getattr(row, "tmdb_id", None)
        tmdb_id = int(tmdb_id_val) if pd.notna(tmdb_id_val) else None

        if movie_idx in completed_indices:
            logger.debug(f"[CHECKPOINT] Movie {movie_idx} ('{title}') already validated. Skipping.")
            continue

        system_prompt, user_prompt = format_emotion_prompt(
            title=title,
            overview=overview,
            movie_id=movie_id,
            movie_idx=movie_idx,
            tmdb_id=tmdb_id,
            prompt_version=resolved_prompt_version,
        )

        extraction_succeeded = False
        last_error: Optional[Exception] = None
        last_raw_response: str = ""

        for attempt in range(1, 1 + max_retries + 1):
            if attempt > 1:
                total_retries_used += 1
                logger.info(f"Retrying emotion extraction for movie {movie_idx} ('{title}') (attempt {attempt})...")
                current_user_prompt = (
                    f"{user_prompt}\n\n"
                    "CRITICAL RETRY INSTRUCTION: Your previous response was invalid. "
                    "Output ONLY a single valid JSON object strictly matching the schema above. "
                    "Ensure: if score > 0, evidence MUST not be empty; if score == 0, evidence MUST be empty []."
                )
            else:
                current_user_prompt = user_prompt

            try:
                raw_response = active_backend.generate(system_prompt, current_user_prompt)
            except Exception as exc:
                raw_response = f"BACKEND_EXCEPTION: {exc}"
                last_error = exc

            last_raw_response = raw_response

            # Append to raw_outputs.jsonl
            raw_record = {
                "movie_idx": movie_idx,
                "movie_id": movie_id,
                "title": title,
                "attempt": attempt,
                "raw_response": raw_response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(raw_outputs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(raw_record) + "\n")

            # Parse and validate
            try:
                extraction = parse_and_validate_emotion_extraction(
                    raw_response,
                    expected_ids={"movie_id": movie_id, "movie_idx": movie_idx},
                )
                # Success: Append to validated_emotion_evidence.jsonl
                with open(validated_evidence_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(extraction.model_dump()) + "\n")

                # Derive and append to emotion_vectors.jsonl
                vector_record = extraction.to_vector_record()
                with open(emotion_vectors_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(vector_record) + "\n")

                completed_indices.add(movie_idx)
                newly_validated_count += 1
                extraction_succeeded = True
                logger.info(f"Successfully validated emotion extraction for movie {movie_idx} ('{title}').")
                break
            except (ExtractionParseError, ExtractionValidationError, Exception) as err:
                last_error = err
                if attempt <= max_retries:
                    logger.warning(
                        f"Attempt {attempt}/{1 + max_retries} failed for movie {movie_idx} ('{title}'): {err}"
                    )

        if not extraction_succeeded:
            newly_failed_count += 1
            if isinstance(last_error, ExtractionParseError):
                error_type = "json_syntax_error"
            elif isinstance(last_error, ExtractionValidationError):
                if "mismatch" in str(last_error).lower():
                    error_type = "id_mismatch"
                else:
                    error_type = "schema_validation_error"
            else:
                error_type = "backend_error"

            failure_record = {
                "movie_idx": movie_idx,
                "movie_id": movie_id,
                "title": title,
                "attempts": attempt,
                "error_type": error_type,
                "error_category": error_type,
                "error_message": str(last_error),
                "last_raw_response": last_raw_response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(failures_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(failure_record) + "\n")
            logger.error(f"Movie {movie_idx} ('{title}') failed emotion extraction: [{error_type}] {last_error}")

    # Post-processing: Cumulative state across runs
    all_validated = load_validated_emotion_extractions(validated_evidence_path)
    all_failed_indices = load_failure_indices(failures_path)

    sample_movie_indices = set(df_sample["movie_idx"])
    cumulative_validated_count = sum(1 for idx in sample_movie_indices if idx in all_validated)
    cumulative_failed_count = sum(
        1 for idx in sample_movie_indices if idx in all_failed_indices and idx not in all_validated
    )

    total_runtime_seconds = round(time.time() - pilot_start_time, 2)
    avg_runtime_per_movie = round(total_runtime_seconds / max(1, total_movies), 3)
    success_rate = round((cumulative_validated_count / max(1, total_movies)) * 100.0, 2)

    # Write pilot_summary.json
    summary_data = {
        "model_id": model_id,
        "emotion_prompt_version": resolved_prompt_version,
        "emotion_schema_version": schema_version,
        "backend": effective_backend_name,
        "total_movies": total_movies,
        "total_movies_in_sample": total_movies,
        "successful_movies": cumulative_validated_count,
        "successfully_validated_count": cumulative_validated_count,
        "failed_movies": cumulative_failed_count,
        "failed_count": cumulative_failed_count,
        "retries_attempted": total_retries_used,
        "total_retries_used": total_retries_used,
        "skipped_already_validated": already_validated_count,
        "average_runtime_per_movie_seconds": avg_runtime_per_movie,
        "total_pilot_runtime_seconds": total_runtime_seconds,
        "schema_success_rate": success_rate,
        "success_rate": success_rate,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_retries": max_retries,
            "max_pilot_movies": MAX_PILOT_MOVIES,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved emotion pilot summary to {summary_path}")

    # Export emotion_review.csv
    export_emotion_review_csv(
        df_sample=df_sample,
        validated_extractions=all_validated,
        output_csv_path=review_path,
    )

    return summary_data
