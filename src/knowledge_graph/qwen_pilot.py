"""GPU Inference Infrastructure for Qwen2.5-7B-Instruct 30-Movie Pilot (Phase 3D).

This module provides a reproducible, checkpointed, and strictly bounded inference
pipeline for extracting structured plot knowledge graph representations using
Qwen2.5-7B-Instruct on ASU SOL GPU compute via vLLM, with local mocking support for
development and testing on macOS.

Strict Pilot Safety Rule:
- Enforces an absolute maximum limit of 30 movies.
- Refuses to run against the full MovieLens catalog (e.g. movies_with_tmdb.parquet).
- Reuses the Phase 3C contract, ontology, prompt, schema, and validation rules.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import yaml

from .kg_l_contract import (
    DEFAULT_EMOTION_EDGE_THRESHOLD,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SCHEMA_VERSION,
    PLUTCHIK_EMOTIONS,
    ExtractionParseError,
    ExtractionValidationError,
    MoviePlotExtraction,
    format_extraction_prompt,
    parse_and_validate_extraction,
)

logger = logging.getLogger(__name__)

# Absolute upper bound for pilot extractions in Phase 3D
MAX_PILOT_MOVIES: int = 30


# ==============================================================================
# 1. STRICT SAFETY GUARD & SAMPLE LOADER
# ==============================================================================

def print_pilot_banner(movie_count: int, model_id: str, backend: str, input_path: Union[str, Path]) -> None:
    """Print an unmistakable visual banner confirming PILOT MODE is active."""
    banner = (
        "\n"
        "======================================================================\n"
        "*** PILOT MODE ACTIVE: Phase 3D Qwen2.5-7B-Instruct Extraction ***\n"
        f"Sample File : {input_path}\n"
        f"Movie Count : {movie_count} (Hard Safety Limit: <= {MAX_PILOT_MOVIES})\n"
        f"Model ID    : {model_id}\n"
        f"Backend     : {backend}\n"
        "======================================================================\n"
    )
    logger.info(banner)
    print(banner, flush=True)


def load_pilot_movies(
    input_path: Union[str, Path],
    max_allowed: int = MAX_PILOT_MOVIES,
) -> pd.DataFrame:
    """Load and strictly validate the input movie sample for the pilot run.

    Enforces:
    - Path must exist and be readable.
    - Path must NOT point to the full catalog (e.g. movies_with_tmdb.parquet).
    - Dataset must contain between 1 and max_allowed movies (max_allowed <= 30).
    - Required columns must be present.

    Args:
        input_path: Path to sample parquet file.
        max_allowed: Maximum allowed movies (must be <= MAX_PILOT_MOVIES).

    Returns:
        Validated sample DataFrame.

    Raises:
        ValueError: If file is too large, refers to full catalog, or is invalid.
        FileNotFoundError: If input path does not exist.
    """
    if max_allowed > MAX_PILOT_MOVIES:
        raise ValueError(
            f"Safety guard violation: max_allowed ({max_allowed}) exceeds the hard "
            f"pilot upper limit of {MAX_PILOT_MOVIES}. Full catalog extraction is strictly prohibited."
        )

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Pilot sample input file not found: {path}")

    # Check for accidental full catalog paths
    path_str_lower = str(path).lower()
    if "movies_with_tmdb" in path_str_lower or "the-movies-dataset" in path_str_lower:
        raise ValueError(
            f"Safety guard violation: Detected attempt to run against full catalog artifact: {path}. "
            f"Phase 3D pilot mode only accepts the Phase 3C sample (sample_movies.parquet, <= {MAX_PILOT_MOVIES} movies)."
        )

    df = pd.read_parquet(path)

    if len(df) == 0:
        raise ValueError(f"Input sample file is empty: {path}")

    if len(df) > max_allowed:
        raise ValueError(
            f"Safety guard violation: Input dataset contains {len(df)} movies, which exceeds "
            f"the maximum pilot limit of {max_allowed}. Refusing to proceed."
        )

    # Validate required columns
    required_cols = {"movie_id", "movie_idx", "overview"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Sample DataFrame is missing required columns: {missing}")

    # Title column can be ml_title or title
    if "ml_title" not in df.columns and "title" not in df.columns:
        raise ValueError("Sample DataFrame must contain either 'ml_title' or 'title' column.")

    return df


# ==============================================================================
# 2. INFERENCE BACKENDS (LOCAL MOCK & ASU SOL VLLM)
# ==============================================================================

class BaseInferenceBackend(ABC):
    """Abstract interface for LLM inference backends."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response text for given system and user prompts."""
        pass


class MockQwenBackend(BaseInferenceBackend):
    """Deterministic mock Qwen backend for local testing without GPU or model weights.

    Supports simulating valid JSON responses as well as specific failure modes for
    testing retry and error logging behavior.
    """

    def __init__(
        self,
        failure_modes: Optional[Dict[int, str]] = None,
        model_id: str = DEFAULT_MODEL_ID,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ):
        """Initialize mock backend.

        Args:
            failure_modes: Optional mapping of movie_idx to failure mode string:
                - 'malformed_json': Returns broken non-JSON text.
                - 'schema_validation_error': Returns JSON violating the schema.
                - 'id_mismatch': Returns JSON with wrong movie_idx/movie_id.
                - 'retry_then_succeed': Fails on attempt 1, succeeds on attempt 2.
                - 'backend_error': Raises an unexpected runtime exception.
            model_id: Model identifier to embed in metadata.
            prompt_version: Prompt version string to embed in metadata.
        """
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.failure_modes = failure_modes or {}
        self.call_counts: Dict[int, int] = {}

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a synthetic, schema-compliant JSON response or simulated error."""
        # Parse movie identifiers from user prompt
        idx_match = re.search(r"- Movie Index:\s*(\d+)", user_prompt)
        id_match = re.search(r"- MovieLens ID:\s*(\d+)", user_prompt)
        title_match = re.search(r"- Title:\s*(.*)", user_prompt)
        tmdb_match = re.search(r"- TMDB ID:\s*(\d+)", user_prompt)

        movie_idx = int(idx_match.group(1)) if idx_match else 0
        movie_id = int(id_match.group(1)) if id_match else 1
        title = title_match.group(1).strip() if title_match else "Mock Movie"
        tmdb_id = int(tmdb_match.group(1)) if tmdb_match else None

        # Track call count per movie index
        self.call_counts[movie_idx] = self.call_counts.get(movie_idx, 0) + 1
        call_num = self.call_counts[movie_idx]

        fmode = self.failure_modes.get(movie_idx)

        # 1. Simulated backend error
        if fmode == "backend_error":
            raise RuntimeError(f"Simulated CUDA out of memory / backend failure on movie {movie_idx}")

        # 2. Simulated retry then succeed
        if fmode == "retry_then_succeed":
            if call_num == 1:
                return "Malformed output on attempt 1: { incomplete json"
            # Attempt 2 falls through to valid generation below

        # 3. Simulated malformed JSON
        if fmode == "malformed_json":
            return "Sorry, I cannot process this overview: {\"broken\": [unclosed array"

        # 4. Simulated schema validation error (out of range emotion score)
        if fmode == "schema_validation_error":
            bad_data = {
                "movie_idx": movie_idx,
                "movie_id": movie_id,
                "tmdb_id": tmdb_id,
                "title": title,
                "themes": ["action"],
                "settings": ["city"],
                "narrative_devices": [],
                "protagonist_types": [],
                "based_on": [],
                "emotions": {
                    "joy": 99.9,  # Invalid: must be <= 1.0
                    "trust": 0.5,
                    "fear": 0.2,
                    "surprise": 0.3,
                    "sadness": 0.1,
                    "disgust": 0.1,
                    "anger": 0.2,
                    "anticipation": 0.4,
                },
            }
            return json.dumps(bad_data)

        # 5. Simulated ID mismatch
        if fmode == "id_mismatch":
            mismatched_data = {
                "movie_idx": movie_idx + 9999,
                "movie_id": movie_id + 9999,
                "tmdb_id": tmdb_id,
                "title": title,
                "themes": ["mismatch"],
                "settings": [],
                "narrative_devices": [],
                "protagonist_types": [],
                "based_on": [],
                "emotions": {emo: 0.2 for emo in PLUTCHIK_EMOTIONS},
            }
            return json.dumps(mismatched_data)

        # Standard valid generation
        # Deterministically generate realistic continuous emotions with some > 0.5
        title_lower = title.lower()
        is_action_comedy = any(w in title_lower for w in ("action", "batman", "hero", "toy", "night"))

        joy_score = 0.72 if is_action_comedy else 0.35
        anticipation_score = 0.68
        fear_score = 0.58 if "night" in title_lower or "batman" in title_lower else 0.15

        emotions = {
            "joy": round(joy_score, 2),
            "trust": 0.48,
            "fear": round(fear_score, 2),
            "surprise": 0.32,
            "sadness": 0.18,
            "disgust": 0.08,
            "anger": 0.22,
            "anticipation": round(anticipation_score, 2),
        }

        valid_data = {
            "movie_idx": movie_idx,
            "movie_id": movie_id,
            "tmdb_id": tmdb_id,
            "title": title,
            "themes": ["heroism", "justice"] if is_action_comedy else ["conflict", "survival"],
            "settings": ["gotham city"] if "batman" in title_lower else ["urban metropolis"],
            "narrative_devices": ["linear narrative"],
            "protagonist_types": ["vigilante"] if "batman" in title_lower else ["protagonist"],
            "based_on": ["comic book"] if "batman" in title_lower else [],
            "emotions": emotions,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "schema_version": DEFAULT_SCHEMA_VERSION,
        }

        return f"```json\n{json.dumps(valid_data, indent=2)}\n```"


class VLLMQwenBackend(BaseInferenceBackend):
    """vLLM-powered inference backend for Qwen2.5-7B-Instruct on ASU SOL GPU nodes."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
    ):
        """Initialize vLLM engine with Qwen2.5-7B-Instruct.

        Args:
            model_id: Hugging Face model repository ID.
            temperature: Sampling temperature (0.0 for deterministic greedy decoding).
            max_tokens: Maximum tokens in generated completion.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory allocated to vLLM.
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. For GPU execution on ASU SOL, install via: "
                "pip install .[gpu]\n"
                "For local testing on macOS without GPU, run with: --backend mock"
            ) from exc

        self.model_id = model_id
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info(
            f"Loading vLLM model '{model_id}' (tp={tensor_parallel_size}, "
            f"gpu_memory_utilization={gpu_memory_utilization})..."
        )
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM model initialized successfully.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Execute Qwen inference using the official chat template."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.llm, "chat"):
            outputs = self.llm.chat(
                messages,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            return outputs[0].outputs[0].text

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = self.llm.generate(
            prompt,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text


# ==============================================================================
# 3. CHECKPOINTING & RESUMPTION HELPERS
# ==============================================================================

def load_validated_movie_indices(validated_path: Path) -> Set[int]:
    """Read existing validated extractions and return the set of completed movie indices.

    Args:
        validated_path: Path to validated_extractions.jsonl.

    Returns:
        Set of integer movie_idx values already successfully validated.
    """
    completed: Set[int] = set()
    if not validated_path.exists():
        return completed

    with open(validated_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "movie_idx" in record:
                    completed.add(int(record["movie_idx"]))
            except Exception as err:
                logger.warning(f"Malformed line {line_no} in {validated_path}: {err}")
    return completed


def load_validated_extractions(validated_path: Path) -> Dict[int, MoviePlotExtraction]:
    """Load all validated extractions from disk into a dictionary keyed by movie_idx.

    Args:
        validated_path: Path to validated_extractions.jsonl.

    Returns:
        Dict mapping movie_idx to MoviePlotExtraction.
    """
    extractions: Dict[int, MoviePlotExtraction] = {}
    if not validated_path.exists():
        return extractions

    with open(validated_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                extraction = MoviePlotExtraction.model_validate(record)
                extractions[extraction.movie_idx] = extraction
            except Exception as err:
                logger.warning(f"Error reading extraction from {validated_path}: {err}")
    return extractions


def load_failure_indices(failures_path: Path) -> Set[int]:
    """Read existing failures and return the set of failed movie indices.

    Args:
        failures_path: Path to failures.jsonl.

    Returns:
        Set of integer movie_idx values recorded in failures.
    """
    failures: Set[int] = set()
    if not failures_path.exists():
        return failures

    with open(failures_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "movie_idx" in record:
                    failures.add(int(record["movie_idx"]))
            except Exception:
                continue
    return failures


# ==============================================================================
# 4. REVIEW CSV AND SUMMARY WRITERS
# ==============================================================================

def export_pilot_review_csv(
    df_sample: pd.DataFrame,
    validated_extractions: Dict[int, MoviePlotExtraction],
    output_csv_path: Path,
) -> pd.DataFrame:
    """Generate human-readable review CSV flattening extracted fields and emotion scores.

    Columns:
    - movie_idx, movie_id, title
    - themes, settings, narrative_devices, protagonist_types, based_on (semicolon-separated)
    - joy, trust, fear, surprise, sadness, disgust, anger, anticipation (continuous float scores)
    - prominent_emotions (semicolon-separated where score > 0.5)
    - status (SUCCESS / FAILED)

    Args:
        df_sample: Input sample DataFrame.
        validated_extractions: Dict of validated MoviePlotExtraction objects keyed by movie_idx.
        output_csv_path: Destination path for the CSV.

    Returns:
        Flattened review DataFrame.
    """
    rows = []
    # Sort deterministically by movie_idx
    sorted_sample = df_sample.sort_values("movie_idx")

    for _, row in sorted_sample.iterrows():
        movie_idx = int(row["movie_idx"])
        movie_id = int(row["movie_id"])
        title = str(row.get("ml_title") or row.get("title") or "")

        if movie_idx in validated_extractions:
            rec = validated_extractions[movie_idx]
            prominent = rec.emotions.get_prominent_emotions(threshold=DEFAULT_EMOTION_EDGE_THRESHOLD)
            prominent_str = "; ".join(prominent.keys())

            row_dict = {
                "movie_idx": movie_idx,
                "movie_id": movie_id,
                "title": title,
                "themes": "; ".join(rec.themes),
                "settings": "; ".join(rec.settings),
                "narrative_devices": "; ".join(rec.narrative_devices),
                "protagonist_types": "; ".join(rec.protagonist_types),
                "based_on": "; ".join(rec.based_on),
                "joy": rec.emotions.joy,
                "trust": rec.emotions.trust,
                "fear": rec.emotions.fear,
                "surprise": rec.emotions.surprise,
                "sadness": rec.emotions.sadness,
                "disgust": rec.emotions.disgust,
                "anger": rec.emotions.anger,
                "anticipation": rec.emotions.anticipation,
                "prominent_emotions": prominent_str,
                "status": "SUCCESS",
            }
        else:
            row_dict = {
                "movie_idx": movie_idx,
                "movie_id": movie_id,
                "title": title,
                "themes": "",
                "settings": "",
                "narrative_devices": "",
                "protagonist_types": "",
                "based_on": "",
                "joy": 0.0,
                "trust": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "disgust": 0.0,
                "anger": 0.0,
                "anticipation": 0.0,
                "prominent_emotions": "",
                "status": "FAILED",
            }
        rows.append(row_dict)

    df_review = pd.DataFrame(rows)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_review.to_csv(output_csv_path, index=False)
    logger.info(f"Saved pilot review CSV ({len(df_review)} rows) to {output_csv_path}")
    return df_review


# ==============================================================================
# 5. CORE PILOT ORCHESTRATOR
# ==============================================================================

def run_qwen_pilot(
    config_path: Union[str, Path] = "configs/kg_l.yaml",
    backend_type: str = "mock",
    input_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    prompt_version: Optional[str] = None,
    max_movies: int = MAX_PILOT_MOVIES,
    mock_failures: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Execute the Phase 3D Qwen2.5-7B-Instruct 30-movie pilot extraction pipeline.

    Args:
        config_path: Path to configs/kg_l.yaml.
        backend_type: 'mock' for local development or 'vllm' for GPU compute on ASU SOL.
        input_path: Optional override for sample parquet path.
        output_dir: Optional override for output directory.
        prompt_version: Optional prompt version ('v1.0.0' or 'v1.1.0', defaults to config).
        max_movies: Maximum movies to process (strictly <= 30).
        mock_failures: Optional failure injection dictionary for MockQwenBackend.

    Returns:
        Dictionary of run summary metrics.
    """
    pilot_start_time = time.time()

    # Load configuration
    cfg_file = Path(config_path)
    cfg: Dict[str, Any] = {}
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    paths_cfg = cfg.get("paths", {})
    llm_cfg = cfg.get("llm", {})

    # LLM and inference parameters
    model_id = str(llm_cfg.get("model_id", DEFAULT_MODEL_ID))
    resolved_prompt_version = str(prompt_version or llm_cfg.get("prompt_version", DEFAULT_PROMPT_VERSION))
    schema_version = str(llm_cfg.get("schema_version", DEFAULT_SCHEMA_VERSION))
    temperature = float(llm_cfg.get("temperature", 0.0))
    max_tokens = int(llm_cfg.get("max_tokens", 1024))
    tensor_parallel_size = int(llm_cfg.get("tensor_parallel_size", 1))
    gpu_memory_utilization = float(llm_cfg.get("gpu_memory_utilization", 0.90))
    max_retries = int(llm_cfg.get("max_retries", 1))

    # Resolve paths (v1.1.0 defaults to pilot_v1_1 so v1.0.0 is not overwritten)
    resolved_input = Path(input_path or paths_cfg.get("sample_movies_parquet", "data/processed/ml-1m/kg_l/sample_movies.parquet"))
    if output_dir:
        resolved_output_dir = Path(output_dir)
    elif resolved_prompt_version in ("v1.1.0", "v1.1"):
        resolved_output_dir = Path(paths_cfg.get("pilot_v1_1_output_dir", "data/processed/ml-1m/kg_l/pilot_v1_1"))
    else:
        resolved_output_dir = Path(paths_cfg.get("pilot_output_dir", "data/processed/ml-1m/kg_l/pilot"))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    raw_outputs_path = resolved_output_dir / "raw_outputs.jsonl"
    validated_path = resolved_output_dir / "validated_extractions.jsonl"
    failures_path = resolved_output_dir / "failures.jsonl"
    summary_path = resolved_output_dir / "pilot_summary.json"
    review_path = resolved_output_dir / "pilot_review.csv"

    # 1. Enforce strict safety guard
    if max_movies > MAX_PILOT_MOVIES:
        raise ValueError(
            f"Safety guard violation: max_movies ({max_movies}) exceeds the hard "
            f"pilot upper limit of {MAX_PILOT_MOVIES}. Full catalog extraction is strictly prohibited."
        )

    df_sample = load_pilot_movies(resolved_input, max_allowed=MAX_PILOT_MOVIES)
    if len(df_sample) > max_movies:
        df_sample = df_sample.head(max_movies).copy()
    total_movies = len(df_sample)

    # 2. Print prominent pilot mode banner
    print_pilot_banner(total_movies, model_id, backend_type, resolved_input)

    # 3. Initialize backend
    if backend_type == "mock":
        backend: BaseInferenceBackend = MockQwenBackend(
            failure_modes=mock_failures,
            model_id=model_id,
            prompt_version=resolved_prompt_version,
        )
    elif backend_type == "vllm":
        backend = VLLMQwenBackend(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type!r}. Supported backends: 'mock', 'vllm'")

    # 4. Checkpoint resumption
    completed_indices = load_validated_movie_indices(validated_path)
    already_validated_count = sum(1 for idx in df_sample["movie_idx"] if int(idx) in completed_indices)

    if already_validated_count > 0:
        logger.info(
            f"Checkpoint found: {already_validated_count}/{total_movies} movies already validated. "
            f"Skipping previously validated movies."
        )

    total_retries_used = 0
    newly_validated_count = 0
    newly_failed_count = 0

    # 5. Process movies sequentially
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

        system_prompt, user_prompt = format_extraction_prompt(
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

        # Retry loop: attempt 1 + up to max_retries
        for attempt in range(1, 1 + max_retries + 1):
            if attempt > 1:
                total_retries_used += 1
                logger.info(f"Retrying extraction for movie {movie_idx} ('{title}') (attempt {attempt})...")
                # Append strict reminder to user prompt on retry
                current_user_prompt = (
                    f"{user_prompt}\n\n"
                    "CRITICAL RETRY INSTRUCTION: Your previous response was invalid. "
                    "Output ONLY the single valid JSON object strictly matching the schema above. "
                    "Do NOT include explanations, markdown commentary, or extra keys."
                )
            else:
                current_user_prompt = user_prompt

            # Generate from backend
            try:
                raw_response = backend.generate(system_prompt, current_user_prompt)
            except Exception as exc:
                raw_response = f"BACKEND_EXCEPTION: {exc}"
                last_error = exc

            last_raw_response = raw_response

            # Append every raw model attempt to raw_outputs.jsonl
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

            # Validate extraction
            try:
                extraction = parse_and_validate_extraction(
                    raw_response,
                    expected_ids={"movie_id": movie_id, "movie_idx": movie_idx},
                )
                # Success: Append to validated_extractions.jsonl
                with open(validated_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(extraction.model_dump()) + "\n")

                completed_indices.add(movie_idx)
                newly_validated_count += 1
                extraction_succeeded = True
                logger.info(f"Successfully validated extraction for movie {movie_idx} ('{title}') on attempt {attempt}.")
                break
            except (ExtractionParseError, ExtractionValidationError, Exception) as err:
                last_error = err
                if attempt <= max_retries:
                    logger.warning(
                        f"Attempt {attempt}/{1 + max_retries} failed for movie {movie_idx} ('{title}'): {err}"
                    )

        # If all attempts failed, record in failures.jsonl
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
                "error_message": str(last_error),
                "last_raw_response": last_raw_response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(failures_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(failure_record) + "\n")
            logger.error(
                f"Movie {movie_idx} ('{title}') failed extraction after {attempt} attempts: "
                f"[{error_type}] {last_error}"
            )

    # 6. Post-processing: Cumulative state across all attempts & past runs
    all_validated = load_validated_extractions(validated_path)
    all_failed_indices = load_failure_indices(failures_path)

    sample_movie_indices = set(df_sample["movie_idx"])
    cumulative_validated_count = sum(1 for idx in sample_movie_indices if idx in all_validated)
    # Movies that failed and have not subsequently succeeded
    cumulative_failed_count = sum(1 for idx in sample_movie_indices if idx in all_failed_indices and idx not in all_validated)

    total_runtime_seconds = round(time.time() - pilot_start_time, 2)
    avg_runtime_per_movie = round(total_runtime_seconds / max(1, total_movies), 3)
    success_rate = round((cumulative_validated_count / max(1, total_movies)) * 100.0, 2)

    # 7. Write pilot_summary.json
    summary_data = {
        "model_id": model_id,
        "prompt_version": resolved_prompt_version,
        "schema_version": schema_version,
        "backend": backend_type,
        "total_movies_in_sample": total_movies,
        "successfully_validated_count": cumulative_validated_count,
        "failed_count": cumulative_failed_count,
        "total_retries_used": total_retries_used,
        "average_runtime_per_movie_seconds": avg_runtime_per_movie,
        "total_pilot_runtime_seconds": total_runtime_seconds,
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
    logger.info(f"Saved pilot summary to {summary_path}")

    # 8. Export pilot_review.csv
    export_pilot_review_csv(
        df_sample=df_sample,
        validated_extractions=all_validated,
        output_csv_path=review_path,
    )

    return summary_data
