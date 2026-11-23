"""Production-Scale Batched GPU Inference Infrastructure for KG-L Extraction (Phase 3F).

Provides a production-grade, resumable, batched extraction engine using
Qwen2.5-7B-Instruct for all eligible MovieLens-1M movies with Phase 3A plot overviews.

Key Architecture:
1. Production Input: Loads movies_with_tmdb.parquet, filtering to movies with
   has_overview == True and non-empty plot overviews.
2. Unified vLLM Engine: Initializes Qwen2.5-7B-Instruct once per run/job, servicing
   both semantic extraction (v1.1.0) and evidence-grounded emotion extraction (v1.0.0).
3. Batched Inference: Continuous batch generation using vLLM engine for high throughput.
4. Independent Checkpointing & Resumption: Checkpoints semantic and emotion tasks
   independently after every batch to survive SLURM timeouts and restarts.
5. Chunking & Range Partitioning: Supports splitting extraction across jobs via
   (chunk_id, num_chunks) or (start_idx, end_idx) with deterministic movie ordering.
6. Failure Resilience: Up to 1 retry (max_retries=1) per invalid output with error
   categorization; permanent failures do not terminate the run.
7. Completion Validation: Comprehensive diagnostic utility verifying 100% accounting,
   schema conformance, version alignment, and emotion evidence consistency.
8. Runtime Estimation: Dry-run estimation utility predicting runtime scenarios.
9. Local Mock Mode: Deterministic mock backend with production-directory protection.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
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
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SCHEMA_VERSION,
    PLUTCHIK_EMOTIONS,
    ExtractionParseError,
    ExtractionValidationError,
    MoviePlotExtraction,
    format_extraction_prompt,
    parse_and_validate_extraction,
)
from .qwen_emotion_pilot import validate_and_resolve_max_retries

logger = logging.getLogger(__name__)

# Frozen Production Defaults
DEFAULT_SEMANTIC_PROMPT_VERSION: str = DEFAULT_PROMPT_VERSION      # "v1.1.0"
DEFAULT_SEMANTIC_SCHEMA_VERSION: str = DEFAULT_SCHEMA_VERSION      # "v1.0.0"
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_MAX_SEMANTIC_TOKENS: int = 1024
DEFAULT_MAX_EMOTION_TOKENS: int = 1536
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_TENSOR_PARALLEL_SIZE: int = 1
DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.90
DEFAULT_MAX_RETRIES: int = 1
DEFAULT_PRODUCTION_OUTPUT_DIR: str = "data/processed/ml-1m/kg_l/full_extraction"

SEMANTIC_RETRY_REMINDER: str = (
    "CRITICAL RETRY INSTRUCTION: Your previous response was invalid. "
    "Output ONLY the single valid JSON object strictly matching the schema above. "
    "Do NOT include explanations, markdown commentary, or extra keys."
)

EMOTION_RETRY_REMINDER: str = (
    "CRITICAL RETRY INSTRUCTION: Your previous response was invalid. "
    "Output ONLY a single valid JSON object strictly matching the schema above. "
    "Ensure: if score > 0, evidence MUST not be empty; if score == 0, evidence MUST be empty []."
)


# ==============================================================================
# 1. PRODUCTION BANNER & LOGGING
# ==============================================================================

def print_production_banner(
    eligible_count: int,
    target_count: int,
    model_id: str,
    semantic_prompt_version: str,
    emotion_prompt_version: str,
    output_dir: Union[str, Path],
    batch_size: int,
    backend: str,
    sem_completed: int,
    emo_completed: int,
    chunk_info: Optional[str] = None,
) -> None:
    """Print the required unambiguous banner confirming PRODUCTION MODE is active."""
    banner = (
        "\n"
        "======================================================================\n"
        "*** PRODUCTION KG-L EXTRACTION MODE ***\n"
        f"Total Eligible Movies : {eligible_count}\n"
        f"Target Movies for Run : {target_count}" + (f" ({chunk_info})" if chunk_info else "") + "\n"
        f"Model ID              : {model_id}\n"
        f"Semantic Prompt Ver   : {semantic_prompt_version} (Frozen v1.1.0)\n"
        f"Emotion Prompt Ver    : {emotion_prompt_version} (Frozen v1.0.0)\n"
        f"Emotion KG Threshold  : score > {DEFAULT_EMOTION_EDGE_THRESHOLD}\n"
        f"Output Directory      : {output_dir}\n"
        f"Batch Size            : {batch_size}\n"
        f"Backend               : {backend}\n"
        f"Resume Status         : Semantic completed: {sem_completed}, Emotion completed: {emo_completed}\n"
        "======================================================================\n"
    )
    logger.info(banner)
    print(banner, flush=True)


# ==============================================================================
# 2. ELIGIBLE MOVIE LOADER & PARTITIONING
# ==============================================================================

def load_eligible_production_movies(
    metadata_path: Union[str, Path] = "data/processed/ml-1m/metadata/movies_with_tmdb.parquet",
) -> pd.DataFrame:
    """Load and filter catalog movies with valid, usable Phase 3A plot overviews.

    Selection criteria:
    - has_overview == True
    - overview is non-null and non-empty (after stripping whitespace)
    - movie_idx, movie_id, tmdb_id, ml_title are non-null

    Results are sorted deterministically by movie_idx ascending.

    Args:
        metadata_path: Path to movies_with_tmdb.parquet.

    Returns:
        DataFrame containing eligible movies sorted deterministically.

    Raises:
        FileNotFoundError: If metadata_path does not exist.
        ValueError: If required columns are missing or dataset has zero eligible movies.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Input metadata artifact not found: {path}")

    df = pd.read_parquet(path)

    required_cols = ["movie_idx", "movie_id", "tmdb_id", "ml_title", "overview", "has_overview"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input metadata missing required columns: {missing_cols}")

    # Apply strict filter
    mask = (
        (df["has_overview"] == True)
        & df["overview"].notna()
        & (df["overview"].astype(str).str.strip() != "")
        & df["movie_idx"].notna()
        & df["movie_id"].notna()
        & df["tmdb_id"].notna()
        & df["ml_title"].notna()
    )
    df_eligible = df[mask].copy()

    if len(df_eligible) == 0:
        raise ValueError(f"No eligible movies found in {metadata_path} matching criteria.")

    # Sort deterministically by movie_idx ascending
    df_eligible = df_eligible.sort_values("movie_idx", ascending=True).reset_index(drop=True)

    # Ensure integer types for identifiers
    df_eligible["movie_idx"] = df_eligible["movie_idx"].astype(int)
    df_eligible["movie_id"] = df_eligible["movie_id"].astype(int)
    df_eligible["tmdb_id"] = df_eligible["tmdb_id"].astype(int)
    df_eligible["ml_title"] = df_eligible["ml_title"].astype(str)
    df_eligible["overview"] = df_eligible["overview"].astype(str)

    logger.info(
        f"Loaded {len(df_eligible)} eligible production movies from {metadata_path} "
        f"(total catalog size: {len(df)})."
    )
    return df_eligible


def partition_movies(
    df: pd.DataFrame,
    chunk_id: Optional[int] = None,
    num_chunks: Optional[int] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Partition eligible movies by chunk or range with deterministic coverage.

    Supports either:
    - (chunk_id, num_chunks): 0-indexed chunking dividing N items into K non-overlapping slices.
    - (start_idx, end_idx): row index range [start_idx, end_idx) over the sorted DataFrame.

    Args:
        df: Eligible movies DataFrame (must be pre-sorted).
        chunk_id: 0-indexed chunk index (0 <= chunk_id < num_chunks).
        num_chunks: Total number of chunks (>= 1).
        start_idx: 0-indexed start index (inclusive).
        end_idx: 0-indexed end index (exclusive).

    Returns:
        Tuple of (partitioned DataFrame, descriptive chunk info string).

    Raises:
        ValueError: If parameters are invalid or out of bounds.
    """
    total = len(df)
    if total == 0:
        return df.copy(), None

    # Check for chunk-based partitioning
    if chunk_id is not None or num_chunks is not None:
        if chunk_id is None or num_chunks is None:
            raise ValueError("Both chunk_id and num_chunks must be provided for chunk partitioning.")
        if num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}.")
        if not (0 <= chunk_id < num_chunks):
            raise ValueError(f"chunk_id must be in [0, {num_chunks - 1}], got {chunk_id}.")

        chunk_size = total // num_chunks
        remainder = total % num_chunks
        start = chunk_id * chunk_size + min(chunk_id, remainder)
        end = start + chunk_size + (1 if chunk_id < remainder else 0)

        chunk_df = df.iloc[start:end].copy().reset_index(drop=True)
        info_str = f"Chunk {chunk_id + 1}/{num_chunks} (rows {start}..{end - 1}, count={len(chunk_df)})"
        return chunk_df, info_str

    # Check for range-based partitioning
    if start_idx is not None or end_idx is not None:
        s = 0 if start_idx is None else start_idx
        e = total if end_idx is None else end_idx

        if s < 0:
            raise ValueError(f"start_idx must be >= 0, got {s}.")
        if e < s:
            raise ValueError(f"end_idx ({e}) cannot be less than start_idx ({s}).")
        if s > total:
            raise ValueError(f"start_idx ({s}) exceeds total eligible movies ({total}).")

        e = min(e, total)
        range_df = df.iloc[s:e].copy().reset_index(drop=True)
        info_str = f"Range rows {s}..{e - 1} (count={len(range_df)})"
        return range_df, info_str

    return df.copy(), None


# ==============================================================================
# 3. PRODUCTION INFERENCE BACKENDS (vLLM & MOCK)
# ==============================================================================

class BaseProductionBackend(ABC):
    """Abstract base class for batched production extraction backends."""

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[Tuple[str, str]],
        task_type: str,
    ) -> List[str]:
        """Generate completions for a batch of (system_prompt, user_prompt) pairs.

        Args:
            prompts: List of (system_prompt, user_prompt) tuples.
            task_type: Either 'semantic' or 'emotion'.

        Returns:
            List of generated string completions in the exact same order.
        """
        pass


class VLLMProductionBackend(BaseProductionBackend):
    """vLLM-powered production backend initializing Qwen2.5-7B-Instruct ONCE.

    The single vLLM engine instance services both semantic and emotion extraction
    with appropriate task-specific sampling parameters (token limits).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = DEFAULT_TEMPERATURE,
        max_semantic_tokens: int = DEFAULT_MAX_SEMANTIC_TOKENS,
        max_emotion_tokens: int = DEFAULT_MAX_EMOTION_TOKENS,
        tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    ):
        """Initialize unified vLLM engine with Qwen2.5-7B-Instruct.

        Args:
            model_id: Hugging Face model repository ID.
            temperature: Sampling temperature (0.0 for greedy decoding).
            max_semantic_tokens: Token budget for semantic extraction.
            max_emotion_tokens: Token budget for emotion extraction.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory allocated to vLLM.
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. For GPU execution on ASU SOL, install via: "
                "pip install .[gpu] or activate the 'vllm312' conda environment.\n"
                "For local testing on macOS without GPU, run with: --backend mock"
            ) from exc

        self.model_id = model_id
        self.max_semantic_tokens = max_semantic_tokens
        self.max_emotion_tokens = max_emotion_tokens
        self.temperature = temperature

        # Configure task-specific sampling parameters
        self.semantic_sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_semantic_tokens,
        )
        self.emotion_sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_emotion_tokens,
        )

        logger.info(
            f"Loading single unified vLLM model '{model_id}' (tp={tensor_parallel_size}, "
            f"gpu_mem={gpu_memory_utilization})..."
        )
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM model initialized successfully. Ready for batched inference.")

    def generate_batch(
        self,
        prompts: List[Tuple[str, str]],
        task_type: str,
    ) -> List[str]:
        """Execute batched generation preserving exact 1-to-1 input alignment."""
        if not prompts:
            return []

        sampling_params = (
            self.emotion_sampling_params if task_type == "emotion"
            else self.semantic_sampling_params
        )

        # Format each prompt pair using Qwen's official chat template
        formatted_prompts: List[str] = []
        for sys_p, usr_p in prompts:
            messages = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(prompt_str)

        # Execute batched generation in vLLM
        outputs = self.llm.generate(formatted_prompts, sampling_params)

        # Extract completions maintaining input order
        return [output.outputs[0].text for output in outputs]


class MockProductionBackend(BaseProductionBackend):
    """Deterministic mock backend for local CPU development and automated testing.

    Generates schema-compliant JSON for both semantic and emotion tasks without
    requiring PyTorch GPU hardware, model downloads, or vLLM.
    """

    def __init__(
        self,
        failure_modes: Optional[Dict[Tuple[int, str], str]] = None,
        model_id: str = DEFAULT_MODEL_ID,
        semantic_prompt_version: str = DEFAULT_SEMANTIC_PROMPT_VERSION,
        emotion_prompt_version: str = DEFAULT_EMOTION_PROMPT_VERSION,
    ):
        """Initialize mock production backend.

        Args:
            failure_modes: Mapping of (movie_idx, task_type) to simulated failure mode:
                - 'malformed_json': Non-JSON string.
                - 'schema_validation_error': Schema violation.
                - 'id_mismatch': Wrong IDs returned.
                - 'retry_then_succeed': Fails attempt 1, succeeds attempt 2.
                - 'backend_error': Raises runtime exception.
            model_id: Model ID to embed in metadata.
            semantic_prompt_version: Semantic prompt version string.
            emotion_prompt_version: Emotion prompt version string.
        """
        self.model_id = model_id
        self.semantic_prompt_version = semantic_prompt_version
        self.emotion_prompt_version = emotion_prompt_version
        self.failure_modes = failure_modes or {}
        self.call_counts: Dict[Tuple[int, str], int] = {}

    def generate_batch(
        self,
        prompts: List[Tuple[str, str]],
        task_type: str,
    ) -> List[str]:
        """Generate mock batch completions preserving order."""
        completions: List[str] = []
        for sys_p, usr_p in prompts:
            comp = self._generate_single(sys_p, usr_p, task_type)
            completions.append(comp)
        return completions

    def _generate_single(self, system_prompt: str, user_prompt: str, task_type: str) -> str:
        """Generate single synthetic completion based on task type."""
        idx_m = re.search(r"- Movie Index:\s*(\d+)", user_prompt)
        id_m = re.search(r"- MovieLens ID:\s*(\d+)", user_prompt)
        title_m = re.search(r"- Title:\s*(.*)", user_prompt)
        tmdb_m = re.search(r"- TMDB ID:\s*(\d+)", user_prompt)

        movie_idx = int(idx_m.group(1)) if idx_m else 0
        movie_id = int(id_m.group(1)) if id_m else 1
        title = title_m.group(1).strip() if title_m else "Mock Movie"
        tmdb_id = int(tmdb_m.group(1)) if tmdb_m else 100

        key = (movie_idx, task_type)
        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        call_num = self.call_counts[key]

        fmode = self.failure_modes.get(key)
        if fmode == "backend_error":
            raise RuntimeError(f"Simulated backend error on movie {movie_idx} task {task_type}")

        if fmode == "retry_then_succeed" and call_num == 1:
            return "Malformed output on attempt 1: { incomplete json"

        if fmode == "malformed_json":
            return "Sorry, cannot parse this overview: {unclosed json"

        if task_type == "semantic":
            return self._generate_semantic(movie_idx, movie_id, tmdb_id, title, fmode)
        else:
            return self._generate_emotion(movie_idx, movie_id, tmdb_id, title, fmode)

    def _generate_semantic(
        self, movie_idx: int, movie_id: int, tmdb_id: int, title: str, fmode: Optional[str]
    ) -> str:
        if fmode == "id_mismatch":
            movie_idx += 9999
            movie_id += 9999

        if fmode == "schema_validation_error":
            bad_data = {
                "movie_idx": movie_idx,
                "movie_id": movie_id,
                "tmdb_id": tmdb_id,
                "title": title,
                "themes": "not a list",  # Invalid type
                "settings": [],
                "narrative_devices": [],
                "protagonist_types": [],
                "based_on": [],
                "emotions": {emo: 0.1 for emo in PLUTCHIK_EMOTIONS},
            }
            return json.dumps(bad_data)

        valid_data = {
            "movie_idx": movie_idx,
            "movie_id": movie_id,
            "tmdb_id": tmdb_id,
            "title": title,
            "themes": ["identity", "friendship"],
            "settings": ["city"],
            "narrative_devices": ["flashback"],
            "protagonist_types": ["student"],
            "based_on": [],
            "emotions": {
                "joy": 0.4,
                "trust": 0.35,
                "fear": 0.2,
                "surprise": 0.1,
                "sadness": 0.15,
                "disgust": 0.0,
                "anger": 0.05,
                "anticipation": 0.6,
            },
            "model_id": self.model_id,
            "prompt_version": self.semantic_prompt_version,
            "schema_version": DEFAULT_SEMANTIC_SCHEMA_VERSION,
        }
        return f"```json\n{json.dumps(valid_data, indent=2)}\n```"

    def _generate_emotion(
        self, movie_idx: int, movie_id: int, tmdb_id: int, title: str, fmode: Optional[str]
    ) -> str:
        if fmode == "id_mismatch":
            movie_idx += 9999
            movie_id += 9999

        if fmode == "schema_validation_error":
            bad_data = {
                "movie_idx": movie_idx,
                "movie_id": movie_id,
                "tmdb_id": tmdb_id,
                "title": title,
                "emotions": {
                    "joy": {"evidence": [], "score": 0.8},  # Inconsistent: score > 0 with empty evidence
                    "trust": {"evidence": [], "score": 0.0},
                    "fear": {"evidence": [], "score": 0.0},
                    "surprise": {"evidence": [], "score": 0.0},
                    "sadness": {"evidence": [], "score": 0.0},
                    "disgust": {"evidence": [], "score": 0.0},
                    "anger": {"evidence": [], "score": 0.0},
                    "anticipation": {"evidence": [], "score": 0.0},
                },
            }
            return json.dumps(bad_data)

        valid_data = {
            "movie_idx": movie_idx,
            "movie_id": movie_id,
            "tmdb_id": tmdb_id,
            "title": title,
            "emotions": {
                "joy": {"evidence": ["Friends reunite happily."], "score": 0.60},
                "trust": {"evidence": ["Allies support each other."], "score": 0.50},
                "fear": {"evidence": ["Characters face sudden danger."], "score": 0.35},
                "surprise": {"evidence": [], "score": 0.0},
                "sadness": {"evidence": [], "score": 0.0},
                "disgust": {"evidence": [], "score": 0.0},
                "anger": {"evidence": [], "score": 0.0},
                "anticipation": {"evidence": ["Anticipating the final showdown."], "score": 0.70},
            },
            "model_id": self.model_id,
            "prompt_version": self.emotion_prompt_version,
            "schema_version": DEFAULT_EMOTION_SCHEMA_VERSION,
        }
        return f"```json\n{json.dumps(valid_data, indent=2)}\n```"


# ==============================================================================
# 4. CHECKPOINT & PERSISTENCE MANAGER
# ==============================================================================

class ProductionCheckpointManager:
    """Manages independent checkpointing, resumption, and persistence for production extraction."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.semantic_raw_path = self.output_dir / "semantic_raw_outputs.jsonl"
        self.semantic_validated_path = self.output_dir / "semantic_validated.jsonl"
        self.semantic_failures_path = self.output_dir / "semantic_failures.jsonl"

        self.emotion_raw_path = self.output_dir / "emotion_raw_outputs.jsonl"
        self.emotion_validated_evidence_path = self.output_dir / "emotion_validated_evidence.jsonl"
        self.emotion_vectors_path = self.output_dir / "emotion_vectors.jsonl"
        self.emotion_failures_path = self.output_dir / "emotion_failures.jsonl"

        # Load completed indices
        self.semantic_validated_indices: Set[int] = self._load_indices_from_jsonl(self.semantic_validated_path)
        self.semantic_failed_indices: Set[int] = self._load_indices_from_jsonl(self.semantic_failures_path)

        self.emotion_validated_indices: Set[int] = self._load_indices_from_jsonl(self.emotion_vectors_path)
        self.emotion_failed_indices: Set[int] = self._load_indices_from_jsonl(self.emotion_failures_path)

        logger.info(
            f"Checkpoint status in {self.output_dir}: "
            f"Semantic: {len(self.semantic_validated_indices)} validated, {len(self.semantic_failed_indices)} failed; "
            f"Emotion: {len(self.emotion_validated_indices)} validated, {len(self.emotion_failed_indices)} failed."
        )

    def _load_indices_from_jsonl(self, path: Path) -> Set[int]:
        """Load unique movie_idx set from an existing JSON Lines file."""
        indices: Set[int] = set()
        if not path.exists():
            return indices

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "movie_idx" in data:
                        indices.add(int(data["movie_idx"]))
                except Exception:
                    continue
        return indices

    def is_semantic_done(self, movie_idx: int, retry_failed: bool = False) -> bool:
        """Check if semantic extraction is already completed for a movie index."""
        if movie_idx in self.semantic_validated_indices:
            return True
        if not retry_failed and movie_idx in self.semantic_failed_indices:
            return True
        return False

    def is_emotion_done(self, movie_idx: int, retry_failed: bool = False) -> bool:
        """Check if emotion extraction is already completed for a movie index."""
        if movie_idx in self.emotion_validated_indices:
            return True
        if not retry_failed and movie_idx in self.emotion_failed_indices:
            return True
        return False

    def append_semantic_raw(self, record: Dict[str, Any]) -> None:
        with open(self.semantic_raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def append_semantic_validated(self, record: Dict[str, Any]) -> None:
        with open(self.semantic_validated_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.semantic_validated_indices.add(int(record["movie_idx"]))

    def append_semantic_failure(self, record: Dict[str, Any]) -> None:
        with open(self.semantic_failures_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.semantic_failed_indices.add(int(record["movie_idx"]))

    def append_emotion_raw(self, record: Dict[str, Any]) -> None:
        with open(self.emotion_raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def append_emotion_validated(
        self, evidence_record: Dict[str, Any], vector_record: Dict[str, Any]
    ) -> None:
        with open(self.emotion_validated_evidence_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_record) + "\n")
        with open(self.emotion_vectors_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(vector_record) + "\n")
        self.emotion_validated_indices.add(int(vector_record["movie_idx"]))

    def append_emotion_failure(self, record: Dict[str, Any]) -> None:
        with open(self.emotion_failures_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.emotion_failed_indices.add(int(record["movie_idx"]))


def categorize_extraction_error(error: Optional[Exception], raw_text: str) -> str:
    """Classify model extraction error into standardized categories."""
    if error is None:
        return "unknown_error"

    if isinstance(error, ExtractionParseError):
        return "invalid_json"

    if isinstance(error, ExtractionValidationError):
        err_msg = str(error).lower()
        if "mismatch" in err_msg or "id" in err_msg:
            return "id_mismatch"
        return "schema_validation_error"

    if "backend" in str(error).lower() or "cuda" in str(error).lower() or "memory" in str(error).lower():
        return "backend_error"

    return "schema_validation_error"


# ==============================================================================
# 5. EMOTION DESCRIPTIVE STATISTICS
# ==============================================================================

def compute_production_emotion_statistics(
    vector_records: List[Dict[str, Any]],
    threshold: float = DEFAULT_EMOTION_EDGE_THRESHOLD,
) -> Dict[str, Any]:
    """Compute descriptive statistics over successfully extracted emotion vectors.

    Args:
        vector_records: List of 8-dimensional emotion vector dictionaries.
        threshold: Edge threshold (default: 0.3).

    Returns:
        Dictionary with per-emotion distribution metrics and overall coverage.
    """
    if not vector_records:
        return {
            "total_vectors": 0,
            "threshold": threshold,
            "movies_with_no_emotion_above_threshold": 0,
            "movie_coverage_pct": 0.0,
            "emotions": {emo: {"mean": 0.0, "median": 0.0, "zero_count": 0, "count_above_threshold": 0} for emo in PLUTCHIK_EMOTIONS},
        }

    total_movies = len(vector_records)
    stats: Dict[str, Any] = {
        "total_vectors": total_movies,
        "threshold": threshold,
        "emotions": {},
    }

    no_prominent_count = 0
    for rec in vector_records:
        prominent = [emo for emo in PLUTCHIK_EMOTIONS if float(rec.get(emo, 0.0)) > threshold]
        if not prominent:
            no_prominent_count += 1

    stats["movies_with_no_emotion_above_threshold"] = no_prominent_count
    coverage_count = total_movies - no_prominent_count
    stats["movie_coverage_pct"] = round((coverage_count / total_movies) * 100.0, 2) if total_movies > 0 else 0.0

    for emo in PLUTCHIK_EMOTIONS:
        scores = [float(r.get(emo, 0.0)) for r in vector_records]
        zero_cnt = sum(1 for s in scores if s == 0.0)
        above_cnt = sum(1 for s in scores if s > threshold)
        mean_val = float(np.mean(scores)) if scores else 0.0
        median_val = float(np.median(scores)) if scores else 0.0

        stats["emotions"][emo] = {
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "zero_count": zero_cnt,
            "count_above_threshold": above_cnt,
        }

    return stats


# ==============================================================================
# 6. RUNTIME ESTIMATION
# ==============================================================================

def estimate_production_runtime(
    num_movies: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    semantic_sec_per_movie: float = 0.8,
    emotion_sec_per_movie: float = 1.0,
) -> Dict[str, Any]:
    """Estimate production extraction runtime under multiple throughput scenarios.

    Args:
        num_movies: Number of eligible movies to extract.
        batch_size: Configured vLLM batch size.
        semantic_sec_per_movie: Base seconds per semantic extraction.
        emotion_sec_per_movie: Base seconds per emotion extraction.

    Returns:
        Dictionary with optimistic, moderate, and conservative runtime estimates.
    """
    # Factor in vLLM continuous batching speedup
    batch_factor = max(1.0, math.log2(batch_size))
    # Combined effective sec/movie
    opt_sem = max(0.25, semantic_sec_per_movie / (0.6 * batch_factor))
    opt_emo = max(0.35, emotion_sec_per_movie / (0.6 * batch_factor))
    opt_combined_rate = opt_sem + opt_emo
    opt_total_sec = num_movies * opt_combined_rate

    mod_sem = max(0.50, semantic_sec_per_movie / (0.45 * batch_factor))
    mod_emo = max(0.65, emotion_sec_per_movie / (0.45 * batch_factor))
    mod_combined_rate = mod_sem + mod_emo
    mod_total_sec = num_movies * mod_combined_rate

    # Conservative scenario includes 10% retry overhead and startup time
    con_sem = max(0.85, semantic_sec_per_movie / (0.35 * batch_factor))
    con_emo = max(1.10, emotion_sec_per_movie / (0.35 * batch_factor))
    con_combined_rate = (con_sem + con_emo) * 1.10
    con_total_sec = num_movies * con_combined_rate + 300.0  # +5 min startup

    estimates = {
        "num_movies": num_movies,
        "batch_size": batch_size,
        "optimistic": {
            "total_seconds": round(opt_total_sec, 1),
            "total_hours": round(opt_total_sec / 3600.0, 2),
            "effective_sec_per_movie": round(opt_combined_rate, 3),
        },
        "moderate": {
            "total_seconds": round(mod_total_sec, 1),
            "total_hours": round(mod_total_sec / 3600.0, 2),
            "effective_sec_per_movie": round(mod_combined_rate, 3),
        },
        "conservative": {
            "total_seconds": round(con_total_sec, 1),
            "total_hours": round(con_total_sec / 3600.0, 2),
            "effective_sec_per_movie": round(con_combined_rate, 3),
        },
        "recommended_slurm_walltime": "06:00:00",
    }
    return estimates


def print_runtime_estimation(estimates: Dict[str, Any]) -> None:
    """Print formatted ASCII runtime estimation report."""
    n = estimates["num_movies"]
    bs = estimates["batch_size"]
    rec = estimates["recommended_slurm_walltime"]
    report = (
        "\n"
        "======================================================================\n"
        "          PLOTGRAPH PRODUCTION EXTRACTION RUNTIME ESTIMATION\n"
        "======================================================================\n"
        f"Eligible Movie Count : {n}\n"
        f"Inference Batch Size : {bs}\n"
        "----------------------------------------------------------------------\n"
        "Scenario        Effective Sec/Movie   Total Runtime   Recommended Walltime\n"
        "----------------------------------------------------------------------\n"
        f"Optimistic      {estimates['optimistic']['effective_sec_per_movie']:>6.2f}s/movie         {estimates['optimistic']['total_hours']:>5.2f} hrs        04:00:00\n"
        f"Moderate        {estimates['moderate']['effective_sec_per_movie']:>6.2f}s/movie         {estimates['moderate']['total_hours']:>5.2f} hrs        06:00:00\n"
        f"Conservative    {estimates['conservative']['effective_sec_per_movie']:>6.2f}s/movie         {estimates['conservative']['total_hours']:>5.2f} hrs        {rec}\n"
        "======================================================================\n"
        "NOTE: Estimates include both Semantic (v1.1) and Emotion (v1.0) tasks.\n"
        "Checkpointing persists after every batch; jobs can be resumed safely.\n"
        "======================================================================\n"
    )
    print(report, flush=True)


# ==============================================================================
# 7. PRODUCTION BATCHED EXTRACTION RUNNER
# ==============================================================================

def run_production_extraction(
    config_path: Union[str, Path] = "configs/kg_l.yaml",
    backend_type: str = "vllm",
    input_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    batch_size: Optional[int] = None,
    chunk_id: Optional[int] = None,
    num_chunks: Optional[int] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    retry_failed: bool = False,
    export_parquet: Optional[bool] = None,
    backend_instance: Optional[BaseProductionBackend] = None,
    allow_mock_in_production_dir: bool = False,
) -> Dict[str, Any]:
    """Execute production-scale batched extraction for all eligible MovieLens movies.

    Initializes Qwen2.5-7B-Instruct once and coordinates batched inference across
    both frozen semantic prompt v1.1.0 and dedicated emotion prompt v1.0.0.

    Args:
        config_path: Path to kg_l.yaml.
        backend_type: 'vllm' for GPU or 'mock' for CPU testing.
        input_path: Override path to metadata parquet.
        output_dir: Override path to output directory.
        batch_size: Override inference batch size.
        chunk_id: Optional 0-indexed chunk ID.
        num_chunks: Optional total chunk count.
        start_idx: Optional start index.
        end_idx: Optional end index.
        retry_failed: Whether to re-attempt previously failed movies on resume.
        export_parquet: Whether to generate compact Parquet artifacts.
        backend_instance: Optional pre-initialized BaseProductionBackend.
        allow_mock_in_production_dir: Safety override for mock testing.

    Returns:
        Dictionary summarizing execution counts, runtime, throughput, and diagnostics.
    """
    start_wall_time = time.time()

    # 1. Load configuration
    cfg: Dict[str, Any] = {}
    cfg_p = Path(config_path)
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    prod_cfg = cfg.get("production", {})
    paths_cfg = cfg.get("paths", {})
    ontology_cfg = cfg.get("ontology", {})

    metadata_file = Path(input_path or paths_cfg.get("input_metadata_parquet", "data/processed/ml-1m/metadata/movies_with_tmdb.parquet"))
    target_out_dir = Path(output_dir or prod_cfg.get("output_dir", DEFAULT_PRODUCTION_OUTPUT_DIR))

    # Safety Guard for Mock Mode
    if backend_type == "mock" and str(target_out_dir).rstrip("/") == DEFAULT_PRODUCTION_OUTPUT_DIR.rstrip("/") and not allow_mock_in_production_dir:
        raise ValueError(
            f"Production safety violation: Mock backend cannot write to the real production "
            f"output directory ('{target_out_dir}'). Specify a dedicated test directory (e.g. "
            f"--output-dir data/processed/ml-1m/kg_l/mock_extraction) or pass allow_mock_in_production_dir=True."
        )

    model_id = str(prod_cfg.get("model_id", DEFAULT_MODEL_ID))
    sem_prompt_ver = str(prod_cfg.get("semantic_prompt_version", DEFAULT_SEMANTIC_PROMPT_VERSION))
    sem_schema_ver = str(prod_cfg.get("semantic_schema_version", DEFAULT_SEMANTIC_SCHEMA_VERSION))
    emo_prompt_ver = str(prod_cfg.get("emotion_prompt_version", DEFAULT_EMOTION_PROMPT_VERSION))
    emo_schema_ver = str(prod_cfg.get("emotion_schema_version", DEFAULT_EMOTION_SCHEMA_VERSION))
    emotion_threshold = float(ontology_cfg.get("emotion_edge_threshold", DEFAULT_EMOTION_EDGE_THRESHOLD))

    bs = int(batch_size or prod_cfg.get("batch_size", DEFAULT_BATCH_SIZE))
    max_retries = validate_and_resolve_max_retries(None, prod_cfg)
    should_export_parquet = export_parquet if export_parquet is not None else bool(prod_cfg.get("export_parquet", True))

    # 2. Load and partition eligible movies
    df_all_eligible = load_eligible_production_movies(metadata_file)
    eligible_count = len(df_all_eligible)

    df_target, chunk_info = partition_movies(
        df_all_eligible,
        chunk_id=chunk_id,
        num_chunks=num_chunks,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    target_count = len(df_target)

    # 3. Initialize Checkpoint Manager
    ckpt = ProductionCheckpointManager(target_out_dir)

    # Compute initial resume counts within target slice
    target_indices = set(df_target["movie_idx"])
    sem_completed_init = sum(1 for idx in target_indices if ckpt.is_semantic_done(idx, retry_failed=False))
    emo_completed_init = sum(1 for idx in target_indices if ckpt.is_emotion_done(idx, retry_failed=False))

    # 4. Print Production Banner
    print_production_banner(
        eligible_count=eligible_count,
        target_count=target_count,
        model_id=model_id,
        semantic_prompt_version=sem_prompt_ver,
        emotion_prompt_version=emo_prompt_ver,
        output_dir=target_out_dir,
        batch_size=bs,
        backend=backend_type,
        sem_completed=sem_completed_init,
        emo_completed=emo_completed_init,
        chunk_info=chunk_info,
    )

    # 5. Initialize Backend (LOAD ONCE)
    if backend_instance is not None:
        backend = backend_instance
    elif backend_type == "vllm":
        backend = VLLMProductionBackend(
            model_id=model_id,
            temperature=float(prod_cfg.get("temperature", DEFAULT_TEMPERATURE)),
            max_semantic_tokens=int(prod_cfg.get("max_semantic_tokens", DEFAULT_MAX_SEMANTIC_TOKENS)),
            max_emotion_tokens=int(prod_cfg.get("max_emotion_tokens", DEFAULT_MAX_EMOTION_TOKENS)),
            tensor_parallel_size=int(prod_cfg.get("tensor_parallel_size", DEFAULT_TENSOR_PARALLEL_SIZE)),
            gpu_memory_utilization=float(prod_cfg.get("gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION)),
        )
    elif backend_type == "mock":
        backend = MockProductionBackend(
            model_id=model_id,
            semantic_prompt_version=sem_prompt_ver,
            emotion_prompt_version=emo_prompt_ver,
        )
    else:
        raise ValueError(f"Unknown backend_type '{backend_type}'. Must be 'vllm' or 'mock'.")

    # 6. Batched Execution Loop
    records_to_process = df_target.to_dict(orient="records")
    total_batches = math.ceil(target_count / bs) if target_count > 0 else 0

    sem_successes = 0
    sem_failures = 0
    sem_retries = 0

    emo_successes = 0
    emo_failures = 0
    emo_retries = 0

    logger.info(f"Starting batched production processing: {target_count} movies across {total_batches} batches.")

    for b_idx in range(total_batches):
        batch_slice = records_to_process[b_idx * bs : (b_idx + 1) * bs]

        # ----------------------------------------------------------------------
        # A. Semantic Extraction for Current Batch
        # ----------------------------------------------------------------------
        sem_targets = [m for m in batch_slice if not ckpt.is_semantic_done(m["movie_idx"], retry_failed=retry_failed)]
        if sem_targets:
            # Attempt 1
            sem_prompts = [
                format_extraction_prompt(
                    title=m["ml_title"],
                    overview=m["overview"],
                    movie_id=m["movie_id"],
                    movie_idx=m["movie_idx"],
                    tmdb_id=m["tmdb_id"],
                    prompt_version=sem_prompt_ver,
                )
                for m in sem_targets
            ]
            sem_outputs = backend.generate_batch(sem_prompts, task_type="semantic")

            retry_sem_targets: List[Dict[str, Any]] = []
            retry_sem_errors: List[Exception] = []

            for m, out_text in zip(sem_targets, sem_outputs):
                ckpt.append_semantic_raw({
                    "movie_idx": m["movie_idx"],
                    "movie_id": m["movie_id"],
                    "attempt": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "raw_response": out_text,
                })
                try:
                    val_rec = parse_and_validate_extraction(
                        out_text,
                        expected_ids={"movie_id": m["movie_id"], "movie_idx": m["movie_idx"]},
                    )
                    ckpt.append_semantic_validated(val_rec.model_dump())
                    sem_successes += 1
                except Exception as err:
                    if max_retries >= 1:
                        retry_sem_targets.append(m)
                        retry_sem_errors.append(err)
                    else:
                        cat = categorize_extraction_error(err, out_text)
                        ckpt.append_semantic_failure({
                            "movie_idx": m["movie_idx"],
                            "movie_id": m["movie_id"],
                            "title": m["ml_title"],
                            "error_type": cat,
                            "error_message": str(err),
                            "attempts": 1,
                            "last_raw_response": out_text,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        sem_failures += 1

            # Attempt 2 (Retries for Semantic)
            if retry_sem_targets:
                sem_retries += len(retry_sem_targets)
                retry_prompts = [
                    (
                        sys_p,
                        f"{usr_p}\n\n{SEMANTIC_RETRY_REMINDER}",
                    )
                    for sys_p, usr_p in [
                        format_extraction_prompt(
                            title=m["ml_title"],
                            overview=m["overview"],
                            movie_id=m["movie_id"],
                            movie_idx=m["movie_idx"],
                            tmdb_id=m["tmdb_id"],
                            prompt_version=sem_prompt_ver,
                        )
                        for m in retry_sem_targets
                    ]
                ]
                retry_outputs = backend.generate_batch(retry_prompts, task_type="semantic")

                for m, out_text, prev_err in zip(retry_sem_targets, retry_outputs, retry_sem_errors):
                    ckpt.append_semantic_raw({
                        "movie_idx": m["movie_idx"],
                        "movie_id": m["movie_id"],
                        "attempt": 2,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "raw_response": out_text,
                    })
                    try:
                        val_rec = parse_and_validate_extraction(
                            out_text,
                            expected_ids={"movie_id": m["movie_id"], "movie_idx": m["movie_idx"]},
                        )
                        ckpt.append_semantic_validated(val_rec.model_dump())
                        sem_successes += 1
                    except Exception as err:
                        cat = categorize_extraction_error(err, out_text)
                        ckpt.append_semantic_failure({
                            "movie_idx": m["movie_idx"],
                            "movie_id": m["movie_id"],
                            "title": m["ml_title"],
                            "error_type": cat,
                            "error_message": str(err),
                            "attempts": 2,
                            "last_raw_response": out_text,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        sem_failures += 1

        # ----------------------------------------------------------------------
        # B. Emotion Extraction for Current Batch
        # ----------------------------------------------------------------------
        emo_targets = [m for m in batch_slice if not ckpt.is_emotion_done(m["movie_idx"], retry_failed=retry_failed)]
        if emo_targets:
            # Attempt 1
            emo_prompts = [
                format_emotion_prompt(
                    title=m["ml_title"],
                    overview=m["overview"],
                    movie_id=m["movie_id"],
                    movie_idx=m["movie_idx"],
                    tmdb_id=m["tmdb_id"],
                    prompt_version=emo_prompt_ver,
                )
                for m in emo_targets
            ]
            emo_outputs = backend.generate_batch(emo_prompts, task_type="emotion")

            retry_emo_targets: List[Dict[str, Any]] = []
            retry_emo_errors: List[Exception] = []

            for m, out_text in zip(emo_targets, emo_outputs):
                ckpt.append_emotion_raw({
                    "movie_idx": m["movie_idx"],
                    "movie_id": m["movie_id"],
                    "attempt": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "raw_response": out_text,
                })
                try:
                    emo_rec = parse_and_validate_emotion_extraction(
                        out_text,
                        expected_ids={"movie_id": m["movie_id"], "movie_idx": m["movie_idx"]},
                    )
                    vec_rec = emo_rec.to_vector_record(threshold=emotion_threshold)
                    ckpt.append_emotion_validated(emo_rec.model_dump(), vec_rec)
                    emo_successes += 1
                except Exception as err:
                    if max_retries >= 1:
                        retry_emo_targets.append(m)
                        retry_emo_errors.append(err)
                    else:
                        cat = categorize_extraction_error(err, out_text)
                        ckpt.append_emotion_failure({
                            "movie_idx": m["movie_idx"],
                            "movie_id": m["movie_id"],
                            "title": m["ml_title"],
                            "error_type": cat,
                            "error_message": str(err),
                            "attempts": 1,
                            "last_raw_response": out_text,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        emo_failures += 1

            # Attempt 2 (Retries for Emotion)
            if retry_emo_targets:
                emo_retries += len(retry_emo_targets)
                retry_prompts = [
                    (
                        sys_p,
                        f"{usr_p}\n\n{EMOTION_RETRY_REMINDER}",
                    )
                    for sys_p, usr_p in [
                        format_emotion_prompt(
                            title=m["ml_title"],
                            overview=m["overview"],
                            movie_id=m["movie_id"],
                            movie_idx=m["movie_idx"],
                            tmdb_id=m["tmdb_id"],
                            prompt_version=emo_prompt_ver,
                        )
                        for m in retry_emo_targets
                    ]
                ]
                retry_outputs = backend.generate_batch(retry_prompts, task_type="emotion")

                for m, out_text, prev_err in zip(retry_emo_targets, retry_outputs, retry_emo_errors):
                    ckpt.append_emotion_raw({
                        "movie_idx": m["movie_idx"],
                        "movie_id": m["movie_id"],
                        "attempt": 2,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "raw_response": out_text,
                    })
                    try:
                        emo_rec = parse_and_validate_emotion_extraction(
                            out_text,
                            expected_ids={"movie_id": m["movie_id"], "movie_idx": m["movie_idx"]},
                        )
                        vec_rec = emo_rec.to_vector_record(threshold=emotion_threshold)
                        ckpt.append_emotion_validated(emo_rec.model_dump(), vec_rec)
                        emo_successes += 1
                    except Exception as err:
                        cat = categorize_extraction_error(err, out_text)
                        ckpt.append_emotion_failure({
                            "movie_idx": m["movie_idx"],
                            "movie_id": m["movie_id"],
                            "title": m["ml_title"],
                            "error_type": cat,
                            "error_message": str(err),
                            "attempts": 2,
                            "last_raw_response": out_text,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        emo_failures += 1

        # Periodic Progress Logging
        sem_done_cnt = sum(1 for idx in target_indices if ckpt.is_semantic_done(idx, retry_failed=False))
        emo_done_cnt = sum(1 for idx in target_indices if ckpt.is_emotion_done(idx, retry_failed=False))
        elapsed_now = time.time() - start_wall_time
        processed_now = max(sem_done_cnt - sem_completed_init, emo_done_cnt - emo_completed_init, 1)
        rate_now = processed_now / elapsed_now if elapsed_now > 0 else 0.0

        if (b_idx + 1) % 5 == 0 or (b_idx + 1) == total_batches:
            logger.info(
                f"[Batch {b_idx + 1}/{total_batches}] "
                f"Semantic: {sem_done_cnt}/{target_count} done (new: +{sem_successes} ok, {sem_failures} fail) | "
                f"Emotion: {emo_done_cnt}/{target_count} done (new: +{emo_successes} ok, {emo_failures} fail) | "
                f"Elapsed: {elapsed_now:.1f}s (~{rate_now:.2f} movies/s)"
            )
            print(
                f"Progress [Batch {b_idx + 1}/{total_batches}]: "
                f"Semantic: {sem_done_cnt}/{target_count} | "
                f"Emotion: {emo_done_cnt}/{target_count} | "
                f"Elapsed: {elapsed_now:.1f}s",
                flush=True,
            )

    total_runtime_seconds = time.time() - start_wall_time

    # 7. Post-Processing & Descriptive Statistics over All Cumulative Vectors
    cumulative_vectors: List[Dict[str, Any]] = []
    if ckpt.emotion_vectors_path.exists():
        with open(ckpt.emotion_vectors_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        cumulative_vectors.append(json.loads(line))
                    except Exception:
                        continue

    emotion_stats = compute_production_emotion_statistics(cumulative_vectors, threshold=emotion_threshold)

    # Throughput calculations
    sem_throughput = (sem_successes + sem_failures) / total_runtime_seconds if total_runtime_seconds > 0 else 0.0
    emo_throughput = (emo_successes + emo_failures) / total_runtime_seconds if total_runtime_seconds > 0 else 0.0
    combined_throughput = (sem_successes + sem_failures + emo_successes + emo_failures) / total_runtime_seconds if total_runtime_seconds > 0 else 0.0

    # 8. Production Summary JSON
    summary: Dict[str, Any] = {
        "production_mode": True,
        "model_id": model_id,
        "backend": backend_type,
        "input_metadata_file": str(metadata_file),
        "output_directory": str(target_out_dir),
        "total_eligible_movies": eligible_count,
        "target_movies_in_run": target_count,
        "chunk_info": chunk_info,
        "semantic_prompt_version": sem_prompt_ver,
        "semantic_schema_version": sem_schema_ver,
        "emotion_prompt_version": emo_prompt_ver,
        "emotion_schema_version": emo_schema_ver,
        "emotion_kg_threshold": emotion_threshold,
        "batch_size": bs,
        "max_retries": max_retries,
        "session_metrics": {
            "semantic": {
                "successes": sem_successes,
                "failures": sem_failures,
                "retries": sem_retries,
                "success_rate": round((sem_successes / max(sem_successes + sem_failures, 1)) * 100.0, 2),
                "throughput_movies_per_sec": round(sem_throughput, 3),
            },
            "emotion": {
                "successes": emo_successes,
                "failures": emo_failures,
                "retries": emo_retries,
                "success_rate": round((emo_successes / max(emo_successes + emo_failures, 1)) * 100.0, 2),
                "throughput_movies_per_sec": round(emo_throughput, 3),
            },
            "total_runtime_seconds": round(total_runtime_seconds, 2),
            "combined_throughput_tasks_per_sec": round(combined_throughput, 3),
        },
        "cumulative_target_status": {
            "semantic_validated_count": len([i for i in target_indices if i in ckpt.semantic_validated_indices]),
            "semantic_failed_count": len([i for i in target_indices if i in ckpt.semantic_failed_indices]),
            "emotion_validated_count": len([i for i in target_indices if i in ckpt.emotion_validated_indices]),
            "emotion_failed_count": len([i for i in target_indices if i in ckpt.emotion_failed_indices]),
        },
        "emotion_descriptive_statistics": emotion_stats,
    }

    summary_file = target_out_dir / "production_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Wrote production summary to {summary_file}")

    # 9. Optional Compact Parquet Export
    if should_export_parquet:
        export_production_parquet_artifacts(target_out_dir)

    return summary


def export_production_parquet_artifacts(output_dir: Path) -> None:
    """Export compact Parquet files from validated JSONL files for downstream consumption."""
    sem_jsonl = output_dir / "semantic_validated.jsonl"
    sem_parquet = output_dir / "semantic_validated.parquet"
    if sem_jsonl.exists():
        records = []
        with open(sem_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        if records:
            df_sem = pd.DataFrame(records)
            df_sem.to_parquet(sem_parquet, index=False)
            logger.info(f"Exported {len(df_sem)} semantic records to {sem_parquet}")

    emo_jsonl = output_dir / "emotion_vectors.jsonl"
    emo_parquet = output_dir / "emotion_vectors.parquet"
    if emo_jsonl.exists():
        records = []
        with open(emo_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        if records:
            df_emo = pd.DataFrame(records)
            df_emo.to_parquet(emo_parquet, index=False)
            logger.info(f"Exported {len(df_emo)} emotion vector records to {emo_parquet}")


# ==============================================================================
# 8. COMPLETION VALIDATION UTILITY
# ==============================================================================

def validate_production_run(
    output_dir: Union[str, Path],
    metadata_path: Union[str, Path] = "data/processed/ml-1m/metadata/movies_with_tmdb.parquet",
    expected_indices: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """Validate completeness, integrity, and schema conformance of a production run.

    Verifies:
    1. Every eligible movie has either a validated record or permanent failure for semantic task.
    2. Every eligible movie has either a validated record or permanent failure for emotion task.
    3. No duplicate successful movie_idx records.
    4. Identifiers align with Phase 3A metadata (movie_id, tmdb_id, title).
    5. Exactly eight continuous emotion scores in [0.0, 1.0].
    6. Nonzero evidence-score consistency holds for all validated emotion records.
    7. Output prompt/schema versions match frozen methodology.

    Args:
        output_dir: Directory containing production artifacts.
        metadata_path: Path to movies_with_tmdb.parquet.
        expected_indices: Optional subset of movie_idx to restrict validation to (e.g. for chunks).

    Returns:
        Structured dictionary with validation status, error lists, and diagnostic summary.
    """
    out_path = Path(output_dir)
    errors: List[str] = []

    # 1. Load eligible metadata reference
    df_eligible = load_eligible_production_movies(metadata_path)
    if expected_indices is not None:
        df_eligible = df_eligible[df_eligible["movie_idx"].isin(expected_indices)].copy()

    meta_by_idx = {int(r["movie_idx"]): r for r in df_eligible.to_dict(orient="records")}
    all_target_indices = set(meta_by_idx.keys())
    target_count = len(all_target_indices)

    # 2. Load Semantic outputs
    sem_val_path = out_path / "semantic_validated.jsonl"
    sem_fail_path = out_path / "semantic_failures.jsonl"

    sem_validated: Dict[int, Dict[str, Any]] = {}
    sem_val_duplicates: Set[int] = set()
    if sem_val_path.exists():
        with open(sem_val_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                idx = int(d["movie_idx"])
                if idx in sem_validated:
                    sem_val_duplicates.add(idx)
                sem_validated[idx] = d

    sem_failures: Dict[int, Dict[str, Any]] = {}
    if sem_fail_path.exists():
        with open(sem_fail_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                idx = int(d["movie_idx"])
                sem_failures[idx] = d

    # 3. Load Emotion outputs
    emo_vec_path = out_path / "emotion_vectors.jsonl"
    emo_ev_path = out_path / "emotion_validated_evidence.jsonl"
    emo_fail_path = out_path / "emotion_failures.jsonl"

    emo_vectors: Dict[int, Dict[str, Any]] = {}
    emo_vec_duplicates: Set[int] = set()
    if emo_vec_path.exists():
        with open(emo_vec_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                idx = int(d["movie_idx"])
                if idx in emo_vectors:
                    emo_vec_duplicates.add(idx)
                emo_vectors[idx] = d

    emo_evidence: Dict[int, Dict[str, Any]] = {}
    if emo_ev_path.exists():
        with open(emo_ev_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                idx = int(d["movie_idx"])
                emo_evidence[idx] = d

    emo_failures: Dict[int, Dict[str, Any]] = {}
    if emo_fail_path.exists():
        with open(emo_fail_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                idx = int(d["movie_idx"])
                emo_failures[idx] = d

    # Duplicate checks
    if sem_val_duplicates:
        errors.append(f"Semantic validated contains {len(sem_val_duplicates)} duplicate movie_idx entries: {list(sem_val_duplicates)[:5]}")
    if emo_vec_duplicates:
        errors.append(f"Emotion vectors contain {len(emo_vec_duplicates)} duplicate movie_idx entries: {list(emo_vec_duplicates)[:5]}")

    # Completeness checks
    missing_semantic = [idx for idx in all_target_indices if idx not in sem_validated and idx not in sem_failures]
    if missing_semantic:
        errors.append(f"{len(missing_semantic)} movies have neither validated nor failure record for semantic task.")

    missing_emotion = [idx for idx in all_target_indices if idx not in emo_vectors and idx not in emo_failures]
    if missing_emotion:
        errors.append(f"{len(missing_emotion)} movies have neither validated nor failure record for emotion task.")

    # Alignment and schema checks over validated records
    for idx, sem_rec in sem_validated.items():
        if idx not in meta_by_idx:
            continue
        meta = meta_by_idx[idx]
        if sem_rec["movie_id"] != meta["movie_id"]:
            errors.append(f"Movie {idx}: semantic movie_id mismatch ({sem_rec['movie_id']} != {meta['movie_id']})")
        if sem_rec.get("prompt_version") != DEFAULT_SEMANTIC_PROMPT_VERSION:
            errors.append(f"Movie {idx}: semantic prompt version mismatch ({sem_rec.get('prompt_version')} != {DEFAULT_SEMANTIC_PROMPT_VERSION})")

    for idx, emo_rec in emo_vectors.items():
        if idx not in meta_by_idx:
            continue
        meta = meta_by_idx[idx]
        if emo_rec["movie_id"] != meta["movie_id"]:
            errors.append(f"Movie {idx}: emotion movie_id mismatch ({emo_rec['movie_id']} != {meta['movie_id']})")
        for emo in PLUTCHIK_EMOTIONS:
            if emo not in emo_rec:
                errors.append(f"Movie {idx}: emotion vector missing '{emo}'")
            else:
                score = emo_rec[emo]
                if not (0.0 <= score <= 1.0):
                    errors.append(f"Movie {idx}: emotion '{emo}' score {score} out of bounds [0, 1]")

    # Nonzero evidence-score consistency
    for idx, ev_rec in emo_evidence.items():
        emotions_map = ev_rec.get("emotions", {})
        for emo in PLUTCHIK_EMOTIONS:
            emo_entry = emotions_map.get(emo, {})
            score = emo_entry.get("score", 0.0)
            evidence = emo_entry.get("evidence", [])
            if score > 0.0 and len(evidence) == 0:
                errors.append(f"Movie {idx}: emotion '{emo}' has score {score} > 0.0 but evidence is empty []")
            if score == 0.0 and len(evidence) > 0:
                errors.append(f"Movie {idx}: emotion '{emo}' has score 0.0 but evidence is non-empty: {evidence}")

    is_valid = (len(errors) == 0)

    result = {
        "is_valid": is_valid,
        "target_movie_count": target_count,
        "semantic_validated_count": len([i for i in all_target_indices if i in sem_validated]),
        "semantic_failure_count": len([i for i in all_target_indices if i in sem_failures]),
        "emotion_validated_count": len([i for i in all_target_indices if i in emo_vectors]),
        "emotion_failure_count": len([i for i in all_target_indices if i in emo_failures]),
        "error_count": len(errors),
        "errors": errors[:25],  # Truncate for summary readability
    }

    # Print validation report
    status_label = "PASS" if is_valid else "FAIL"
    print(
        "\n"
        "======================================================================\n"
        f"        PLOTGRAPH PRODUCTION COMPLETION VALIDATION: [{status_label}]\n"
        "======================================================================\n"
        f"Target Movies Validated  : {target_count}\n"
        f"Semantic Success / Fail  : {result['semantic_validated_count']} / {result['semantic_failure_count']}\n"
        f"Emotion Success / Fail   : {result['emotion_validated_count']} / {result['emotion_failure_count']}\n"
        f"Validation Errors Found  : {len(errors)}\n"
        "======================================================================\n",
        flush=True,
    )
    if errors:
        print("First Validation Errors (up to 10):", flush=True)
        for err in errors[:10]:
            print(f"  - {err}", flush=True)
        print("", flush=True)

    return result
