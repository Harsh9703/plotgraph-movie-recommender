"""NRC Emotion Lexicon Loader, Text Preprocessor, and Catalog Vectorizer (Phase 4C).

Provides:
1. NRCLexicon:
   - Loads the canonical NRC Word-Emotion Association Lexicon (EmoLex) via NRCLex 4.1.0.
   - Maps tokens to the 8 Plutchik emotions: joy, trust, fear, surprise, sadness, disgust, anger, anticipation.
   - Filters out non-Plutchik binary sentiment polarities (positive, negative).
2. Deterministic Preprocessing:
   - Unicode NFKD normalization, lowercasing, punctuation stripping, and regex word extraction.
3. 8-Dimensional Movie NRC Vector Construction:
   - Computes raw emotion token counts for overview text.
   - Normalizes to a continuous distribution summing to 1.0: count_e / sum(all_8_counts).
   - Distinguishes missing overviews and zero-match overviews via explicit availability flags and NaN vectors.
4. Caching and Serialization:
   - Precomputes and caches movie_nrc_vectors.parquet for fast downstream feature building and analysis.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .constants import PLUTCHIK_EMOTIONS

logger = logging.getLogger(__name__)

# Attribution and Citation Metadata
NRC_METADATA: Dict[str, str] = {
    "lexicon_name": "NRC Word-Emotion Association Lexicon (EmoLex)",
    "package_name": "NRCLex",
    "package_version": "4.1.0",
    "lexicon_version": "0.92",
    "authors": "Saif M. Mohammad and Peter D. Turney",
    "citation": (
        "Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a Word-Emotion "
        "Association Lexicon. Computational Intelligence, 29(3), 436-465."
    ),
    "license": "Research and non-commercial educational use (National Research Council Canada EmoLex License)",
    "mapping_description": "Maps unigram words to Plutchik's 8 basic emotions and 2 sentiment polarities.",
}


class NRCLexicon:
    """NRC emotion lexicon loader and deterministic movie overview scorer."""

    def __init__(self, custom_lexicon_path: Optional[Union[str, Path]] = None):
        """Initialize NRC lexicon with bundled EmoLex or custom path.

        Args:
            custom_lexicon_path: Optional path to custom NRC lexicon JSON/TSV file.
        """
        self.custom_path = custom_lexicon_path
        self.lexicon: Dict[str, Set[str]] = {}
        self._load_lexicon()

    def _load_lexicon(self) -> None:
        """Load lexicon dictionary mapping word -> set of Plutchik emotions."""
        raw_dict: Dict[str, list[str]] = {}
        if self.custom_path is not None and Path(self.custom_path).exists():
            import json

            logger.info(f"Loading custom NRC lexicon from {self.custom_path}...")
            with open(self.custom_path, "r", encoding="utf-8") as f:
                raw_dict = json.load(f)
        else:
            try:
                from nrclex.core import _load_bundled_lexicon

                logger.info("Loading bundled NRC EmoLex from NRCLex package...")
                raw_dict = _load_bundled_lexicon()
            except ImportError as e:
                raise ImportError(
                    "NRCLex is required for NRC lexicon features. Please install NRCLex (pip install NRCLex)."
                ) from e

        # Restrict strictly to the 8 Plutchik emotions; ignore binary 'positive'/'negative'
        plutchik_set = set(PLUTCHIK_EMOTIONS)
        cleaned: Dict[str, Set[str]] = {}
        for word, emotions in raw_dict.items():
            w = word.strip().lower()
            matched = set(emotions).intersection(plutchik_set)
            if matched:
                cleaned[w] = matched

        self.lexicon = cleaned
        logger.info(
            f"NRC Lexicon loaded: {len(self.lexicon):,} words with at least one Plutchik emotion association."
        )

    @staticmethod
    def preprocess_text(text: str) -> List[str]:
        """Deterministic text preprocessing: Unicode normalization, lowercase, word tokenization.

        Args:
            text: Raw input text.

        Returns:
            List of alphabetic lowercase tokens with punctuation stripped.
        """
        if not text or not isinstance(text, str):
            return []
        # Unicode NFKD normalization and ASCII encoding to strip diacritics/accents cleanly
        text_clean = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        # Extract lowercase alphabetic words (length >= 1)
        tokens = re.findall(r"[a-z]+", text_clean.lower())
        return tokens

    def score_tokens(self, tokens: List[str]) -> Tuple[Optional[np.ndarray], int, int, bool]:
        """Score a list of tokens against the 8 Plutchik emotions.

        Args:
            tokens: Preprocessed word tokens.

        Returns:
            Tuple of:
            - normalized_vector: np.ndarray of shape (8,) summing to 1.0, or None if unavailable
            - matched_token_count: number of tokens that matched at least one Plutchik emotion
            - total_emotion_counts: total emotion matches across all 8 dimensions
            - is_available: bool flag indicating whether a valid emotion distribution was derived
        """
        if not tokens:
            return None, 0, 0, False

        emotion_counts: Dict[str, int] = {e: 0 for e in PLUTCHIK_EMOTIONS}
        matched_tokens = 0

        for token in tokens:
            emotions = self.lexicon.get(token)
            if emotions:
                matched_tokens += 1
                for emo in emotions:
                    emotion_counts[emo] += 1

        total_emotion_matches = sum(emotion_counts.values())

        if total_emotion_matches > 0:
            # Normalized 8D distribution summing to 1.0
            raw_vec = np.array([emotion_counts[e] for e in PLUTCHIK_EMOTIONS], dtype=np.float32)
            norm_vec = raw_vec / float(total_emotion_matches)
            # Re-verify sum approximation
            norm_vec = norm_vec / float(norm_vec.sum())
            return norm_vec, matched_tokens, total_emotion_matches, True

        return None, 0, 0, False

    def score_overview(self, text: Optional[str]) -> Tuple[Optional[np.ndarray], int, int, bool]:
        """Preprocess and score a movie overview string.

        Args:
            text: Movie plot overview string.

        Returns:
            Tuple of (normalized_vector, matched_token_count, total_emotion_matches, is_available).
        """
        if text is None or not isinstance(text, str) or not text.strip():
            return None, 0, 0, False
        tokens = self.preprocess_text(text)
        return self.score_tokens(tokens)

    def compute_catalog_vectors(
        self,
        metadata_df: pd.DataFrame,
        catalog_size: int = 3883,
    ) -> pd.DataFrame:
        """Compute NRC emotion vectors for all catalog movies.

        Args:
            metadata_df: DataFrame containing 'movie_idx' and 'overview' columns.
            catalog_size: Total number of catalog movies (default: 3883).

        Returns:
            DataFrame with columns:
            ['movie_idx', 'candidate_has_overview', 'candidate_nrc_available',
             'candidate_nrc_matched_token_count', 'candidate_nrc_total_emotion_matches']
            + 8 candidate_nrc_{emotion} columns.
        """
        logger.info(f"Computing NRC emotion vectors across {catalog_size:,} catalog movies...")

        # Build overview lookup map
        overviews: Dict[int, str] = {}
        for _, row in metadata_df.iterrows():
            m_idx = int(row["movie_idx"])
            ov = row.get("overview")
            overviews[m_idx] = str(ov).strip() if pd.notna(ov) and str(ov).strip() else ""

        rows: List[Dict[str, Any]] = []

        for m_idx in range(catalog_size):
            text = overviews.get(m_idx, "")
            has_overview = int(bool(text))

            vec, matched_tokens, total_matches, is_avail = self.score_overview(text)

            row_data: Dict[str, Any] = {
                "movie_idx": m_idx,
                "candidate_has_overview": has_overview,
                "candidate_nrc_available": int(is_avail),
                "candidate_nrc_matched_token_count": int(matched_tokens),
                "candidate_nrc_total_emotion_matches": int(total_matches),
            }

            for idx, emo in enumerate(PLUTCHIK_EMOTIONS):
                col_name = f"candidate_nrc_{emo}"
                row_data[col_name] = float(vec[idx]) if is_avail and vec is not None else np.nan

            rows.append(row_data)

        df = pd.DataFrame(rows)
        usable_nrc = int(df["candidate_nrc_available"].sum())
        usable_ov = int(df["candidate_has_overview"].sum())
        logger.info(
            f"Catalog NRC vectors computed: {usable_nrc:,} / {catalog_size:,} movies have valid NRC vectors "
            f"({usable_ov:,} had overviews, {usable_ov - usable_nrc:,} overviews had zero NRC emotion matches)."
        )
        return df

    @staticmethod
    def save_vectors(df: pd.DataFrame, output_dir: Union[str, Path]) -> Path:
        """Persist movie NRC vectors to parquet."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / "movie_nrc_vectors.parquet"
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved movie NRC vectors to {file_path} ({file_path.stat().st_size / 1024:.1f} KB).")
        return file_path

    @staticmethod
    def load_vectors(input_dir: Union[str, Path]) -> pd.DataFrame:
        """Load persisted movie NRC vectors from parquet."""
        file_path = Path(input_dir) / "movie_nrc_vectors.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Movie NRC vectors file not found: {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded movie NRC vectors from {file_path} ({len(df):,} rows).")
        return df
