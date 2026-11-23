"""Text Representation and Embedding Encoders for Movie Overviews (Phase 4B).

Provides:
1. TFIDFMovieEncoder:
   - Fits on catalog movie overviews with configurable n-grams and stop words.
   - Produces unit L2-normalized sparse TF-IDF vectors for each movie.
2. SBERTMovieEncoder:
   - Compact sentence-transformers inference (all-MiniLM-L6-v2, 384 dims, CPU-friendly).
   - Produces unit L2-normalized dense embeddings for all usable overviews.
3. FAISSIndexHelper:
   - Optional FAISS index (IndexFlatIP) for fast nearest-neighbor retrieval.
   - Graceful pure NumPy fallback if FAISS is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def is_faiss_available() -> bool:
    """Check if faiss is available without eagerly importing it at module load."""
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


class TFIDFMovieEncoder:
    """Deterministic TF-IDF encoder for catalog movie overviews."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.8,
        stop_words: str = "english",
        sublinear_tf: bool = False,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words
        self.sublinear_tf = sublinear_tf

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            sublinear_tf=sublinear_tf,
            norm="l2",
            lowercase=True,
        )
        self.is_fitted: bool = False
        self.catalog_size: int = 0
        self.vocab_size: int = 0
        self.movie_tfidf_matrix: Optional[sparse.csr_matrix] = None
        self.has_text_mask: Optional[np.ndarray] = None

    def fit_transform(
        self,
        overviews_by_idx: Dict[int, str],
        catalog_size: int = 3883,
    ) -> sparse.csr_matrix:
        """Fit vectorizer on available movie overviews and build catalog TF-IDF matrix.

        Args:
            overviews_by_idx: Map of movie_idx -> overview string.
            catalog_size: Total catalog size (default 3883).

        Returns:
            scipy.sparse.csr_matrix of shape (catalog_size, vocab_size).
        """
        self.catalog_size = catalog_size
        self.has_text_mask = np.zeros(catalog_size, dtype=bool)

        # Collect corpus strictly from catalog movies with non-empty text
        corpus_texts: List[str] = []
        corpus_indices: List[int] = []

        for m_idx in range(catalog_size):
            text = overviews_by_idx.get(m_idx, "")
            if text and isinstance(text, str) and text.strip():
                corpus_texts.append(text.strip())
                corpus_indices.append(m_idx)
                self.has_text_mask[m_idx] = True

        logger.info(
            f"Fitting TF-IDF vectorizer on {len(corpus_texts):,} catalog movie overviews "
            f"(ngram_range={self.ngram_range}, min_df={self.min_df})..."
        )
        fitted_matrix = self.vectorizer.fit_transform(corpus_texts)
        self.vocab_size = fitted_matrix.shape[1]
        self.is_fitted = True
        logger.info(f"TF-IDF vocabulary constructed with {self.vocab_size:,} distinct terms.")

        # Construct full catalog sparse matrix: shape (catalog_size, vocab_size)
        # Allocate rows for valid movies
        full_rows: List[sparse.csr_matrix] = []
        c_ptr = 0
        zero_row = sparse.csr_matrix((1, self.vocab_size), dtype=np.float32)

        for m_idx in range(catalog_size):
            if self.has_text_mask[m_idx]:
                full_rows.append(fitted_matrix[c_ptr].astype(np.float32))
                c_ptr += 1
            else:
                full_rows.append(zero_row)

        self.movie_tfidf_matrix = sparse.vstack(full_rows, format="csr")
        return self.movie_tfidf_matrix

    def get_movie_vector(self, movie_idx: int) -> Optional[sparse.csr_matrix]:
        """Return unit-normalized TF-IDF row vector for movie_idx if available."""
        if not self.is_fitted or self.movie_tfidf_matrix is None:
            raise RuntimeError("TFIDFMovieEncoder is not fitted.")
        if self.has_text_mask is not None and not self.has_text_mask[movie_idx]:
            return None
        return self.movie_tfidf_matrix[movie_idx]

    def save(self, output_dir: Union[str, Path]) -> None:
        """Persist vectorizer and movie TF-IDF matrix."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, out_path / "tfidf_vectorizer.joblib")
        if self.movie_tfidf_matrix is not None:
            sparse.save_npz(out_path / "movie_tfidf_matrix.npz", self.movie_tfidf_matrix)
        if self.has_text_mask is not None:
            np.save(out_path / "tfidf_has_text_mask.npy", self.has_text_mask)
        logger.info(f"Saved TF-IDF artifacts to {out_path}")

    @classmethod
    def load(cls, input_dir: Union[str, Path]) -> TFIDFMovieEncoder:
        """Load fitted TF-IDF artifacts from disk."""
        in_path = Path(input_dir)
        vectorizer = joblib.load(in_path / "tfidf_vectorizer.joblib")
        movie_matrix = sparse.load_npz(in_path / "movie_tfidf_matrix.npz")
        has_text_mask = np.load(in_path / "tfidf_has_text_mask.npy")

        encoder = cls(
            ngram_range=vectorizer.ngram_range,
            min_df=vectorizer.min_df,
            max_df=vectorizer.max_df,
            stop_words=vectorizer.stop_words,
            sublinear_tf=vectorizer.sublinear_tf,
        )
        encoder.vectorizer = vectorizer
        encoder.is_fitted = True
        encoder.movie_tfidf_matrix = movie_matrix
        encoder.has_text_mask = has_text_mask
        encoder.catalog_size = movie_matrix.shape[0]
        encoder.vocab_size = movie_matrix.shape[1]
        return encoder


class SBERTMovieEncoder:
    """Sentence-BERT encoder for catalog movie plot overviews."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self.embedding_dimension: int = 384
        self.embeddings_matrix: Optional[np.ndarray] = None  # shape (catalog_size, dim)
        self.has_embedding_mask: Optional[np.ndarray] = None  # shape (catalog_size,)

    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading SBERT model '{self.model_name}' on device '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Embedding dimension
            if hasattr(self._model, "get_sentence_embedding_dimension"):
                self.embedding_dimension = self._model.get_sentence_embedding_dimension()
            elif hasattr(self._model, "get_embedding_dimension"):
                self.embedding_dimension = self._model.get_embedding_dimension()
        return self._model

    def encode_catalog(
        self,
        overviews_by_idx: Dict[int, str],
        catalog_size: int = 3883,
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Encode all available movie overviews into unit L2-normalized vectors.

        Args:
            overviews_by_idx: Map of movie_idx -> overview string.
            catalog_size: Total catalog size (default 3883).
            batch_size: Encoding batch size.
            show_progress_bar: Whether to display tqdm progress bar.

        Returns:
            np.ndarray of shape (catalog_size, embedding_dim), dtype float32.
        """
        model = self._get_model()
        self.has_embedding_mask = np.zeros(catalog_size, dtype=bool)
        self.embeddings_matrix = np.full(
            (catalog_size, self.embedding_dimension), np.nan, dtype=np.float32
        )

        valid_indices: List[int] = []
        valid_texts: List[str] = []

        for m_idx in range(catalog_size):
            text = overviews_by_idx.get(m_idx, "")
            if text and isinstance(text, str) and text.strip():
                valid_indices.append(m_idx)
                valid_texts.append(text.strip())
                self.has_embedding_mask[m_idx] = True

        logger.info(
            f"Encoding {len(valid_texts):,} catalog movie overviews using SBERT "
            f"'{self.model_name}' (batch_size={batch_size})..."
        )
        raw_embeddings = model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,  # Crucial: unit L2-norm
            convert_to_numpy=True,
        )

        for idx, m_idx in enumerate(valid_indices):
            self.embeddings_matrix[m_idx] = raw_embeddings[idx].astype(np.float32)

        logger.info(
            f"SBERT catalog encoding complete: {len(valid_indices):,} valid embeddings "
            f"(dim={self.embedding_dimension})."
        )
        return self.embeddings_matrix

    def get_movie_embedding(self, movie_idx: int) -> Optional[np.ndarray]:
        """Return unit-normalized SBERT embedding for movie_idx if available."""
        if self.embeddings_matrix is None or self.has_embedding_mask is None:
            raise RuntimeError("SBERTMovieEncoder has not encoded catalog yet.")
        if not self.has_embedding_mask[movie_idx]:
            return None
        return self.embeddings_matrix[movie_idx]

    def save(self, output_dir: Union[str, Path]) -> None:
        """Persist embeddings matrix and metadata index."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if self.embeddings_matrix is not None:
            np.save(out_path / "movie_sbert_embeddings.npy", self.embeddings_matrix)
        if self.has_embedding_mask is not None:
            np.save(out_path / "sbert_has_embedding_mask.npy", self.has_embedding_mask)
        logger.info(f"Saved SBERT embeddings to {out_path}")

    @classmethod
    def load(
        cls,
        input_dir: Union[str, Path],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> SBERTMovieEncoder:
        """Load precomputed SBERT embeddings from disk."""
        in_path = Path(input_dir)
        embeddings = np.load(in_path / "movie_sbert_embeddings.npy")
        mask = np.load(in_path / "sbert_has_embedding_mask.npy")

        encoder = cls(model_name=model_name)
        encoder.embeddings_matrix = embeddings
        encoder.has_embedding_mask = mask
        encoder.embedding_dimension = embeddings.shape[1]
        return encoder


class FAISSIndexHelper:
    """Optional FAISS index for SBERT embeddings with pure NumPy fallback."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.valid_movie_indices: List[int] = []
        self.valid_embeddings: Optional[np.ndarray] = None
        self.index = None
        self.uses_faiss: bool = False

    def build_index(
        self,
        embeddings_matrix: np.ndarray,
        has_embedding_mask: np.ndarray,
    ) -> None:
        """Build FAISS IndexFlatIP (Inner Product) on unit-normalized embeddings."""
        self.valid_movie_indices = [
            int(idx) for idx, valid in enumerate(has_embedding_mask) if valid
        ]
        self.valid_embeddings = embeddings_matrix[self.valid_movie_indices].astype(np.float32)

        if is_faiss_available():
            try:
                import faiss

                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(self.valid_embeddings)
                self.uses_faiss = True
                logger.info(
                    f"Built FAISS IndexFlatIP with {self.index.ntotal:,} vectors (dim={self.dimension})."
                )
                return
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS index: {e}. Falling back to NumPy.")

        self.uses_faiss = False
        logger.info(
            f"FAISS unavailable or skipped. Using NumPy vectorized inner product fallback "
            f"for {len(self.valid_movie_indices):,} vectors."
        )

    def search_nearest(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Search nearest catalog movies by cosine similarity (inner product).

        Returns:
            List of (movie_idx, similarity) sorted by similarity desc.
        """
        if self.valid_embeddings is None or len(self.valid_movie_indices) == 0:
            return []

        q = query_embedding.astype(np.float32).reshape(1, -1)
        # Ensure query is unit normalized
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm

        if self.uses_faiss and self.index is not None:
            distances, indices = self.index.search(q, top_k)
            results: List[Tuple[int, float]] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.valid_movie_indices):
                    m_idx = self.valid_movie_indices[idx]
                    results.append((m_idx, float(dist)))
            return results

        # NumPy fallback
        scores = np.dot(self.valid_embeddings, q.T).flatten()
        top_ranks = np.argsort(-scores)[:top_k]
        return [
            (self.valid_movie_indices[r_idx], float(scores[r_idx]))
            for r_idx in top_ranks
        ]
