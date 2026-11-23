"""In-Memory Movie Entity & Emotion Indexer for Fast Feature Engineering.

Pre-indexes all 3,883 MovieLens catalog movies by internal `movie_idx`:
- KG-M metadata entities (genre, director, actor, keyword, collection, decade)
- KG-L semantic entities (theme, setting, narrative_device, protagonist_type, source_material)
- Continuous 8-dimensional Plutchik emotion vectors
- Metadata and KG coverage flags

Avoids runtime SPARQL query bottlenecks by supporting sub-millisecond
in-memory lookups during candidate feature vectorization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from rdflib import Graph, Literal, RDF, RDFS, URIRef

from src.knowledge_graph.kg_l_contract import (
    canonicalize_semantic_entity,
    PLUTCHIK_EMOTIONS,
)
from src.knowledge_graph.namespaces import (
    CLASS_COLLECTION,
    CLASS_GENRE,
    CLASS_KEYWORD,
    CLASS_MOVIE,
    CLASS_NARRATIVE_DEVICE,
    CLASS_PERSON,
    CLASS_PROTAGONIST_TYPE,
    CLASS_RELEASE_DECADE,
    CLASS_SETTING,
    CLASS_SOURCE_MATERIAL,
    CLASS_THEME,
    PRED_ASSOCIATED_EMOTION,
    PRED_BASED_ON,
    PRED_DIRECTED_BY,
    PRED_EMOTION_INTENSITY,
    PRED_EVOKES_EMOTION,
    PRED_HAS_EMOTION_ASSOCIATION,
    PRED_HAS_GENRE,
    PRED_HAS_KEYWORD,
    PRED_HAS_NARRATIVE_DEVICE,
    PRED_HAS_PROTAGONIST_TYPE,
    PRED_HAS_SETTING,
    PRED_HAS_THEME,
    PRED_IN_COLLECTION,
    PRED_INTERNAL_INDEX,
    PRED_RELEASE_DECADE,
    PRED_STARS_ACTOR,
)
from .constants import KGL_CATEGORIES, KGM_CATEGORIES

logger = logging.getLogger(__name__)


@dataclass
class MovieMetadataItem:
    """Indexed representations for a single movie in the catalog."""

    movie_idx: int
    movie_id: int
    tmdb_id: Optional[int]
    title: str
    has_tmdb_link: bool
    has_overview: bool
    kgm_entities: Dict[str, List[str]] = field(default_factory=dict)
    kgl_entities: Dict[str, List[str]] = field(default_factory=dict)
    emotion_vector: Optional[np.ndarray] = None  # Shape (8,), float32 in [0, 1]
    has_kgm_metadata: bool = False
    has_kgl_semantics: bool = False
    has_emotion: bool = False


class MovieEntityIndex:
    """Fast in-memory index mapping movie_idx to KG-M, KG-L, and emotion profiles."""

    def __init__(self, catalog_size: int = 3883):
        self.catalog_size = catalog_size
        self._movies: Dict[int, MovieMetadataItem] = {}
        # Pre-computed matrix of continuous emotion vectors: shape (catalog_size, 8)
        self.emotion_matrix = np.full((catalog_size, 8), np.nan, dtype=np.float32)
        self.emotion_available_mask = np.zeros(catalog_size, dtype=bool)

    def add_movie(self, item: MovieMetadataItem) -> None:
        """Register an indexed movie item."""
        self._movies[item.movie_idx] = item
        if item.emotion_vector is not None:
            self.emotion_matrix[item.movie_idx] = item.emotion_vector
            self.emotion_available_mask[item.movie_idx] = True

    def get(self, movie_idx: int) -> Optional[MovieMetadataItem]:
        """Retrieve indexed movie item by movie_idx."""
        return self._movies.get(movie_idx)

    def __len__(self) -> int:
        return len(self._movies)

    def __contains__(self, movie_idx: int) -> bool:
        return movie_idx in self._movies

    def get_kgm_entities(self, movie_idx: int, category: str) -> List[str]:
        """Return list of entities for a given KG-M category."""
        item = self._movies.get(movie_idx)
        if item is None:
            return []
        return item.kgm_entities.get(category, [])

    def get_kgl_entities(self, movie_idx: int, category: str) -> List[str]:
        """Return list of entities for a given KG-L semantic category."""
        item = self._movies.get(movie_idx)
        if item is None:
            return []
        return item.kgl_entities.get(category, [])

    def get_emotion_vector(self, movie_idx: int) -> Optional[np.ndarray]:
        """Return continuous 8D emotion vector for movie_idx if available."""
        item = self._movies.get(movie_idx)
        if item is None:
            return None
        return item.emotion_vector

    @classmethod
    def from_dataframes(
        cls,
        metadata_df: pd.DataFrame,
        semantic_df: Optional[pd.DataFrame] = None,
        emotion_df: Optional[pd.DataFrame] = None,
        top_k_actors: int = 5,
    ) -> MovieEntityIndex:
        """Build MovieEntityIndex directly from validated production DataFrames/Parquets.

        Sub-millisecond construction without Turtle parsing overhead.
        """
        catalog_size = len(metadata_df)
        index = cls(catalog_size=catalog_size)

        # Pre-index semantics by movie_idx
        semantics_by_idx: Dict[int, Dict[str, List[str]]] = {}
        if semantic_df is not None:
            for _, s_row in semantic_df.iterrows():
                m_idx = int(s_row["movie_idx"])
                sem_map: Dict[str, List[str]] = {}
                for cat, col_name in [
                    ("theme", "themes"),
                    ("setting", "settings"),
                    ("narrative_device", "narrative_devices"),
                    ("protagonist_type", "protagonist_types"),
                    ("source_material", "based_on"),
                ]:
                    raw_items = s_row.get(col_name)
                    if isinstance(raw_items, (list, tuple, np.ndarray)):
                        sem_map[cat] = [
                            canonicalize_semantic_entity(str(x))
                            for x in raw_items
                            if str(x).strip()
                        ]
                    else:
                        sem_map[cat] = []
                semantics_by_idx[m_idx] = sem_map

        # Pre-index continuous emotion vectors by movie_idx
        emotions_by_idx: Dict[int, np.ndarray] = {}
        if emotion_df is not None:
            for _, e_row in emotion_df.iterrows():
                m_idx = int(e_row["movie_idx"])
                vec = np.array(
                    [float(e_row[emo]) for emo in PLUTCHIK_EMOTIONS],
                    dtype=np.float32,
                )
                emotions_by_idx[m_idx] = vec

        # Process each movie in metadata
        for _, row in metadata_df.iterrows():
            m_idx = int(row["movie_idx"])
            m_id = int(row.get("movie_id", m_idx + 1))
            raw_tmdb = row.get("tmdb_id")
            tmdb_id = int(raw_tmdb) if pd.notna(raw_tmdb) else None
            ml_title = str(row.get("ml_title", f"Movie {m_idx}")).strip()

            has_tmdb_link = bool(row.get("has_tmdb_link", tmdb_id is not None))
            has_overview = bool(
                row.get("has_overview", pd.notna(row.get("overview")) and bool(str(row.get("overview")).strip()))
            )

            # 1. Build KG-M entities
            kgm_map: Dict[str, List[str]] = {}

            # Genres
            genres = row.get("tmdb_genres")
            if isinstance(genres, (list, tuple, np.ndarray)):
                kgm_map["genre"] = [
                    canonicalize_semantic_entity(str(g))
                    for g in genres
                    if str(g).strip() and str(g).strip().lower() != "none"
                ]
            else:
                kgm_map["genre"] = []

            # Director
            director = row.get("director")
            if pd.notna(director) and str(director).strip() and str(director).strip().lower() != "none":
                kgm_map["director"] = [canonicalize_semantic_entity(str(director))]
            else:
                kgm_map["director"] = []

            # Top Actors (up to top_k_actors)
            actors = row.get("top_actors")
            if isinstance(actors, (list, tuple, np.ndarray)):
                kgm_map["actor"] = [
                    canonicalize_semantic_entity(str(a))
                    for a in actors[:top_k_actors]
                    if str(a).strip() and str(a).strip().lower() != "none"
                ]
            else:
                kgm_map["actor"] = []

            # Keywords
            keywords = row.get("keywords")
            if isinstance(keywords, (list, tuple, np.ndarray)):
                kgm_map["keyword"] = [
                    canonicalize_semantic_entity(str(k))
                    for k in keywords
                    if str(k).strip() and str(k).strip().lower() != "none"
                ]
            else:
                kgm_map["keyword"] = []

            # Collection
            collection = row.get("collection_name")
            if pd.notna(collection) and str(collection).strip() and str(collection).strip().lower() != "none":
                kgm_map["collection"] = [canonicalize_semantic_entity(str(collection))]
            else:
                kgm_map["collection"] = []

            # Release Decade
            decade = row.get("release_decade")
            if pd.notna(decade) and str(decade).strip() and str(decade).strip().lower() != "none":
                kgm_map["decade"] = [canonicalize_semantic_entity(str(decade))]
            else:
                kgm_map["decade"] = []

            # 2. Retrieve KG-L entities
            kgl_map = semantics_by_idx.get(m_idx, {cat: [] for cat in KGL_CATEGORIES})

            # 3. Retrieve emotion vector
            emo_vec = emotions_by_idx.get(m_idx, None)

            # 4. Coverage flags
            has_kgm = any(len(entities) > 0 for entities in kgm_map.values())
            has_kgl = any(len(entities) > 0 for entities in kgl_map.values())
            has_emo = emo_vec is not None

            item = MovieMetadataItem(
                movie_idx=m_idx,
                movie_id=m_id,
                tmdb_id=tmdb_id,
                title=ml_title,
                has_tmdb_link=has_tmdb_link,
                has_overview=has_overview,
                kgm_entities=kgm_map,
                kgl_entities=kgl_map,
                emotion_vector=emo_vec,
                has_kgm_metadata=has_kgm,
                has_kgl_semantics=has_kgl,
                has_emotion=has_emo,
            )
            index.add_movie(item)

        return index

    @classmethod
    def from_graphs(
        cls,
        kg_m_graph_or_path: Union[Graph, str, Path],
        kg_l_graph_or_path: Union[Graph, str, Path],
        emotion_df_or_path: Union[pd.DataFrame, str, Path],
        catalog_size: int = 3883,
    ) -> MovieEntityIndex:
        """Build MovieEntityIndex by parsing RDF Turtle graphs (for cross-validation)."""
        # Load KG-M
        if isinstance(kg_m_graph_or_path, (str, Path)):
            g_m = Graph()
            g_m.parse(str(kg_m_graph_or_path), format="turtle")
        else:
            g_m = kg_m_graph_or_path

        # Load KG-L
        if isinstance(kg_l_graph_or_path, (str, Path)):
            g_l = Graph()
            g_l.parse(str(kg_l_graph_or_path), format="turtle")
        else:
            g_l = kg_l_graph_or_path

        # Load Emotion DataFrame
        if isinstance(emotion_df_or_path, (str, Path)):
            p = Path(emotion_df_or_path)
            if p.suffix == ".parquet":
                emotion_df = pd.read_parquet(p)
            else:
                emotion_df = pd.read_json(p, lines=True)
        else:
            emotion_df = emotion_df_or_path

        index = cls(catalog_size=catalog_size)

        # Pre-index continuous emotions
        emotions_by_idx: Dict[int, np.ndarray] = {}
        for _, e_row in emotion_df.iterrows():
            m_idx = int(e_row["movie_idx"])
            emotions_by_idx[m_idx] = np.array(
                [float(e_row[emo]) for emo in PLUTCHIK_EMOTIONS],
                dtype=np.float32,
            )

        # Index movie resources by internal movie_idx from g_m
        uri_to_idx: Dict[URIRef, int] = {}
        idx_to_meta: Dict[int, Dict[str, Any]] = {}

        for m_uri, _, idx_lit in g_m.triples((None, PRED_INTERNAL_INDEX, None)):
            m_idx = int(idx_lit)
            uri_to_idx[m_uri] = m_idx
            idx_to_meta[m_idx] = {
                "uri": m_uri,
                "title": "",
                "movie_id": 0,
                "tmdb_id": None,
                "has_tmdb_link": False,
                "kgm": {c: [] for c in KGM_CATEGORIES},
                "kgl": {c: [] for c in KGL_CATEGORIES},
            }

        # Query KG-M predicates
        for m_uri, m_idx in uri_to_idx.items():
            meta = idx_to_meta[m_idx]
            # Title
            for _, _, title_lit in g_m.triples((m_uri, RDFS.label, None)):
                meta["title"] = str(title_lit)
                break

            # KG-M relations
            pred_map = {
                "genre": PRED_HAS_GENRE,
                "director": PRED_DIRECTED_BY,
                "actor": PRED_STARS_ACTOR,
                "keyword": PRED_HAS_KEYWORD,
                "collection": PRED_IN_COLLECTION,
                "decade": PRED_RELEASE_DECADE,
            }
            for cat, pred in pred_map.items():
                for _, _, obj in g_m.triples((m_uri, pred, None)):
                    # Extract label of object if present, else URI slug
                    labels = list(g_m.objects(obj, RDFS.label))
                    if labels:
                        meta["kgm"][cat].append(canonicalize_semantic_entity(str(labels[0])))
                    else:
                        slug = str(obj).split("/")[-1].split("#")[-1]
                        meta["kgm"][cat].append(canonicalize_semantic_entity(slug))

        # Query KG-L predicates
        kgl_pred_map = {
            "theme": PRED_HAS_THEME,
            "setting": PRED_HAS_SETTING,
            "narrative_device": PRED_HAS_NARRATIVE_DEVICE,
            "protagonist_type": PRED_HAS_PROTAGONIST_TYPE,
            "source_material": PRED_BASED_ON,
        }
        for m_uri, m_idx in uri_to_idx.items():
            meta = idx_to_meta[m_idx]
            for cat, pred in kgl_pred_map.items():
                for _, _, obj in g_l.triples((m_uri, pred, None)):
                    labels = list(g_l.objects(obj, RDFS.label))
                    if labels:
                        meta["kgl"][cat].append(canonicalize_semantic_entity(str(labels[0])))
                    else:
                        slug = str(obj).split("/")[-1].split("#")[-1]
                        meta["kgl"][cat].append(canonicalize_semantic_entity(slug))

        for m_idx, meta in idx_to_meta.items():
            emo_vec = emotions_by_idx.get(m_idx)
            has_kgm = any(len(ent) > 0 for ent in meta["kgm"].values())
            has_kgl = any(len(ent) > 0 for ent in meta["kgl"].values())
            item = MovieMetadataItem(
                movie_idx=m_idx,
                movie_id=meta.get("movie_id", m_idx + 1),
                tmdb_id=meta.get("tmdb_id"),
                title=meta.get("title", f"Movie {m_idx}"),
                has_tmdb_link=True,
                has_overview=has_kgl,
                kgm_entities=meta["kgm"],
                kgl_entities=meta["kgl"],
                emotion_vector=emo_vec,
                has_kgm_metadata=has_kgm,
                has_kgl_semantics=has_kgl,
                has_emotion=emo_vec is not None,
            )
            index.add_movie(item)

        return index
