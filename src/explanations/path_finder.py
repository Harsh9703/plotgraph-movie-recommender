"""Knowledge graph indexing and path finding for recommendation explanations.

Provides fast in-memory indexing of KG-M metadata and KG-L semantic entities,
deterministic set-intersection path extraction, and recency-weighted historical
support movie ranking.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import numpy as np
import pandas as pd

from src.explanations.schemas import EvidencePath, HistoricalSupportMovie


RELATION_WEIGHTS = {
    # KG-M relations
    "directedBy": 10.0,
    "inCollection": 10.0,
    "starsActor": 3.0,
    "hasKeyword": 2.5,
    "hasGenre": 1.5,
    "releaseDecade": 0.5,
    # KG-L relations
    "hasTheme": 6.0,
    "basedOn": 6.0,
    "hasSetting": 4.0,
    "hasNarrativeDevice": 4.0,
    "hasProtagonistType": 3.5,
    "evokesEmotion": 2.0,
}

ENTITY_TYPE_MAP = {
    "directedBy": "director",
    "inCollection": "collection",
    "starsActor": "actor",
    "hasKeyword": "keyword",
    "hasGenre": "genre",
    "releaseDecade": "decade",
    "hasTheme": "theme",
    "basedOn": "source_material",
    "hasSetting": "setting",
    "hasNarrativeDevice": "narrative_device",
    "hasProtagonistType": "protagonist_type",
    "evokesEmotion": "emotion",
}


class MovieKnowledgeIndex:
    """Fast in-memory index for MovieLens-1M catalog movies, KG-M relations, and KG-L semantics."""

    def __init__(
        self,
        metadata_path: Union[str, Path] = "data/processed/ml-1m/metadata/movies_with_tmdb.parquet",
        semantic_path: Union[str, Path] = "data/processed/ml-1m/kg_l/full_extraction/semantic_validated.parquet",
        emotion_vectors_path: Union[str, Path] = "data/processed/ml-1m/kg_l/full_extraction/emotion_vectors.parquet",
        emotion_evidence_path: Optional[Union[str, Path]] = "data/processed/ml-1m/kg_l/full_extraction/emotion_validated_evidence.jsonl",
    ):
        self.metadata_path = Path(metadata_path)
        self.semantic_path = Path(semantic_path)
        self.emotion_vectors_path = Path(emotion_vectors_path)
        self.emotion_evidence_path = Path(emotion_evidence_path) if emotion_evidence_path else None

        self.movie_titles: Dict[int, str] = {}
        self.movie_overviews: Dict[int, str] = {}
        self.movie_ids: Dict[int, int] = {}  # movie_idx -> movie_id
        self.idx_to_movie_id: Dict[int, int] = {}
        self.movie_id_to_idx: Dict[int, int] = {}

        # Relation storage: movie_idx -> {relation_type -> Set[str]}
        self.kgm_entities: Dict[int, Dict[str, Set[str]]] = {}
        self.kgl_entities: Dict[int, Dict[str, Set[str]]] = {}

        # Continuous emotions: movie_idx -> {emotion_name -> float}
        self.emotion_vectors: Dict[int, Dict[str, float]] = {}
        self.prominent_emotions: Dict[int, Set[str]] = {}

        # Raw emotion evidence snippets from Qwen
        self.emotion_evidence: Dict[int, Dict[str, List[str]]] = {}

        # Inverted index for ultra-fast candidate support filtering
        self.entity_to_movies: Dict[Tuple[str, str], Set[int]] = {}
        self.cand_shared_catalog: Dict[int, Set[int]] = {}

        self._build_index()

    def _build_index(self):
        """Load and index all catalog entities and relations into fast lookup tables."""
        # 1. Load metadata (KG-M)
        if self.metadata_path.exists():
            df_meta = pd.read_parquet(self.metadata_path)
            for _, row in df_meta.iterrows():
                midx = int(row["movie_idx"])
                mid = int(row["movie_id"])
                title = str(row["ml_title"])
                overview = str(row["overview"]) if pd.notna(row.get("overview")) else ""

                self.movie_titles[midx] = title
                self.movie_overviews[midx] = overview
                self.movie_ids[midx] = mid
                self.idx_to_movie_id[midx] = mid
                self.movie_id_to_idx[mid] = midx

                rel_dict: Dict[str, Set[str]] = {
                    "directedBy": set(),
                    "inCollection": set(),
                    "starsActor": set(),
                    "hasGenre": set(),
                    "hasKeyword": set(),
                    "releaseDecade": set(),
                }

                # Director
                director = row.get("director")
                if director and pd.notna(director) and str(director).strip():
                    rel_dict["directedBy"].add(str(director).strip())

                # Collection
                collection = row.get("collection_name")
                if collection and pd.notna(collection) and str(collection).strip():
                    rel_dict["inCollection"].add(str(collection).strip())

                # Decade
                decade = row.get("release_decade")
                if decade and pd.notna(decade) and str(decade).strip():
                    rel_dict["releaseDecade"].add(str(decade).strip())

                # Actors
                actors = row.get("top_actors")
                if actors is not None and hasattr(actors, "__iter__"):
                    for act in actors:
                        if act and pd.notna(act) and str(act).strip():
                            rel_dict["starsActor"].add(str(act).strip())

                # Genres
                genres = row.get("tmdb_genres")
                if genres is not None and hasattr(genres, "__iter__") and len(genres) > 0:
                    for g in genres:
                        if g and pd.notna(g) and str(g).strip():
                            rel_dict["hasGenre"].add(str(g).strip())
                else:
                    ml_g = row.get("ml_genres")
                    if ml_g and pd.notna(ml_g):
                        for g in str(ml_g).split("|"):
                            if g.strip():
                                rel_dict["hasGenre"].add(g.strip())

                # Keywords
                kw = row.get("keywords")
                if kw is not None and hasattr(kw, "__iter__"):
                    for k in kw:
                        if k and pd.notna(k) and str(k).strip():
                            rel_dict["hasKeyword"].add(str(k).strip())

                self.kgm_entities[midx] = rel_dict

        # 2. Load KG-L semantics
        if self.semantic_path.exists():
            df_sem = pd.read_parquet(self.semantic_path)
            for _, row in df_sem.iterrows():
                midx = int(row["movie_idx"])
                sem_dict: Dict[str, Set[str]] = {
                    "hasTheme": set(),
                    "hasSetting": set(),
                    "hasNarrativeDevice": set(),
                    "hasProtagonistType": set(),
                    "basedOn": set(),
                    "evokesEmotion": set(),
                }

                for col, rel in [
                    ("themes", "hasTheme"),
                    ("settings", "hasSetting"),
                    ("narrative_devices", "hasNarrativeDevice"),
                    ("protagonist_types", "hasProtagonistType"),
                    ("based_on", "basedOn"),
                ]:
                    vals = row.get(col)
                    if vals is not None and hasattr(vals, "__iter__"):
                        for v in vals:
                            if v and pd.notna(v) and str(v).strip():
                                sem_dict[rel].add(str(v).strip().lower())

                self.kgl_entities[midx] = sem_dict

        # 3. Load Qwen continuous emotion vectors
        plutchik_dims = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
        if self.emotion_vectors_path.exists():
            df_emo = pd.read_parquet(self.emotion_vectors_path)
            for _, row in df_emo.iterrows():
                midx = int(row["movie_idx"])
                vec: Dict[str, float] = {}
                prominent: Set[str] = set()

                for d in plutchik_dims:
                    val = float(row.get(d, 0.0))
                    vec[d] = val
                    if val > 0.3:
                        prominent.add(d)

                self.emotion_vectors[midx] = vec
                self.prominent_emotions[midx] = prominent

                if midx in self.kgl_entities:
                    self.kgl_entities[midx]["evokesEmotion"] = set(prominent)

        # 4. Load emotion evidence if available
        if self.emotion_evidence_path and self.emotion_evidence_path.exists():
            with open(self.emotion_evidence_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    midx = int(item["movie_idx"])
                    emo_map: Dict[str, List[str]] = {}
                    for emo, data in item.get("emotions", {}).items():
                        ev_list = data.get("evidence", [])
                        if ev_list:
                            emo_map[emo] = [str(e) for e in ev_list]
                    self.emotion_evidence[midx] = emo_map

        # 5. Build inverted entity index for fast candidate support filtering
        for midx, rels in self.kgm_entities.items():
            for rel, vals in rels.items():
                for val in vals:
                    self.entity_to_movies.setdefault((rel, val), set()).add(midx)
        for midx, rels in self.kgl_entities.items():
            for rel, vals in rels.items():
                for val in vals:
                    self.entity_to_movies.setdefault((rel, val), set()).add(midx)

        for midx in self.movie_titles.keys():
            shared = set()
            for rel, vals in self.kgm_entities.get(midx, {}).items():
                for val in vals:
                    shared.update(self.entity_to_movies.get((rel, val), set()))
            for rel, vals in self.kgl_entities.get(midx, {}).items():
                for val in vals:
                    shared.update(self.entity_to_movies.get((rel, val), set()))
            self.cand_shared_catalog[midx] = shared

    def find_shared_paths(
        self,
        source_idx: int,
        target_idx: int,
    ) -> Tuple[List[EvidencePath], List[EvidencePath]]:
        """Extract all verified paths between source_idx and target_idx via KG-M and KG-L entities.

        Returns
        -------
        Tuple[List[EvidencePath], List[EvidencePath]]
            (kgm_paths, kgl_paths)
        """
        source_title = self.movie_titles.get(source_idx, f"Movie {source_idx}")
        kgm_paths: List[EvidencePath] = []
        kgl_paths: List[EvidencePath] = []

        # KG-M relations
        src_kgm = self.kgm_entities.get(source_idx, {})
        tgt_kgm = self.kgm_entities.get(target_idx, {})
        for rel, src_set in src_kgm.items():
            tgt_set = tgt_kgm.get(rel, set())
            shared = src_set & tgt_set
            for val in sorted(shared):
                kgm_paths.append(
                    EvidencePath(
                        source_movie_idx=source_idx,
                        source_movie_title=source_title,
                        relation_type=rel,
                        target_entity_type=ENTITY_TYPE_MAP.get(rel, rel),
                        target_entity_value=val,
                        kg_source="KG-M",
                    )
                )

        # KG-L relations
        src_kgl = self.kgl_entities.get(source_idx, {})
        tgt_kgl = self.kgl_entities.get(target_idx, {})
        for rel, src_set in src_kgl.items():
            tgt_set = tgt_kgl.get(rel, set())
            shared = src_set & tgt_set
            for val in sorted(shared):
                kgl_paths.append(
                    EvidencePath(
                        source_movie_idx=source_idx,
                        source_movie_title=source_title,
                        relation_type=rel,
                        target_entity_type=ENTITY_TYPE_MAP.get(rel, rel),
                        target_entity_value=val,
                        kg_source="KG-L",
                    )
                )

        return kgm_paths, kgl_paths

    def find_top_support_movies(
        self,
        candidate_idx: int,
        user_positive_history: List[Tuple[int, float, int]],  # (movie_idx, rating, timestamp)
        max_movies: int = 3,
        half_life_days: float = 365.0,
    ) -> List[HistoricalSupportMovie]:
        """Rank and select top 1–3 historical positive movies providing the strongest grounded support.

        Ranks historical movies by:
        - Entity specificity weight (director, collection, theme > broad genre/decade)
        - Recency weight: 2^(-delta_days / 365)
        - Rating weight: 1.25 for 5.0 vs 1.0 for 4.0
        """
        if not user_positive_history:
            return []

        # Reference timestamp is the latest positive interaction
        max_ts = max(ts for _, _, ts in user_positive_history)

        user_movie_map = {midx: (rating, ts) for midx, rating, ts in user_positive_history}
        cand_shared_map = getattr(self, "cand_shared_catalog", None)
        if cand_shared_map is not None:
            cand_shared = cand_shared_map.get(candidate_idx)
            relevant_midxs = set(user_movie_map.keys()) & cand_shared if cand_shared is not None else set(user_movie_map.keys())
        else:
            relevant_midxs = set(user_movie_map.keys())

        if not relevant_midxs:
            candidate_pool = user_positive_history[:5]
        else:
            candidate_pool = [(m, user_movie_map[m][0], user_movie_map[m][1]) for m in relevant_midxs]

        scored_candidates: List[Tuple[float, int, float, int, List[EvidencePath], List[EvidencePath]]] = []

        for midx, rating, ts in candidate_pool:
            kgm_paths, kgl_paths = self.find_shared_paths(midx, candidate_idx)
            total_paths = len(kgm_paths) + len(kgl_paths)

            # Calculate grounded score
            weight_score = 0.0
            for p in kgm_paths:
                weight_score += RELATION_WEIGHTS.get(p.relation_type, 1.0)
            for p in kgl_paths:
                weight_score += RELATION_WEIGHTS.get(p.relation_type, 1.0)

            # Rating boost
            rating_mult = 1.25 if rating >= 5.0 else 1.0

            # Recency discount
            delta_days = max(0.0, (max_ts - ts) / 86400.0)
            recency_mult = 2.0 ** (-delta_days / half_life_days)

            final_score = weight_score * rating_mult * recency_mult
            scored_candidates.append((final_score, midx, rating, ts, kgm_paths, kgl_paths))

        # Sort descending by score, tiebreak by rating, timestamp
        scored_candidates.sort(key=lambda x: (x[0], x[2], x[3]), reverse=True)

        results: List[HistoricalSupportMovie] = []
        for score, midx, rating, ts, kgm_p, kgl_p in scored_candidates[:max_movies]:
            title = self.movie_titles.get(midx, f"Movie {midx}")
            shared_summary = []
            for p in kgm_p + kgl_p:
                shared_summary.append(f"{p.target_entity_type}: {p.target_entity_value}")

            results.append(
                HistoricalSupportMovie(
                    movie_idx=midx,
                    title=title,
                    user_rating=rating,
                    timestamp=ts,
                    shared_entity_count=len(kgm_p) + len(kgl_p),
                    shared_entity_summary=shared_summary[:5],  # top 5 entities for summary
                )
            )

        return results
