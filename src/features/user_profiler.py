"""Leakage-Safe Recency-Weighted User Profile Construction (Phase 4A).

Builds deterministic preference profiles for all 6,040 MovieLens users
exclusively from positive TRAIN interactions (rating >= 4.0 / positive == 1):
- KG-M metadata entity preference distributions (genre, director, actor, keyword, collection, decade)
- KG-L semantic entity preference distributions (theme, setting, narrative_device, protagonist_type, source_material)
- Continuous 8-dimensional Plutchik emotion vectors (recency-weighted average of continuous scores)
- Exact entity sets seen in positive history (for explanation overlap counts)

STRICT LEAKAGE RULES:
- VAL and TEST interactions are strictly forbidden from profile construction.
- Reference timestamps are established strictly from each user's max TRAIN timestamp.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_POSITIVE_THRESHOLD,
    KGL_CATEGORIES,
    KGM_CATEGORIES,
    PLUTCHIK_EMOTIONS,
)
from .movie_indexer import MovieEntityIndex

logger = logging.getLogger(__name__)


def compute_age_days(timestamp: int, max_timestamp: int) -> float:
    """Compute interaction age in days relative to reference timestamp."""
    if timestamp > max_timestamp:
        raise ValueError(
            f"Temporal leakage detected: interaction timestamp {timestamp} "
            f"exceeds user reference timestamp {max_timestamp}."
        )
    return (max_timestamp - timestamp) / 86400.0


def compute_recency_weight(
    age_days: float,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Compute exponential decay weight with configurable half-life in days: 0.5 ** (age / half_life)."""
    if half_life_days <= 0.0:
        return 1.0
    return float(0.5 ** (age_days / half_life_days))


@dataclass
class UserProfile:
    """Encapsulates all leakage-safe preference profiles for a single user."""

    user_idx: int
    positive_train_count: int
    positive_train_movies: Set[int] = field(default_factory=set)
    max_train_timestamp: Optional[int] = None
    kgm_profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    kgm_entity_sets: Dict[str, Set[str]] = field(default_factory=dict)
    kgl_profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    kgl_entity_sets: Dict[str, Set[str]] = field(default_factory=dict)
    emotion_profile: Optional[np.ndarray] = None  # Shape (8,), float32 in [0, 1]
    emotion_profile_available: bool = False


class UserProfileIndex:
    """Collection of UserProfile instances across the user catalog."""

    def __init__(self, profiles: Optional[Dict[int, UserProfile]] = None):
        self.profiles: Dict[int, UserProfile] = profiles or {}

    def get(self, user_idx: int) -> Optional[UserProfile]:
        return self.profiles.get(user_idx)

    def __len__(self) -> int:
        return len(self.profiles)

    def __contains__(self, user_idx: int) -> bool:
        return user_idx in self.profiles

    def get_coverage_stats(self) -> Dict[str, Any]:
        """Compute coverage statistics across all user profiles."""
        total = len(self.profiles)
        if total == 0:
            return {}

        with_pos = sum(1 for p in self.profiles.values() if p.positive_train_count > 0)
        with_emo = sum(1 for p in self.profiles.values() if p.emotion_profile_available)
        with_kgm = {
            cat: sum(1 for p in self.profiles.values() if len(p.kgm_profiles.get(cat, {})) > 0)
            for cat in KGM_CATEGORIES
        }
        with_kgl = {
            cat: sum(1 for p in self.profiles.values() if len(p.kgl_profiles.get(cat, {})) > 0)
            for cat in KGL_CATEGORIES
        }

        return {
            "total_users": total,
            "users_with_positive_train": with_pos,
            "users_with_emotion_profile": with_emo,
            "emotion_profile_coverage_pct": round(with_emo / total * 100, 2),
            "kgm_profile_coverage": with_kgm,
            "kgl_profile_coverage": with_kgl,
        }

    def save_to_dir(self, output_dir: Union[str, Path]) -> None:
        """Persist user profiles to disk in compact, Git-ignored Parquet and JSON files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Emotion profiles table
        emo_rows = []
        for u_idx, prof in sorted(self.profiles.items()):
            row: Dict[str, Any] = {
                "user_idx": u_idx,
                "positive_train_count": prof.positive_train_count,
                "emotion_profile_available": prof.emotion_profile_available,
                "max_train_timestamp": prof.max_train_timestamp,
            }
            if prof.emotion_profile is not None:
                for idx, emo in enumerate(PLUTCHIK_EMOTIONS):
                    row[f"user_emotion_{emo}"] = float(prof.emotion_profile[idx])
            else:
                for emo in PLUTCHIK_EMOTIONS:
                    row[f"user_emotion_{emo}"] = np.nan
            emo_rows.append(row)

        df_emo = pd.DataFrame(emo_rows)
        df_emo.to_parquet(out_path / "user_emotion_profiles.parquet", index=False)

        # 2. KG-M and KG-L profiles (JSON for nested distributions)
        kgm_export = {
            u_idx: prof.kgm_profiles for u_idx, prof in self.profiles.items() if prof.kgm_profiles
        }
        with open(out_path / "user_kgm_profiles.json", "w", encoding="utf-8") as f:
            json.dump(kgm_export, f, indent=2)

        kgl_export = {
            u_idx: prof.kgl_profiles for u_idx, prof in self.profiles.items() if prof.kgl_profiles
        }
        with open(out_path / "user_kgl_profiles.json", "w", encoding="utf-8") as f:
            json.dump(kgl_export, f, indent=2)

        logger.info(f"Successfully saved user profiles to {out_path}")

    @classmethod
    def load_from_dir(cls, input_dir: Union[str, Path]) -> UserProfileIndex:
        """Load user profiles from directory."""
        in_path = Path(input_dir)
        df_emo = pd.read_parquet(in_path / "user_emotion_profiles.parquet")

        with open(in_path / "user_kgm_profiles.json", "r", encoding="utf-8") as f:
            kgm_data = {int(k): v for k, v in json.load(f).items()}

        with open(in_path / "user_kgl_profiles.json", "r", encoding="utf-8") as f:
            kgl_data = {int(k): v for k, v in json.load(f).items()}

        profiles: Dict[int, UserProfile] = {}
        for _, row in df_emo.iterrows():
            u_idx = int(row["user_idx"])
            avail = bool(row["emotion_profile_available"])
            emo_vec = None
            if avail:
                emo_vec = np.array(
                    [float(row[f"user_emotion_{e}"]) for e in PLUTCHIK_EMOTIONS],
                    dtype=np.float32,
                )

            u_kgm = kgm_data.get(u_idx, {})
            u_kgl = kgl_data.get(u_idx, {})

            prof = UserProfile(
                user_idx=u_idx,
                positive_train_count=int(row["positive_train_count"]),
                max_train_timestamp=int(row["max_train_timestamp"]) if pd.notna(row["max_train_timestamp"]) else None,
                kgm_profiles=u_kgm,
                kgm_entity_sets={cat: set(ents.keys()) for cat, ents in u_kgm.items()},
                kgl_profiles=u_kgl,
                kgl_entity_sets={cat: set(ents.keys()) for cat, ents in u_kgl.items()},
                emotion_profile=emo_vec,
                emotion_profile_available=avail,
            )
            profiles[u_idx] = prof

        return cls(profiles)


def build_user_profiles(
    train_df: pd.DataFrame,
    movie_index: MovieEntityIndex,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    total_users: int = 6040,
) -> UserProfileIndex:
    """Construct leakage-safe, recency-weighted user profiles from TRAIN split only.

    Args:
        train_df: Chronological TRAIN DataFrame containing columns:
                  ['user_idx', 'movie_idx', 'timestamp', 'rating'] (and optionally 'positive').
        movie_index: In-memory MovieEntityIndex providing KG-M, KG-L, and emotion profiles.
        half_life_days: Half-life parameter in days for exponential recency decay.
        positive_threshold: Minimum rating to consider interaction positive (default: 4.0).
        total_users: Total number of catalog users (default: 6040).

    Returns:
        UserProfileIndex containing profiles for all catalog users.
    """
    # -------------------------------------------------------------------------
    # Strict Leakage Assertion Guards
    # -------------------------------------------------------------------------
    col_names_lower = [str(c).lower() for c in train_df.columns]
    for forbidden in ["val", "test", "is_val", "is_test", "val_positive", "test_positive"]:
        if forbidden in col_names_lower:
            raise ValueError(
                f"Data leakage detected: train_df contains forbidden column '{forbidden}'. "
                "User profiles MUST be built strictly from the TRAIN split."
            )

    # Filter positive interactions strictly
    if "positive" in train_df.columns:
        pos_df = train_df[train_df["positive"] == 1].copy()
    else:
        pos_df = train_df[train_df["rating"] >= positive_threshold].copy()

    logger.info(
        f"Building user profiles from {len(pos_df):,} positive TRAIN interactions "
        f"across {pos_df['user_idx'].nunique():,} active users (half-life: {half_life_days} days)."
    )

    # Group positive history by user_idx: list of (movie_idx, timestamp)
    # Using groupby for fast in-memory extraction
    user_histories: Dict[int, List[Tuple[int, int]]] = {u: [] for u in range(total_users)}
    for row in pos_df[["user_idx", "movie_idx", "timestamp"]].itertuples(index=False):
        u_idx = int(row.user_idx)
        m_idx = int(row.movie_idx)
        t_val = int(row.timestamp)
        user_histories[u_idx].append((m_idx, t_val))

    profiles: Dict[int, UserProfile] = {}

    for u_idx in range(total_users):
        history = user_histories.get(u_idx, [])
        pos_count = len(history)

        if pos_count == 0:
            # Handle user with zero positive train interactions
            profiles[u_idx] = UserProfile(
                user_idx=u_idx,
                positive_train_count=0,
                positive_train_movies=set(),
                max_train_timestamp=None,
                kgm_profiles={cat: {} for cat in KGM_CATEGORIES},
                kgm_entity_sets={cat: set() for cat in KGM_CATEGORIES},
                kgl_profiles={cat: {} for cat in KGL_CATEGORIES},
                kgl_entity_sets={cat: set() for cat in KGL_CATEGORIES},
                emotion_profile=None,
                emotion_profile_available=False,
            )
            continue

        pos_movies = {m_idx for m_idx, _ in history}
        max_ts = max(t for _, t in history)

        # 1. Compute recency weights for positive interactions
        # raw_w_i = 0.5 ** (age_days / half_life)
        raw_weights: List[float] = []
        for m_idx, t_val in history:
            age_d = compute_age_days(t_val, max_ts)
            raw_w = compute_recency_weight(age_d, half_life_days=half_life_days)
            raw_weights.append(raw_w)

        total_raw_w = sum(raw_weights)
        norm_weights = [w / total_raw_w for w in raw_weights] if total_raw_w > 0 else [1.0 / pos_count] * pos_count

        # 2. KG-M Entity Profiles
        kgm_profiles: Dict[str, Dict[str, float]] = {}
        kgm_entity_sets: Dict[str, Set[str]] = {}

        for cat in KGM_CATEGORIES:
            raw_cat_weights: Dict[str, float] = {}
            for (m_idx, _), w_norm in zip(history, norm_weights):
                entities = movie_index.get_kgm_entities(m_idx, cat)
                for ent in entities:
                    raw_cat_weights[ent] = raw_cat_weights.get(ent, 0.0) + w_norm

            kgm_entity_sets[cat] = set(raw_cat_weights.keys())
            sum_cat_w = sum(raw_cat_weights.values())
            if sum_cat_w > 0:
                kgm_profiles[cat] = {ent: w / sum_cat_w for ent, w in raw_cat_weights.items()}
            else:
                kgm_profiles[cat] = {}

        # 3. KG-L Semantic Entity Profiles
        kgl_profiles: Dict[str, Dict[str, float]] = {}
        kgl_entity_sets: Dict[str, Set[str]] = {}

        for cat in KGL_CATEGORIES:
            raw_cat_weights: Dict[str, float] = {}
            for (m_idx, _), w_norm in zip(history, norm_weights):
                entities = movie_index.get_kgl_entities(m_idx, cat)
                for ent in entities:
                    raw_cat_weights[ent] = raw_cat_weights.get(ent, 0.0) + w_norm

            kgl_entity_sets[cat] = set(raw_cat_weights.keys())
            sum_cat_w = sum(raw_cat_weights.values())
            if sum_cat_w > 0:
                kgl_profiles[cat] = {ent: w / sum_cat_w for ent, w in raw_cat_weights.items()}
            else:
                kgl_profiles[cat] = {}

        # 4. Continuous Emotion Profile (recency-weighted average of continuous 8D vectors)
        # Omit movies without valid continuous emotion vectors (e.g. 22 failures)
        emo_weights: List[float] = []
        emo_vectors: List[np.ndarray] = []

        for (m_idx, _), raw_w in zip(history, raw_weights):
            vec = movie_index.get_emotion_vector(m_idx)
            if vec is not None:
                emo_weights.append(raw_w)
                emo_vectors.append(vec)

        if len(emo_vectors) > 0 and sum(emo_weights) > 0:
            sum_emo_w = sum(emo_weights)
            # Weighted average: sum(w_i * v_i) / sum(w_i)
            weighted_sum = np.zeros(8, dtype=np.float32)
            for w, v in zip(emo_weights, emo_vectors):
                weighted_sum += (w * v).astype(np.float32)
            avg_emo_vec = weighted_sum / float(sum_emo_w)
            # Ensure values strictly within [0.0, 1.0]
            avg_emo_vec = np.clip(avg_emo_vec, 0.0, 1.0)
            emotion_profile = avg_emo_vec
            emotion_profile_available = True
        else:
            emotion_profile = None
            emotion_profile_available = False

        profiles[u_idx] = UserProfile(
            user_idx=u_idx,
            positive_train_count=pos_count,
            positive_train_movies=pos_movies,
            max_train_timestamp=max_ts,
            kgm_profiles=kgm_profiles,
            kgm_entity_sets=kgm_entity_sets,
            kgl_profiles=kgl_profiles,
            kgl_entity_sets=kgl_entity_sets,
            emotion_profile=emotion_profile,
            emotion_profile_available=emotion_profile_available,
        )

    return UserProfileIndex(profiles)
