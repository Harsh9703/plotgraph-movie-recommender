"""Cold-start user profile builder with equal weighting for explicitly selected movies.

Constructs:
- KG-M metadata profile distributions (genre, director, actor, keyword, collection, decade)
- KG-L semantic profile distributions (theme, setting, narrative_device, protagonist_type, source_material)
- Continuous 8D Qwen emotion user vector
- Text-semantic user profiles (TF-IDF and SBERT)
- 8D NRC baseline emotion user vector

SEMANTICS:
Cold-start demo profiles assign equal weight (1/N) to explicitly selected liked movies
because no historical interaction timestamps exist. No timestamps or ratings are fabricated.
"""

from typing import List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy import sparse

from src.features.constants import (
    KGL_CATEGORIES,
    KGM_CATEGORIES,
    PLUTCHIK_EMOTIONS,
)
from src.features.movie_indexer import MovieEntityIndex
from src.features.nrc_profiler import NRCUserProfile, NRCUserProfileIndex
from src.features.text_encoder import SBERTMovieEncoder, TFIDFMovieEncoder
from src.features.text_profiler import TextUserProfile, TextUserProfileIndex
from src.features.user_profiler import UserProfile, UserProfileIndex


def build_cold_start_profiles(
    selected_movie_indices: List[int],
    movie_index: MovieEntityIndex,
    tfidf_encoder: TFIDFMovieEncoder,
    sbert_encoder: SBERTMovieEncoder,
    movie_nrc_df: pd.DataFrame,
    demo_user_idx: int = 0,
) -> Tuple[UserProfileIndex, TextUserProfileIndex, NRCUserProfileIndex]:
    """Construct equal-weight preference profiles for a new cold-start user.

    Parameters
    ----------
    selected_movie_indices : List[int]
        Indices of movies explicitly selected by the user.
    movie_index : MovieEntityIndex
        Indexed catalog entities and continuous emotion vectors.
    tfidf_encoder : TFIDFMovieEncoder
        Pre-fitted TF-IDF encoder and catalog matrix.
    sbert_encoder : SBERTMovieEncoder
        Pre-computed SBERT catalog embeddings.
    movie_nrc_df : pd.DataFrame
        Pre-computed NRC emotion vectors.
    demo_user_idx : int
        Internal index assigned to the temporary demo user (default 0).

    Returns
    -------
    Tuple[UserProfileIndex, TextUserProfileIndex, NRCUserProfileIndex]
        Container indices wrapping the cold-start user's profiles.
    """
    n_selected = len(selected_movie_indices)
    if n_selected == 0:
        raise ValueError("Cannot construct cold-start profile from empty movie selection.")

    equal_weight = 1.0 / n_selected

    # 1. KG-M and KG-L Entity Profiles
    kgm_profiles = {cat: {} for cat in KGM_CATEGORIES}
    kgm_entity_sets = {cat: set() for cat in KGM_CATEGORIES}
    kgl_profiles = {cat: {} for cat in KGL_CATEGORIES}
    kgl_entity_sets = {cat: set() for cat in KGL_CATEGORIES}

    for midx in selected_movie_indices:
        for cat in KGM_CATEGORIES:
            ents = movie_index.get_kgm_entities(midx, cat)
            kgm_entity_sets[cat].update(ents)
            for e in ents:
                kgm_profiles[cat][e] = kgm_profiles[cat].get(e, 0.0) + equal_weight

        for cat in KGL_CATEGORIES:
            ents = movie_index.get_kgl_entities(midx, cat)
            kgl_entity_sets[cat].update(ents)
            for e in ents:
                kgl_profiles[cat][e] = kgl_profiles[cat].get(e, 0.0) + equal_weight

    # Normalize distributions to sum to 1.0 per category if non-empty
    for cat in KGM_CATEGORIES:
        s = sum(kgm_profiles[cat].values())
        if s > 0:
            kgm_profiles[cat] = {k: v / s for k, v in kgm_profiles[cat].items()}

    for cat in KGL_CATEGORIES:
        s = sum(kgl_profiles[cat].values())
        if s > 0:
            kgl_profiles[cat] = {k: v / s for k, v in kgl_profiles[cat].items()}

    # 2. Qwen Emotion Profile
    valid_qwen_vecs: List[np.ndarray] = []
    for midx in selected_movie_indices:
        vec = movie_index.get_emotion_vector(midx)
        if vec is not None and np.all(np.isfinite(vec)):
            valid_qwen_vecs.append(vec)

    if valid_qwen_vecs:
        qwen_profile = np.mean(valid_qwen_vecs, axis=0).astype(np.float32)
        qwen_available = True
    else:
        qwen_profile = None
        qwen_available = False

    user_profile = UserProfile(
        user_idx=demo_user_idx,
        positive_train_count=n_selected,
        positive_train_movies=set(selected_movie_indices),
        max_train_timestamp=None,
        kgm_profiles=kgm_profiles,
        kgm_entity_sets=kgm_entity_sets,
        kgl_profiles=kgl_profiles,
        kgl_entity_sets=kgl_entity_sets,
        emotion_profile=qwen_profile,
        emotion_profile_available=qwen_available,
    )
    user_profile_index = UserProfileIndex({demo_user_idx: user_profile})

    # 3. Text User Profile (TF-IDF & SBERT)
    valid_tfidf_vecs = []
    for midx in selected_movie_indices:
        vec = tfidf_encoder.get_movie_vector(midx)
        if vec is not None:
            valid_tfidf_vecs.append(vec)

    if valid_tfidf_vecs:
        tfidf_sum = valid_tfidf_vecs[0].copy()
        for v in valid_tfidf_vecs[1:]:
            tfidf_sum = tfidf_sum + v
        norm = np.linalg.norm(tfidf_sum.data)
        if norm > 0:
            tfidf_user_vec = tfidf_sum / norm
        else:
            tfidf_user_vec = tfidf_sum
        tfidf_avail = True
    else:
        tfidf_user_vec = None
        tfidf_avail = False

    valid_sbert_vecs = []
    for midx in selected_movie_indices:
        emb = sbert_encoder.get_movie_embedding(midx)
        if emb is not None and np.all(np.isfinite(emb)):
            valid_sbert_vecs.append(emb)

    if valid_sbert_vecs:
        sbert_mean = np.mean(valid_sbert_vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(sbert_mean)
        if norm > 0:
            sbert_user_vec = (sbert_mean / norm).astype(np.float32)
        else:
            sbert_user_vec = sbert_mean
        sbert_avail = True
    else:
        sbert_user_vec = None
        sbert_avail = False

    text_user_profile = TextUserProfile(
        user_idx=demo_user_idx,
        tfidf_profile=tfidf_user_vec,
        tfidf_available=tfidf_avail,
        sbert_profile=sbert_user_vec,
        sbert_available=sbert_avail,
    )
    text_profile_index = TextUserProfileIndex(
        profiles={demo_user_idx: text_user_profile},
        sbert_dimension=sbert_encoder.embedding_dimension,
        vocab_size=tfidf_encoder.vocab_size,
        total_users=1,
    )

    # 4. NRC Emotion User Profile
    nrc_sub = movie_nrc_df[movie_nrc_df["movie_idx"].isin(selected_movie_indices)]
    nrc_valid = nrc_sub[nrc_sub["candidate_nrc_available"] == 1]

    if len(nrc_valid) > 0:
        nrc_cols = [f"candidate_nrc_{emo}" for emo in PLUTCHIK_EMOTIONS]
        raw_nrc_vec = nrc_valid[nrc_cols].to_numpy().mean(axis=0).astype(np.float32)
        s = raw_nrc_vec.sum()
        if s > 0:
            norm_nrc_vec = (raw_nrc_vec / s).astype(np.float32)
        else:
            norm_nrc_vec = raw_nrc_vec
        nrc_user_profile = NRCUserProfile(
            user_idx=demo_user_idx,
            nrc_profile=norm_nrc_vec,
            nrc_available=True,
            positive_train_count=n_selected,
            nrc_movie_count=len(nrc_valid),
        )
    else:
        nrc_user_profile = NRCUserProfile(
            user_idx=demo_user_idx,
            nrc_profile=None,
            nrc_available=False,
            positive_train_count=n_selected,
            nrc_movie_count=0,
        )

    nrc_profile_index = NRCUserProfileIndex(
        profiles={demo_user_idx: nrc_user_profile},
        total_users=1,
    )

    return user_profile_index, text_profile_index, nrc_profile_index
