"""Non-personalized popularity-based candidate recommendation baseline."""

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from .base import BaseCandidateGenerator


class PopularityRecommender(BaseCandidateGenerator):
    """Recommends movies ranked by global popularity in positive training interactions.

    Tie-breaking is strictly deterministic: items with equal popularity are sorted
    by movie_idx ascending.
    """

    def __init__(self, name: str = "Popularity") -> None:
        super().__init__(name=name)
        self.item_counts: np.ndarray = np.array([], dtype=np.int64)
        self.ranked_items: np.ndarray = np.array([], dtype=np.int64)

    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
    ) -> "PopularityRecommender":
        """Compute item popularity from positive training interactions only."""
        train_pos = self._extract_positive_train_interactions(
            train_df, num_users=num_users, num_items=num_items
        )

        # Count positive training interactions per movie
        item_indices = train_pos["movie_idx"].values.astype(np.int64)
        self.item_counts = np.bincount(item_indices, minlength=self.num_items)

        # Stable sort by negative counts: items with equal positive counts preserve
        # their natural index order (0, 1, ..., num_items - 1)
        self.ranked_items = np.argsort(-self.item_counts, kind="stable")
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:
        """Return top-K popular movies unseen by the user in positive training data."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before calling recommend.")

        seen = self.get_user_seen(user_idx) if exclude_seen else set()
        recommendations: List[int] = []

        for item in self.ranked_items:
            item_int = int(item)
            if item_int not in seen:
                recommendations.append(item_int)
                if len(recommendations) == k:
                    break

        return recommendations

    def recommend_all(
        self,
        user_indices: Sequence[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[int, List[int]]:
        """Fast batch recommendation for a collection of users."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before calling recommend_all.")

        result: Dict[int, List[int]] = {}
        for u in user_indices:
            result[u] = self.recommend(u, k=k, exclude_seen=exclude_seen)
        return result
