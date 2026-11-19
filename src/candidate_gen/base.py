"""Abstract base class for candidate generation recommendation models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Set, Tuple
import pandas as pd


class BaseCandidateGenerator(ABC):
    """Abstract base class for candidate recommenders.

    Enforces that models train strictly on positive implicit-feedback interactions
    from the training split (positive == 1, rating >= threshold).
    """

    def __init__(self, name: str = "BaseCandidateGenerator") -> None:
        self.name = name
        self.user_seen: Dict[int, Set[int]] = {}
        self.num_users: int = 0
        self.num_items: int = 0
        self.is_fitted: bool = False

    def _extract_positive_train_interactions(
        self,
        train_df: pd.DataFrame,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
    ) -> pd.DataFrame:
        """Filter and record positive training interactions only.

        Ratings below the positive threshold (positive != 1) are strictly ignored.
        """
        if "positive" not in train_df.columns:
            raise ValueError("train_df must contain a 'positive' column.")

        train_pos = train_df[train_df["positive"] == 1].copy()

        # Record user seen positive items from training history
        self.user_seen = (
            train_pos.groupby("user_idx")["movie_idx"].apply(set).to_dict()
        )

        max_user = int(train_df["user_idx"].max()) if not train_df.empty else 0
        max_item = int(train_df["movie_idx"].max()) if not train_df.empty else 0

        self.num_users = num_users if num_users is not None else max_user + 1
        self.num_items = num_items if num_items is not None else max_item + 1

        return train_pos

    def get_user_seen(self, user_idx: int) -> Set[int]:
        """Return the set of positive movie indices seen by this user in training."""
        return self.user_seen.get(user_idx, set())

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
    ) -> "BaseCandidateGenerator":
        """Fit model using positive training interactions.

        Args:
            train_df: DataFrame with 'user_idx', 'movie_idx', 'rating', 'positive'.
            num_users: Optional total number of users in universe.
            num_items: Optional total number of items in universe.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:
        """Generate top-K recommended movie indices for a single user.

        Args:
            user_idx: Internal user index.
            k: Number of recommendations to return.
            exclude_seen: If True, exclude movies the user positively interacted with in train.

        Returns:
            List of movie indices of length at most k.
        """
        pass

    def recommend_with_scores(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-K recommended movie indices with scores for a single user.

        Default implementation falls back to recommend() with pseudo-scores (descending ranks).
        """
        item_ids = self.recommend(user_idx, k=k, exclude_seen=exclude_seen)
        return [(item_id, float(len(item_ids) - rank)) for rank, item_id in enumerate(item_ids)]

    def recommend_all(
        self,
        user_indices: Sequence[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[int, List[int]]:
        """Generate top-K recommendations for a sequence of users.

        Subclasses may override this method for optimized batched matrix operations.

        Returns:
            Dict mapping user_idx to list of top-K movie indices.
        """
        return {
            u: self.recommend(u, k=k, exclude_seen=exclude_seen)
            for u in user_indices
        }

    def recommend_all_with_scores(
        self,
        user_indices: Sequence[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Generate top-K recommendations with scores for a sequence of users."""
        return {
            u: self.recommend_with_scores(u, k=k, exclude_seen=exclude_seen)
            for u in user_indices
        }
