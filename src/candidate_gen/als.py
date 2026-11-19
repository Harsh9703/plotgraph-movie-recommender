"""Implicit Alternating Least Squares (ALS) candidate recommender."""

from typing import Dict, List, Optional, Sequence, Tuple
import implicit.als
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .base import BaseCandidateGenerator


class ALSRecommender(BaseCandidateGenerator):
    """Matrix Factorization using Implicit Alternating Least Squares (Hu et al., ICDM 2008)."""

    def __init__(
        self,
        factors: int = 32,
        regularization: float = 0.01,
        iterations: int = 15,
        random_state: int = 42,
        name: str = "ALS",
    ) -> None:
        super().__init__(name=name)
        self.factors = int(factors)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.random_state = int(random_state)
        self.model: Optional[implicit.als.AlternatingLeastSquares] = None
        self.X_train: Optional[sp.csr_matrix] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
    ) -> "ALSRecommender":
        """Fit ALS model on positive training interactions."""
        train_pos = self._extract_positive_train_interactions(
            train_df, num_users=num_users, num_items=num_items
        )

        rows = train_pos["user_idx"].values.astype(np.int64)
        cols = train_pos["movie_idx"].values.astype(np.int64)
        data = np.ones(len(train_pos), dtype=np.float32)

        self.X_train = sp.csr_matrix(
            (data, (rows, cols)), shape=(self.num_users, self.num_items), dtype=np.float32
        )

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )

        # implicit expects user-item matrix of shape (users, items)
        self.model.fit(self.X_train, show_progress=False)
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:
        if not self.is_fitted or self.model is None or self.X_train is None:
            raise RuntimeError(f"{self.name} must be fitted before calling recommend.")

        if user_idx >= self.num_users:
            return list(range(k))

        u_row = self.X_train[user_idx]
        ids, _ = self.model.recommend(
            user_idx,
            u_row,
            N=k,
            filter_already_liked_items=exclude_seen,
        )
        return [int(x) for x in ids]

    def recommend_with_scores(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        if not self.is_fitted or self.model is None or self.X_train is None:
            raise RuntimeError(f"{self.name} must be fitted before calling recommend_with_scores.")

        if user_idx >= self.num_users:
            return [(item, 0.0) for item in range(k)]

        u_row = self.X_train[user_idx]
        ids, scores = self.model.recommend(
            user_idx,
            u_row,
            N=k,
            filter_already_liked_items=exclude_seen,
        )
        return [(int(item_id), float(score)) for item_id, score in zip(ids, scores)]

    def recommend_all(
        self,
        user_indices: Sequence[int],
        k: int = 10,
        exclude_seen: bool = True,
        batch_size: int = 2000,
    ) -> Dict[int, List[int]]:
        """Fast batch recommendation using implicit's native batch recommend."""
        if not self.is_fitted or self.model is None or self.X_train is None:
            raise RuntimeError(f"{self.name} must be fitted before calling recommend_all.")

        user_list = list(user_indices)
        results: Dict[int, List[int]] = {}

        for start_idx in range(0, len(user_list), batch_size):
            batch_users = user_list[start_idx : start_idx + batch_size]
            valid_users = [u for u in batch_users if u < self.num_users]

            if valid_users:
                sub_X = self.X_train[valid_users]
                ids, _ = self.model.recommend(
                    valid_users,
                    sub_X,
                    N=k,
                    filter_already_liked_items=exclude_seen,
                )
                for i, u in enumerate(valid_users):
                    results[u] = [int(x) for x in ids[i]]

            for u in batch_users:
                if u not in results:
                    results[u] = self.recommend(u, k=k, exclude_seen=exclude_seen)

        return results

    def recommend_all_with_scores(
        self,
        user_indices: Sequence[int],
        k: int = 10,
        exclude_seen: bool = True,
        batch_size: int = 2000,
    ) -> Dict[int, List[Tuple[int, float]]]:
        if not self.is_fitted or self.model is None or self.X_train is None:
            raise RuntimeError(f"{self.name} must be fitted before calling recommend_all_with_scores.")

        user_list = list(user_indices)
        results: Dict[int, List[Tuple[int, float]]] = {}

        for start_idx in range(0, len(user_list), batch_size):
            batch_users = user_list[start_idx : start_idx + batch_size]
            valid_users = [u for u in batch_users if u < self.num_users]

            if valid_users:
                sub_X = self.X_train[valid_users]
                ids, scores = self.model.recommend(
                    valid_users,
                    sub_X,
                    N=k,
                    filter_already_liked_items=exclude_seen,
                )
                for i, u in enumerate(valid_users):
                    results[u] = [
                        (int(item_id), float(score))
                        for item_id, score in zip(ids[i], scores[i])
                    ]

            for u in batch_users:
                if u not in results:
                    results[u] = self.recommend_with_scores(u, k=k, exclude_seen=exclude_seen)

        return results
