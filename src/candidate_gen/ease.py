"""EASE (Embarrassingly Shallow Autoencoders) candidate recommender."""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .base import BaseCandidateGenerator


class EASERecommender(BaseCandidateGenerator):
    """Embarrassingly Shallow Autoencoders for Sparse Data (Steck, WWW 2019).

    Learns a linear item-item weight matrix B with zero diagonal by solving
    a regularized closed-form regression on the binary implicit training matrix:
        min_B ||X - XB||_F^2 + lambda ||B||_F^2  s.t. diag(B) = 0
    """

    def __init__(self, reg: float = 500.0, name: str = "EASE") -> None:
        super().__init__(name=name)
        self.reg = float(reg)
        self.B: Optional[np.ndarray] = None
        self.X_train: Optional[sp.csr_matrix] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
    ) -> "EASERecommender":
        """Fit EASE closed-form model on positive training interactions."""
        train_pos = self._extract_positive_train_interactions(
            train_df, num_users=num_users, num_items=num_items
        )

        rows = train_pos["user_idx"].values.astype(np.int64)
        cols = train_pos["movie_idx"].values.astype(np.int64)
        data = np.ones(len(train_pos), dtype=np.float32)

        self.X_train = sp.csr_matrix(
            (data, (rows, cols)), shape=(self.num_users, self.num_items), dtype=np.float32
        )

        # Item-item Gram matrix G = X^T X
        G = (self.X_train.T @ self.X_train).toarray()
        # Add L2 regularization to diagonal
        G += self.reg * np.eye(self.num_items, dtype=np.float32)

        # Invert Gram matrix
        P = np.linalg.inv(G)
        diag_P = np.diag(P)

        # Closed form: B_ij = -P_ij / P_jj for i != j, B_ii = 0
        B = -P / diag_P[None, :]
        np.fill_diagonal(B, 0.0)

        self.B = B.astype(np.float32)
        self.is_fitted = True
        return self

    def _score_user(self, user_idx: int) -> np.ndarray:
        if not self.is_fitted or self.X_train is None or self.B is None:
            raise RuntimeError(f"{self.name} must be fitted before scoring.")

        if user_idx >= self.num_users:
            return np.zeros(self.num_items, dtype=np.float32)

        u_vec = self.X_train.getrow(user_idx)
        return (u_vec @ self.B).flatten()

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:
        scores = self._score_user(user_idx)
        if exclude_seen:
            seen = self.get_user_seen(user_idx)
            if seen:
                scores[list(seen)] = -np.inf

        ranked = np.argsort(-scores, kind="stable")
        return [int(x) for x in ranked[:k]]

    def recommend_with_scores(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        scores = self._score_user(user_idx)
        if exclude_seen:
            seen = self.get_user_seen(user_idx)
            if seen:
                scores[list(seen)] = -np.inf

        ranked = np.argsort(-scores, kind="stable")
        top_k = ranked[:k]
        return [(int(item_id), float(scores[item_id])) for item_id in top_k]

    def recommend_all(
        self,
        user_indices: Sequence[int],
        k: int = 10,
        exclude_seen: bool = True,
        batch_size: int = 1000,
    ) -> Dict[int, List[int]]:
        """Fast vectorized batch scoring for EASE."""
        if not self.is_fitted or self.X_train is None or self.B is None:
            raise RuntimeError(f"{self.name} must be fitted before calling recommend_all.")

        user_list = list(user_indices)
        results: Dict[int, List[int]] = {}

        for start_idx in range(0, len(user_list), batch_size):
            batch_users = user_list[start_idx : start_idx + batch_size]
            valid_users = [u for u in batch_users if u < self.num_users]

            if not valid_users:
                for u in batch_users:
                    results[u] = list(range(k))
                continue

            sub_X = self.X_train[valid_users, :]
            sub_scores = (sub_X @ self.B)

            for i, u in enumerate(valid_users):
                if exclude_seen:
                    seen = self.get_user_seen(u)
                    if seen:
                        sub_scores[i, list(seen)] = -np.inf

                ranked = np.argsort(-sub_scores[i], kind="stable")
                results[u] = [int(x) for x in ranked[:k]]

            for u in batch_users:
                if u not in results:
                    results[u] = self.recommend(u, k=k, exclude_seen=exclude_seen)

        return results

    def recommend_all_with_scores(
        self,
        user_indices: Sequence[int],
        k: int = 10,
        exclude_seen: bool = True,
        batch_size: int = 1000,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Fast vectorized batch scoring returning both item indices and model scores."""
        if not self.is_fitted or self.X_train is None or self.B is None:
            raise RuntimeError(f"{self.name} must be fitted before calling recommend_all_with_scores.")

        user_list = list(user_indices)
        results: Dict[int, List[Tuple[int, float]]] = {}

        for start_idx in range(0, len(user_list), batch_size):
            batch_users = user_list[start_idx : start_idx + batch_size]
            valid_users = [u for u in batch_users if u < self.num_users]

            if not valid_users:
                for u in batch_users:
                    results[u] = [(item, 0.0) for item in range(k)]
                continue

            sub_X = self.X_train[valid_users, :]
            sub_scores = (sub_X @ self.B)

            for i, u in enumerate(valid_users):
                if exclude_seen:
                    seen = self.get_user_seen(u)
                    if seen:
                        sub_scores[i, list(seen)] = -np.inf

                ranked = np.argsort(-sub_scores[i], kind="stable")
                top_k = ranked[:k]
                results[u] = [(int(item_id), float(sub_scores[i, item_id])) for item_id in top_k]

            for u in batch_users:
                if u not in results:
                    results[u] = self.recommend_with_scores(u, k=k, exclude_seen=exclude_seen)

        return results
