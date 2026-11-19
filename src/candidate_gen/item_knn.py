"""ItemKNN candidate recommendation model with sparse cosine similarity."""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .base import BaseCandidateGenerator


class ItemKNNRecommender(BaseCandidateGenerator):
    """Item-based collaborative filtering recommender with cosine similarity.

    Constructed strictly from the positive implicit feedback training matrix.
    Uses sparse representations, top-K neighborhood pruning, and deterministic tie-breaking.
    """

    def __init__(
        self,
        name: str = "ItemKNN",
        k_neighbors: int = 40,
        shrinkage: float = 0.0,
    ) -> None:
        super().__init__(name=name)
        self.k_neighbors = int(k_neighbors)
        self.shrinkage = float(shrinkage)
        self.W_sparse: Optional[sp.csr_matrix] = None
        self.X_train: Optional[sp.csr_matrix] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
    ) -> "ItemKNNRecommender":
        """Fit ItemKNN on positive training interactions."""
        train_pos = self._extract_positive_train_interactions(
            train_df, num_users=num_users, num_items=num_items
        )

        rows = train_pos["user_idx"].values.astype(np.int64)
        cols = train_pos["movie_idx"].values.astype(np.int64)
        data = np.ones(len(train_pos), dtype=np.float32)

        # Binary user-item matrix X (shape: num_users x num_items)
        self.X_train = sp.csr_matrix(
            (data, (rows, cols)), shape=(self.num_users, self.num_items)
        )

        # Item co-occurrence matrix C = X^T X
        Xt = self.X_train.tocsc()
        C = (Xt.T @ Xt).tocsr()

        # Item L2 norms: sqrt(sum of squares), which equals sqrt(positive count)
        diag = C.diagonal()
        norms = np.sqrt(diag)
        inv_norms = np.zeros_like(norms, dtype=np.float32)
        nz_mask = norms > 0
        inv_norms[nz_mask] = 1.0 / (norms[nz_mask] + self.shrinkage)

        D_inv = sp.diags(inv_norms)
        S = (D_inv @ C @ D_inv).tocsr()

        # Zero out diagonal (an item should not recommend itself)
        S.setdiag(0.0)
        S.eliminate_zeros()

        # Prune each item's neighborhood to top k_neighbors
        S_lil = S.tolil()
        pruned_rows: List[int] = []
        pruned_cols: List[int] = []
        pruned_data: List[float] = []

        for i in range(self.num_items):
            row_data = S_lil.data[i]
            row_cols = S_lil.rows[i]
            n_entries = len(row_data)

            if n_entries > self.k_neighbors:
                arr_data = np.array(row_data, dtype=np.float32)
                top_indices = np.argpartition(-arr_data, self.k_neighbors)[: self.k_neighbors]
                for idx in top_indices:
                    pruned_rows.append(i)
                    pruned_cols.append(row_cols[idx])
                    pruned_data.append(float(row_data[idx]))
            elif n_entries > 0:
                for idx in range(n_entries):
                    pruned_rows.append(i)
                    pruned_cols.append(row_cols[idx])
                    pruned_data.append(float(row_data[idx]))

        # W has top k_neighbors for each item row
        # Candidate item j score given user positive history H_u+: sum_{i in H_u+} W[j, i]
        # In matrix notation: R = X @ W^T
        self.W_sparse = sp.csr_matrix(
            (pruned_data, (pruned_rows, pruned_cols)),
            shape=(self.num_items, self.num_items),
            dtype=np.float32,
        )

        self.is_fitted = True
        return self

    def _score_user(self, user_idx: int) -> np.ndarray:
        """Compute unmasked item scores for a single user."""
        if not self.is_fitted or self.X_train is None or self.W_sparse is None:
            raise RuntimeError(f"{self.name} must be fitted before scoring.")

        if user_idx >= self.num_users:
            return np.zeros(self.num_items, dtype=np.float32)

        user_vec = self.X_train.getrow(user_idx)
        # user_vec is (1, num_items), W_sparse.T is (num_items, num_items)
        scores = (user_vec @ self.W_sparse.T).toarray().flatten()
        return scores

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:
        """Generate top-K recommendations for a single user with deterministic tie-breaking."""
        scores = self._score_user(user_idx)

        if exclude_seen:
            seen = self.get_user_seen(user_idx)
            if seen:
                scores[list(seen)] = -np.inf

        # Stable sort: scores descending, ties broken by movie_idx ascending
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
        """Optimized batched candidate generation for full-catalog ranking."""
        if not self.is_fitted or self.X_train is None or self.W_sparse is None:
            raise RuntimeError(f"{self.name} must be fitted before calling recommend_all.")

        user_list = list(user_indices)
        results: Dict[int, List[int]] = {}

        # Process in user batches to balance memory and vectorized speed
        for start_idx in range(0, len(user_list), batch_size):
            batch_users = user_list[start_idx : start_idx + batch_size]
            batch_rows = [u for u in batch_users if u < self.num_users]

            if not batch_rows:
                for u in batch_users:
                    results[u] = list(range(k))
                continue

            sub_X = self.X_train[batch_rows, :]
            # Sub_scores shape: (len(batch_rows), num_items)
            sub_scores = (sub_X @ self.W_sparse.T).toarray()

            for i, u in enumerate(batch_rows):
                if exclude_seen:
                    seen = self.get_user_seen(u)
                    if seen:
                        sub_scores[i, list(seen)] = -np.inf

                ranked = np.argsort(-sub_scores[i], kind="stable")
                results[u] = [int(x) for x in ranked[:k]]

            # Fallback for users outside known universe
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
        if not self.is_fitted or self.X_train is None or self.W_sparse is None:
            raise RuntimeError(f"{self.name} must be fitted before calling recommend_all_with_scores.")

        user_list = list(user_indices)
        results: Dict[int, List[Tuple[int, float]]] = {}

        for start_idx in range(0, len(user_list), batch_size):
            batch_users = user_list[start_idx : start_idx + batch_size]
            batch_rows = [u for u in batch_users if u < self.num_users]

            if not batch_rows:
                for u in batch_users:
                    results[u] = [(item, 0.0) for item in range(k)]
                continue

            sub_X = self.X_train[batch_rows, :]
            sub_scores = (sub_X @ self.W_sparse.T).toarray()

            for i, u in enumerate(batch_rows):
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
