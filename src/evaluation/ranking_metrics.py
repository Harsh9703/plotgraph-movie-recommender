"""Full-catalog Top-K ranking evaluation metrics."""

import math
from typing import Dict, Iterable, List, Sequence, Set, Union


def hit_rate_at_k(recommended: Sequence[int], relevant: Union[Set[int], Sequence[int]], k: int) -> float:
    """Compute Hit Rate @ K (HR@K).

    Returns 1.0 if at least one recommended item in the top K is relevant, else 0.0.
    """
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    rel_set = set(relevant)
    return 1.0 if any(item in rel_set for item in rec_k) else 0.0


def recall_at_k(recommended: Sequence[int], relevant: Union[Set[int], Sequence[int]], k: int) -> float:
    """Compute Recall @ K.

    Proportion of relevant items retrieved in top K:
        Recall@K = |Top_K ∩ Relevant| / |Relevant|

    Returns 0.0 if relevant is empty.
    """
    if k <= 0 or not relevant:
        return 0.0
    rec_k = set(recommended[:k])
    rel_set = set(relevant)
    hits = len(rec_k & rel_set)
    return float(hits) / float(len(rel_set))


def mrr_at_k(recommended: Sequence[int], relevant: Union[Set[int], Sequence[int]], k: int) -> float:
    """Compute Mean Reciprocal Rank @ K (MRR@K / RR@K for a single user).

    Reciprocal rank of the first relevant item in the top-K list (1-indexed).
    If no relevant item is found in top K, returns 0.0.
    """
    if k <= 0 or not relevant:
        return 0.0
    rel_set = set(relevant)
    for rank, item in enumerate(recommended[:k], start=1):
        if item in rel_set:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(recommended: Sequence[int], relevant: Union[Set[int], Sequence[int]], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain @ K (NDCG@K) with binary relevance.

        DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
        IDCG@K = sum_{i=1}^{min(K, |Relevant|)} (1 / log2(i + 1))
        NDCG@K = DCG@K / IDCG@K

    Returns 0.0 if relevant is empty.
    """
    if k <= 0 or not relevant:
        return 0.0

    rel_set = set(relevant)
    rec_k = recommended[:k]

    dcg = 0.0
    for rank, item in enumerate(rec_k, start=1):
        if item in rel_set:
            dcg += 1.0 / math.log2(rank + 1)

    if dcg == 0.0:
        return 0.0

    num_ideal = min(k, len(rel_set))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, num_ideal + 1))

    return dcg / idcg if idcg > 0.0 else 0.0


def evaluate_user_ranking(
    recommended: Sequence[int],
    relevant: Union[Set[int], Sequence[int]],
    cutoffs: Sequence[int] = (10, 20),
) -> Dict[str, float]:
    """Compute all ranking metrics (NDCG, Recall, HR, MRR) for a single user across multiple cutoffs."""
    results = {}
    rel_set = set(relevant)
    for k in cutoffs:
        results[f"ndcg@{k}"] = ndcg_at_k(recommended, rel_set, k)
        results[f"recall@{k}"] = recall_at_k(recommended, rel_set, k)
        results[f"hr@{k}"] = hit_rate_at_k(recommended, rel_set, k)
        results[f"mrr@{k}"] = mrr_at_k(recommended, rel_set, k)
    return results


def evaluate_all_users(
    recommendations: Dict[int, Sequence[int]],
    ground_truth: Dict[int, Union[Set[int], Sequence[int]]],
    cutoffs: Sequence[int] = (10, 20),
    exclude_empty_users: bool = True,
) -> Dict[str, float]:
    """Evaluate recommendations across all users and return average metrics.

    Args:
        recommendations: Dict mapping user_id to ordered list of recommended item_ids.
        ground_truth: Dict mapping user_id to set/list of positive held-out item_ids.
        cutoffs: List of cutoffs K to evaluate.
        exclude_empty_users: If True, only averages over users with |Relevant| > 0.

    Returns:
        Dict of average metric names to float scores.
    """
    accumulators: Dict[str, List[float]] = {}
    all_metric_keys = [
        f"{m}@{k}"
        for k in cutoffs
        for m in ("ndcg", "recall", "hr", "mrr")
    ]
    for key in all_metric_keys:
        accumulators[key] = []

    eval_users = list(recommendations.keys())

    for u in eval_users:
        rel = ground_truth.get(u, set())
        if exclude_empty_users and len(rel) == 0:
            continue

        user_metrics = evaluate_user_ranking(recommendations[u], rel, cutoffs=cutoffs)
        for key, val in user_metrics.items():
            accumulators[key].append(val)

    # Average
    averaged = {}
    for key, values in accumulators.items():
        averaged[key] = float(sum(values) / len(values)) if values else 0.0

    return averaged
