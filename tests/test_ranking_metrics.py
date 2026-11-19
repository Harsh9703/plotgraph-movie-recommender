"""Unit tests for full-catalog ranking metrics."""

import math
import pytest
from src.evaluation.ranking_metrics import (
    hit_rate_at_k,
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    evaluate_user_ranking,
    evaluate_all_users,
)


def test_perfect_ranking():
    rec = [1, 2, 3, 4, 5]
    rel = {1, 2, 3, 4, 5}
    for k in [1, 3, 5]:
        assert hit_rate_at_k(rec, rel, k) == 1.0
        assert recall_at_k(rec, rel, k) == pytest.approx(k / 5.0)
        assert mrr_at_k(rec, rel, k) == 1.0
        assert ndcg_at_k(rec, rel, k) == pytest.approx(1.0)


def test_zero_hits():
    rec = [1, 2, 3, 4, 5]
    rel = {10, 20}
    for k in [1, 3, 5, 10]:
        assert hit_rate_at_k(rec, rel, k) == 0.0
        assert recall_at_k(rec, rel, k) == 0.0
        assert mrr_at_k(rec, rel, k) == 0.0
        assert ndcg_at_k(rec, rel, k) == 0.0


def test_hand_calculated_ndcg_and_mrr_case1():
    # Relevant item at rank 2 only
    rec = [99, 10, 88, 77]
    rel = {10}

    # At K=1: nothing
    assert hit_rate_at_k(rec, rel, k=1) == 0.0
    assert recall_at_k(rec, rel, k=1) == 0.0
    assert mrr_at_k(rec, rel, k=1) == 0.0
    assert ndcg_at_k(rec, rel, k=1) == 0.0

    # At K=2: item 10 is at rank 2
    # DCG = 1 / log2(3) = ~0.63092975
    # IDCG = 1 / log2(2) = 1.0
    expected_ndcg = 1.0 / math.log2(3)
    assert hit_rate_at_k(rec, rel, k=2) == 1.0
    assert recall_at_k(rec, rel, k=2) == 1.0
    assert mrr_at_k(rec, rel, k=2) == 0.5
    assert ndcg_at_k(rec, rel, k=2) == pytest.approx(expected_ndcg)


def test_hand_calculated_ndcg_case2():
    # 4 relevant items in catalog, 2 retrieved at rank 1 and rank 3
    rec = [10, 99, 20, 88, 30]
    rel = {10, 20, 30, 40}

    # Evaluate at K=3 (retrieved: 10 at rank 1, 20 at rank 3)
    # DCG = 1 / log2(2) + 0 + 1 / log2(4) = 1.0 + 0.5 = 1.5
    # IDCG = 1 / log2(2) + 1 / log2(3) + 1 / log2(4) = 1.0 + 1/log2(3) + 0.5
    expected_dcg = 1.0 + 1.0 / math.log2(4)
    expected_idcg = 1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    expected_ndcg = expected_dcg / expected_idcg

    assert hit_rate_at_k(rec, rel, k=3) == 1.0
    assert recall_at_k(rec, rel, k=3) == 2.0 / 4.0
    assert mrr_at_k(rec, rel, k=3) == 1.0
    assert ndcg_at_k(rec, rel, k=3) == pytest.approx(expected_ndcg)


def test_empty_relevant_set():
    rec = [1, 2, 3]
    rel = set()
    assert hit_rate_at_k(rec, rel, 5) == 0.0
    assert recall_at_k(rec, rel, 5) == 0.0
    assert mrr_at_k(rec, rel, 5) == 0.0
    assert ndcg_at_k(rec, rel, 5) == 0.0


def test_k_cutoff_truncation():
    rec = [1, 2, 3, 4, 5, 50]
    rel = {50}
    # Before rank 6, item 50 should not count
    assert hit_rate_at_k(rec, rel, k=5) == 0.0
    assert recall_at_k(rec, rel, k=5) == 0.0
    assert mrr_at_k(rec, rel, k=5) == 0.0
    assert ndcg_at_k(rec, rel, k=5) == 0.0

    # At K=6, it counts
    assert hit_rate_at_k(rec, rel, k=6) == 1.0
    assert recall_at_k(rec, rel, k=6) == 1.0
    assert mrr_at_k(rec, rel, k=6) == 1.0 / 6.0
    assert ndcg_at_k(rec, rel, k=6) == pytest.approx((1.0 / math.log2(7)) / 1.0)


def test_evaluate_all_users():
    recs = {
        1: [10, 20, 30],
        2: [99, 88, 77],
    }
    ground_truth = {
        1: {10}, # User 1: hit at rank 1, NDCG=1, Recall=1, HR=1, MRR=1
        2: {77}, # User 2: hit at rank 3, NDCG=0.5/1.0=0.5, Recall=1, HR=1, MRR=1/3
    }
    metrics = evaluate_all_users(recs, ground_truth, cutoffs=[3])

    assert metrics["hr@3"] == 1.0
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@3"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert metrics["ndcg@3"] == pytest.approx((1.0 + 0.5) / 2.0)
