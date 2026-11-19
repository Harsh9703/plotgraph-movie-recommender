# PlotGraph: Explainable, Emotion-Aware Movie Recommendation

PlotGraph is a university research project combining collaborative filtering, knowledge graphs, and large language models for explainable and emotion-aware movie recommendation.

## What Phase 1 Implements

Phase 1 provides the foundation for the project using MovieLens-1M:
- **Reproducible Data Ingestion**: Clean parsers for MovieLens-1M `ratings.dat` and `movies.dat`, preserving original identifiers and binarizing ratings into positive interactions ($r \ge 4.0$).
- **Bidirectional ID Mapping**: Bijective, deterministic mapping between original MovieLens IDs and 0-indexed consecutive internal indices (`user_idx` $\in [0, N_u-1]$, `movie_idx` $\in [0, N_m-1]$).
- **Chronological Per-User Splitting**: Strict chronological 80/10/10 train/validation/test split per user based on interaction timestamps, eliminating temporal leakage with deterministic tie-breaking.
- **Leakage Verification**: Automated assertions confirming $\max(t_{\text{train}}) \le \min(t_{\text{val}}) \le \max(t_{\text{val}}) \le \min(t_{\text{test}})$ for every user.
- **Top-K Ranking Evaluation Suite**: Standalone, full-catalog ranking metrics (NDCG@K, Recall@K, Hit Rate@K, MRR@K) tested against analytical ground-truth test cases.
- **Centralized Configuration**: All paths, split ratios, seeds, thresholds, and cutoffs defined in `configs/data.yaml`.

---

## Environment Setup

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- A virtual environment tool (`venv`, `conda`, etc.)

### 2. Virtual Environment Creation & Installation

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install Phase 1 dependencies in editable mode
pip install -e ".[test]"
```

Core dependencies for Phase 1 are kept minimal:
- `pyyaml` (configuration)
- `pandas`, `numpy`, `scipy`, `pyarrow` (data manipulation, sparse matrices, and parquet serialization)
- `pytest` (test execution)

---

## Obtaining MovieLens-1M

You can download and extract MovieLens-1M using the provided script:

```bash
python scripts/download_ml1m.py
```

This downloads `https://files.grouplens.org/datasets/movielens/ml-1m.zip` and extracts `ratings.dat`, `movies.dat`, and `users.dat` into `data/raw/ml-1m/`. The raw data is ignored by Git.

---

## Running the Data Pipeline

To process MovieLens-1M, generate internal ID mappings, split chronologically (80/10/10), verify zero temporal leakage, and save Parquet splits:

```bash
python scripts/run_phase1_pipeline.py --config configs/data.yaml
```

Output artifacts are saved under `data/processed/ml-1m/`:
- `id_maps/user_id_map.parquet` (6,040 users)
- `id_maps/movie_id_map.parquet` (3,883 movies)
- `splits/train.parquet` (800,309 interactions, 80.0%)
- `splits/val.parquet` (99,950 interactions, 10.0%)
- `splits/test.parquet` (99,950 interactions, 10.0%)
- `movies.parquet` (3,883 catalog titles and genres)

---

## Running the Tests

Execute the automated test suite using `pytest`:

```bash
pytest -v
```

The test suite covers:
1. `tests/test_ranking_metrics.py`: Correctness of NDCG@K, Recall@K, HR@K, MRR@K against hand-calculated analytical rankings, empty sets, and edge cases.
2. `tests/test_id_mapper.py`: Consecutive 0-indexing, round-trip mapping, determinism, and parquet serialization.
3. `tests/test_splitter.py`: Chronological ordering, split proportions, zero temporal leakage verification, and small-history rounding.
4. `tests/test_loader.py`: Delimiter parsing, encoding (`latin-1`), and positive interaction labeling ($r \ge 4.0$).
5. `tests/test_ml1m_integration.py`: End-to-end integrity checks on real MovieLens-1M data (if downloaded).
6. `tests/test_popularity.py`: Popularity baseline positive training counts, exclusion of seen positive items, and deterministic tie-breaking.
7. `tests/test_item_knn.py`: ItemKNN sparse matrix cosine similarity, cluster-based personalization, ignoring non-positives, and deterministic top-K selection.
8. `tests/test_candidate_eval.py`: Evaluation harness validation relevance ($r \ge 4$), training history isolation, and empty user handling.

---

## Phase 2A: Candidate Recommendation Baselines

Phase 2A implements the first two candidate-generation recommendation models evaluated on the validation split:
- **`BaseCandidateGenerator`**: Reusable recommender interface enforcing strict positive-only training (`positive == 1`, $r \ge 4.0$).
- **`PopularityRecommender`**: Non-personalized baseline ranking unseen movies by positive interaction frequency in the training split.
- **`ItemKNNRecommender`**: Collaborative-filtering model utilizing sparse cosine item-item similarity with top-$K$ neighborhood pruning ($K=40$).
- **Validation Evaluation Harness**: Full-catalog ranking evaluation across all unseen catalog movies for each validation user against held-out positive items ($r \ge 4.0$), strictly isolating the TEST split.

### Running Phase 2A Baselines

```bash
python scripts/run_phase2a_baselines.py --config configs/candidates.yaml
```

### Benchmark Validation Results (Full-Catalog Ranking, K=10 & 20)

Evaluated across **5,805 validation users** who have at least one positive interaction in the validation split (out of 6,040 total users, 96.11% coverage):

| Model | NDCG@10 | Recall@10 | HR@10 | MRR@10 | NDCG@20 | Recall@20 | HR@20 | MRR@20 | Train Time | Eval Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Popularity** | 0.0490 | 0.0432 | 0.2305 | 0.0954 | 0.0617 | 0.0857 | 0.3611 | 0.1044 | ~0.13s | ~0.19s |
| **ItemKNN** ($K=40$) | **0.0679** | **0.0772** | **0.3077** | **0.1161** | **0.0883** | **0.1383** | **0.4711** | **0.1275** | ~2.04s | ~1.55s |

**ItemKNN relative lift over Popularity**:
- **NDCG@10**: **+38.72%**
- **Recall@10**: **+78.98%**
- **HR@10**: **+33.49%**
- **MRR@10**: **+21.70%**

