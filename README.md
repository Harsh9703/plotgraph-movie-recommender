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
9. `tests/test_als.py`: Implicit ALS fitting, positive-only interactions, and exclusion of seen training items.
10. `tests/test_bpr.py`: BPR matrix factorization fitting, scoring, and deterministic recommendations.
11. `tests/test_ease.py`: EASE closed-form Gram matrix inversion, zero diagonal assertion, and personalization.
12. `tests/test_phase2b_selection.py`: Validation grid search, model selection by validation NDCG@10, Top-100 candidate pool export, and Recall@100 evaluation.
13. `tests/test_tmdb_metadata.py`: Safe parsing (`ast.literal_eval`), malformed non-numeric ID rejection, deterministic deduplication, director and top-5 actor extraction, release decade derivation, zero catalog dropping, and end-to-end integration.
14. `tests/test_kg_m.py`: Deterministic entity/movie URI canonicalization, Unicode handling, synthetic graph construction, missing metadata handling, Turtle round-trip preservation, SPARQL query engine helpers, and real KG-M graph validation.

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

---

## Phase 2B: Classical Model Tuning & Top-100 Candidate Selection

Phase 2B completes the classical candidate-generation tier:
- **`ALSRecommender`**: Matrix Factorization using Implicit Alternating Least Squares (Hu et al., 2008).
- **`BPRRecommender`**: Bayesian Personalized Ranking for implicit feedback (Rendle et al., 2009).
- **`EASERecommender`**: Embarrassingly Shallow Autoencoders (Steck, 2019) with closed-form L2-regularized weight matrix inversion.
- **Validation Hyperparameter Grid Search**: Small grid search across models using strictly validation NDCG@10 (TEST split remains untouched).
- **Five-Model Fair Benchmark**: Comparison across all 5 classical models under the exact same chronological validation protocol.
- **Top-100 Candidate Pool Generation**: Full candidate pool ($6,040 \times 100 = 604,000$ rows) exported to `data/processed/ml-1m/candidates/top100_candidates.parquet` for later PlotGraph reranking stages.
- **Recall@100 Coverage Evaluation**: Evaluating the percentage of validation positive items captured within the Top-100 candidate pool.

### Running Phase 2B Pipeline

```bash
python scripts/run_phase2b_models.py --config configs/candidates.yaml
```

### Five-Model Validation Benchmark (Full-Catalog Ranking, 5,805 Evaluated Users)

| Model | Selected Configuration | NDCG@10 | Recall@10 | HR@10 | MRR@10 | NDCG@20 | Recall@20 | HR@20 | MRR@20 | Train Time | Eval Time |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Popularity** | non-personalized | 0.0490 | 0.0432 | 0.2305 | 0.0954 | 0.0617 | 0.0857 | 0.3611 | 0.1044 | 0.135s | 0.234s |
| **BPR** | factors=32, reg=0.01, iters=50 | 0.0592 | 0.0704 | 0.2734 | 0.1009 | 0.0761 | 0.1215 | 0.4169 | 0.1107 | 4.625s | 0.396s |
| **ItemKNN** | k_neighbors=40 | 0.0679 | 0.0772 | 0.3077 | 0.1161 | 0.0883 | 0.1383 | 0.4711 | 0.1275 | 2.011s | 1.556s |
| **ALS** | factors=32, reg=0.01, iters=15 | 0.0722 | 0.0844 | 0.3276 | 0.1216 | 0.0916 | 0.1433 | 0.4844 | 0.1323 | 1.396s | 0.508s |
| **EASE** | reg=500.0 | **0.0776** | **0.0909** | **0.3414** | **0.1295** | **0.0984** | **0.1540** | **0.5104** | **0.1410** | 3.491s | 2.554s |

### Selected Candidate Generator & Top-100 Coverage
- **Selected Model**: **EASE** ($\lambda = 500.0$), achieving the highest validation NDCG@10 (**0.0776**).
- **Candidate Pool**: 604,000 candidate items exported across all 6,040 MovieLens-1M users (100 candidates per user).
- **Validation Recall@100**: **0.4421** (44.21% of held-out future validation positives are present in the candidate pool).
- **Data Leakage Check**: 0 seen positive training items leaked into the candidate pool.

---

## Phase 3A: TMDB Metadata Ingestion, Cleaning & MovieLens Linking

Phase 3A ingests and defensively cleans metadata from [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (placed at `data/raw/the-movies-dataset/`), safely linking it to MovieLens-1M movies:
- **Defensive Cleaning**: Drops corrupted rows with shifted dates in ID columns (e.g. rows with non-numeric IDs), resolves duplicate TMDB entries deterministically by vote count and overview completeness.
- **Safe Evaluation**: Strictly avoids unsafe `eval()`; uses `ast.literal_eval` with robust exception handling across all nested JSON/literal fields.
- **Explicit ID Linking**: Uses `links.csv` (`movieId` $\leftrightarrow$ `tmdbId`) exclusively. Zero ambiguous title fuzzy matching.
- **Catalog Preservation**: Never drops MovieLens movies that lack TMDB metadata. All 3,883 movies are retained in the unified catalog with explicit boolean flags (`has_tmdb_link`, `has_overview`, `has_credits`, `has_keywords`) and nullable fields, preserving exact `movie_idx` and `movie_id` mappings.
- **Entity & Feature Extraction**:
  - Primary director extracted from `crew`.
  - Top-5 billed actors extracted from `cast` sorted by billing order.
  - Clean keyword string lists extracted from `keywords`.
  - Franchise/collection names extracted from `belongs_to_collection`.
  - Cinema release decade derived (e.g., `1995` $\to$ `1990s`).

### Running Phase 3A Pipeline

```bash
python scripts/run_phase3a_metadata.py --config configs/metadata.yaml
```

Output artifacts:
- `data/processed/ml-1m/metadata/movies_with_tmdb.parquet` (3,883 rows, 21 columns)
- `data/processed/ml-1m/metadata/coverage_report.json`

### Metadata Coverage Across Catalog, Interactions & Candidates

| Metric | All Catalog Movies (N=3,883) | Rated Movies (N=3,706) | Top-100 Candidates (N=1,850) |
|:---|:---:|:---:|:---:|
| **Linked to TMDB (`links.csv`)** | 3,828 (98.58%) | 3,667 (98.95%) | **1,843 (99.62%)** |
| **Matching TMDB Metadata** | 3,828 (98.58%) | 3,667 (98.95%) | **1,843 (99.62%)** |
| **Non-Empty Plot Overview** | 3,810 (98.12%) | 3,657 (98.68%) | **1,842 (99.57%)** |
| **Credits Available (cast/crew)** | 3,812 (98.17%) | 3,660 (98.76%) | **1,843 (99.62%)** |
| **Director Present** | 3,803 (97.94%) | 3,653 (98.57%) | **1,843 (99.62%)** |
| **Top Actors Present ($\ge 1$)** | 3,768 (97.04%) | 3,626 (97.84%) | **1,838 (99.35%)** |
| **Keywords Present ($\ge 1$)** | 3,342 (86.07%) | 3,262 (88.02%) | **1,796 (97.08%)** |
| **Belongs to Collection** | 669 (17.23%) | 665 (17.94%) | 444 (24.00%) |

---

## Phase 3B: Metadata Knowledge Graph (KG-M) & SPARQL Query Engine

Phase 3B constructs the metadata Knowledge Graph (**KG-M**) using RDFlib from the verified Phase 3A unified metadata table (`movies_with_tmdb.parquet`), serializes it deterministically as RDF/Turtle (`kg_m.ttl`), validates graph integrity and ID recoverability, and provides a reusable SPARQL query engine.

### Knowledge Graph Architecture

- **Namespaces**:
  - Ontology Namespace: `@prefix pgo: <https://plotgraph.org/ontology/> .`
  - Resource Namespace: `@prefix pgr: <https://plotgraph.org/resource/> .`
- **Core Entity Classes**:
  - `pgo:Movie`: 3,883 movies preserving 100% of the MovieLens catalog.
  - `pgo:Person`: 10,417 distinct directors and top-5 billed actors.
  - `pgo:Genre`: 20 standardized TMDB genres.
  - `pgo:Keyword`: 7,815 distinct plot keywords.
  - `pgo:Collection`: 364 movie franchise/collection entities.
  - `pgo:ReleaseDecade`: 11 cinema release decades.
- **Relational Predicates**:
  - `pgo:hasGenre`: Connects Movie $\to$ Genre.
  - `pgo:directedBy`: Connects Movie $\to$ Person (director).
  - `pgo:starsActor`: Connects Movie $\to$ Person (top-5 billed actors).
  - `pgo:hasKeyword`: Connects Movie $\to$ Keyword.
  - `pgo:inCollection`: Connects Movie $\to$ Collection.
  - `pgo:releaseDecade`: Connects Movie $\to$ ReleaseDecade.
- **Literal Properties**:
  - `pgo:title`: Clean title string.
  - `pgo:movieLensId`: Original MovieLens ID (`xsd:integer`).
  - `pgo:internalMovieIndex`: Consecutive internal index (`xsd:integer`).
  - `pgo:tmdbId`: TMDB ID (`xsd:integer`, where available).
  - `pgo:tmdbTitle`: Clean TMDB title string.
  - `pgo:releaseDate`: Release date string (`YYYY-MM-DD`).
  - `rdfs:label`: Human-readable entity label for every resource.

### Deterministic URI Canonicalization Strategy

- **Movie URIs**:
  - Unique TMDB ID $\to$ `pgr:movie_tmdb_{tmdb_id}` (e.g. `pgr:movie_tmdb_862` for *Toy Story*).
  - Rare duplicate TMDB ID $\to$ `pgr:movie_tmdb_{tmdb_id}_ml_{movie_id}` (disambiguates distinct movies sharing TMDB IDs in Kaggle source data, avoiding silent entity merging).
  - Unmapped MovieLens movies (55 movies) $\to$ `pgr:movie_ml_{movie_id}` (catalog identity preserved with zero hallucinated metadata edges).
- **Metadata Entity URIs**:
  - Normalized with Unicode NFKC, lowercased, and converted into URL-safe slugs with alphanumeric characters and underscores (`pgr:{entity_type}_{slug}`).
  - Each entity receives exactly one canonical `rdfs:label` and `rdf:type` definition.

### Running Phase 3B Pipeline

```bash
python scripts/run_phase3b_kg_m.py --config configs/kg.yaml
```

Output artifacts (Git-ignored under `data/processed/`):
- `data/processed/ml-1m/graphs/kg_m.ttl` (4.43 MB, 129,658 triples)
- `data/processed/ml-1m/graphs/kg_m_validation.json`

### KG-M Graph Statistics & Validation

| Entity / Triple Metric | Count |
|:---|:---:|
| **Total Graph Triples** | **129,658** |
| `pgo:Movie` Entities | 3,883 |
| `pgo:Person` Entities (Directors & Actors) | 10,417 |
| `pgo:Genre` Entities | 20 |
| `pgo:Keyword` Entities | 7,815 |
| `pgo:Collection` Entities | 364 |
| `pgo:ReleaseDecade` Entities | 11 |
| **Total Named Entities** | **22,510** |
| `pgo:hasKeyword` Triples | 25,591 |
| `rdfs:label` Triples | 22,510 |
| `rdf:type` Triples | 22,510 |
| `pgo:starsActor` Triples | 18,502 |
| `pgo:hasGenre` Triples | 9,120 |
| `pgo:title` / `movieLensId` / `internalMovieIndex` Triples | 3,883 each |
| `pgo:tmdbId` / `tmdbTitle` Triples | 3,828 each |
| `pgo:releaseDecade` / `releaseDate` Triples | 3,824 each |
| `pgo:directedBy` Triples | 3,803 |
| `pgo:inCollection` Triples | 669 |
| **All MovieLens IDs Recoverable** | **Yes (3,883 / 3,883, 100%)** |
| **All Internal Indices Recoverable** | **Yes (3,883 / 3,883, 100%)** |
| **Zero Bogus / NaN Placeholder Resources** | **Confirmed (0 bogus URIs, 0 bogus labels)** |
| **Turtle Syntax Round-Trip Preservation** | **Confirmed (129,658 / 129,658 triples)** |

### SPARQL Query Utilities (`KGMQueryEngine`)

The reusable `KGMQueryEngine` (`src/knowledge_graph/query_engine.py`) provides helper methods for downstream reranking features and explanation paths:

```python
from src.knowledge_graph import KGMQueryEngine

engine = KGMQueryEngine("data/processed/ml-1m/graphs/kg_m.ttl")

# 1. Full movie metadata retrieval
meta = engine.get_movie_metadata(movie_id=1)

# 2. Shared director, actors, collection, genres, or keywords
same_dir = engine.find_movies_sharing_director(movie_id=1)
same_actors = engine.find_movies_sharing_actor(movie_id=1)
same_coll = engine.find_movies_in_same_collection(movie_id=1)
same_genres = engine.find_movies_sharing_genre(movie_id=1)
same_kws = engine.find_movies_sharing_keyword(movie_id=1)

# 3. Multi-hop 2-hop path query connecting two movies
paths = engine.find_paths_between_movies(movie_a=1, movie_b=2355)
# Returns:
# - 'Toy Story (1995)' --(directedBy)--> [Person: John Lasseter] <--(directedBy)-- 'A Bug's Life (1998)'
# - 'Toy Story (1995)' --(hasGenre)--> [Genre: Animation] <--(hasGenre)-- 'A Bug's Life (1998)'
# - 'Toy Story (1995)' --(hasGenre)--> [Genre: Comedy] <--(hasGenre)-- 'A Bug's Life (1998)'
# - 'Toy Story (1995)' --(hasGenre)--> [Genre: Family] <--(hasGenre)-- 'A Bug's Life (1998)'
# - 'Toy Story (1995)' --(releaseDecade)--> [ReleaseDecade: 1990s] <--(releaseDecade)-- 'A Bug's Life (1998)'
```




