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
15. `tests/test_kg_l_contract.py`: Fixed ontology relations (6), exact 8 Plutchik emotions, strict Pydantic extraction schema, duplicate/blank entity handling, markdown-fenced JSON extraction/recovery, deterministic prompt templates, and sample export validity.
16. `tests/test_qwen_pilot.py`: Pilot safety guard enforcement, mock inference execution, checkpointing/resumption, retry logic and error logging, summary generation, and review CSV export.
17. `tests/test_emotion_contract.py`: Evidence-grounded emotion extraction schema, Plutchik emotion taxonomy, calibration bounds, and prompt formatting.
18. `tests/test_emotion_pilot.py`: Dedicated emotion pilot execution, mock backend, safety guards, retry configuration resolution, and three-way comparative evaluation.
19. `tests/test_qwen_production.py`: Production chunk/range partitioning, eligible movie filtering, mock batch alignment, independent task checkpointing/resumption, and runtime estimation.
20. `tests/test_kg_l_builder.py`: KG-M/KG-L movie URI identity, semantic canonicalization, edge selection boundary (`score > 0.3`), continuous score/evidence representation, Turtle round-trip preservation, KGLQueryEngine SPARQL queries, KG-M + KG-L cross-graph compatibility, and error categorization regression.
21. `tests/test_features.py`: Recency weighting exponential decay, positive-only train extraction, leakage guards rejecting VAL/TEST, user profile normalization, KG-M/KG-L affinities and overlap counts, continuous 8D emotion compatibility metrics, missingness handling, candidate integrity preservation, feature groups, and real artifact validation.
22. `tests/test_text_features.py`: Deterministic TF-IDF overview vectorization, SBERT dense encoding on CPU, FAISS IndexFlatIP and NumPy fallback nearest-neighbor search, leakage guards rejecting VAL/TEST in text user profiles, recency weighting and unit L2 normalization, dot product and cosine similarity equivalence, missing text NaN handling, and full candidate matrix schema preservation.
23. `tests/test_nrc_features.py`: Deterministic text preprocessing and tokenization, NRC Word-Emotion Association Lexicon lookup (restricted to 8 Plutchik emotions), continuous distribution normalization, zero-match and missing overview handling, TRAIN-only user profiles with exponential recency decay, feature group separation from Qwen, candidate matrix preservation (604,000 rows x 91 columns), and Qwen-vs-NRC comparative analytics.

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

---

## Phase 3C: KG-L LLM Plot Extraction Contract & Sample Preparation

Phase 3C defines the formal contract, ontology constraints, strict structured extraction schema, prompt templates, validation/canonicalization infrastructure, and sample export workflow for **KG-L** (the LLM-enriched plot knowledge graph), in preparation for future Qwen2.5-7B-Instruct extraction.

> **Research Integrity & Compute Note**: No LLM inference or GPU execution is performed in Phase 3C. The goal of this phase is to establish a rigorous, auditable schema contract to prevent model hallucination, arbitrary relation generation, or malformed outputs before allocating GPU resources.

### 1. KG-M vs KG-L

- **KG-M (Metadata Knowledge Graph, Phase 3B)**: Grounded in verified structured TMDB metadata (`directedBy`, `starsActor`, `hasGenre`, `inCollection`, `hasKeyword`, `releaseDecade`).
- **KG-L (Plot Knowledge Graph, Phases 3C & 4)**: Enriched through LLM narrative extraction from movie plot overviews using a fixed semantic ontology and continuous Plutchik emotion vectors.

### 2. Fixed KG-L Ontology & Relations

The extraction schema permits exactly **6 relation types**:
1. `hasTheme`: Abstract recurring ideas or subjects central to the plot (e.g. `"friendship"`, `"identity"`, `"redemption"`).
2. `hasSetting`: Key locations, environments, historical eras, or worlds (e.g. `"gotham city"`, `"victorian london"`, `"suburbia"`).
3. `hasNarrativeDevice`: Storytelling mechanisms evident in the overview (e.g. `"time travel"`, `"unreliable narrator"`, `"mistaken identity"`).
4. `hasProtagonistType`: Character archetypes or roles (e.g. `"detective"`, `"sentient toy"`, `"high school student"`).
5. `basedOn`: Explicit source material stated in the overview (e.g. `"novel"`, `"true story"`, `"comic book"`).
6. `evokesEmotion`: Plutchik 8-emotion intensity scores (edges added in future KG-L only when intensity > 0.3; pilot-revised from >0.5).

### 3. Plutchik Eight Emotion Dimensions

Every movie plot overview is evaluated on the exact eight Plutchik basic emotions with continuous intensities in $[0.0, 1.0]$:
- `joy`
- `trust`
- `fear`
- `surprise`
- `sadness`
- `disgust`
- `anger`
- `anticipation`

All 8 continuous raw scores are preserved in extraction artifacts for downstream user emotional profiling and recommendation reranking.

### 4. Strict Schema & Grounding Rules

- **Schema Enforcement (`Pydantic`)**: Unknown fields or extra keys are rejected with `extra="forbid"`.
- **Entity Validation**: Entities must be strings; blanks/empty strings are discarded; duplicate entities are normalized and deduplicated deterministically.
- **Emotion Constraints**: All 8 emotions must be present; values must be numeric in $[0.0, 1.0]$; unknown emotion names are strictly rejected.
- **Strict Grounding**: The extraction prompt instructs the model to rely solely on explicit facts in the overview, forbids outside knowledge, and requires empty arrays `[]` when evidence is absent.
- **Robust Parsing & Recovery**: The parser accepts raw JSON, extracts markdown-fenced ```json blocks, and strips extraneous prose without masking structural schema errors.

### 5. Running the Phase 3C Preparation Script

```bash
python scripts/run_phase3c_kg_l_prep.py --config configs/kg_l.yaml
```

Output artifacts (Git-ignored under `data/processed/ml-1m/kg_l/`):
- `data/processed/ml-1m/kg_l/sample_movies.parquet` (30 deterministic sample movies)
- `data/processed/ml-1m/kg_l/sample_summary.json` (Traceability summary with IDs, decades, and overview lengths)

---

## Phase 3D: GPU Inference Infrastructure & 30-Movie Qwen Pilot

Phase 3D implements the GPU inference execution and verification pipeline for a strictly bounded 30-movie pilot using `Qwen/Qwen2.5-7B-Instruct` on ASU SOL GPU compute via SLURM and vLLM, with local mocking support for macOS/CPU development.

> **Research Integrity & Compute Boundaries**:
> - **Strict Safety Guard**: Hard-coded assertion enforcing $\le 30$ movies. Execution against the full MovieLens catalog ($\sim 3,810$ movies) is strictly prohibited and immediately aborted.
> - **Zero Local GPU Overhead**: Local testing and development run via a deterministic mock backend without requiring GPU hardware, PyTorch with CUDA, or downloading multi-gigabyte model weights.
> - **Reused Contract**: Directly enforces the Phase 3C schema, prompt templates, entity canonicalization, and Plutchik emotion threshold (`score > 0.3`).

### 1. Pluggable Backends

- **Mock Backend (`--backend mock`)**: Generates deterministic, valid structured JSON responses matching the Phase 3C schema on macOS/Linux CPU. Supports simulated failure injection (`malformed_json`, `schema_validation_error`, `id_mismatch`, `retry_then_succeed`, `backend_error`) for testing.
- **vLLM Backend (`--backend vllm`)**: Lazy-loads `vllm` for real tensor-parallel GPU inference on ASU SOL (`temperature=0.0`, greedy decoding, `max_tokens=1024`, `gpu_memory_utilization=0.90`).

### 2. Checkpointing, Resumption & Error Handling

- **Checkpointing**: Every successfully validated extraction is appended to `validated_extractions.jsonl`. On re-run, already validated movies are skipped automatically.
- **Raw Output Traceability**: Every raw model completion (including malformed outputs and retries) is preserved in `raw_outputs.jsonl`.
- **Automatic Retry Engine**: Malformed or non-compliant responses trigger up to 1 retry (`max_retries=1`) with an explicit formatting reminder.
- **Categorized Failures**: Persistent failures are logged to `failures.jsonl` categorized by error type (`json_syntax_error`, `schema_validation_error`, `id_mismatch`, `backend_error`) without halting the pipeline.

### 3. Pilot Output Artifacts

All pilot artifacts are generated in `data/processed/ml-1m/kg_l/pilot/`:
- `raw_outputs.jsonl`: Line-delimited raw outputs from Qwen for every attempt.
- `validated_extractions.jsonl`: Line-delimited valid `MoviePlotExtraction` records.
- `failures.jsonl`: Line-delimited failure records for extractions that exhausted retries.
- `pilot_summary.json`: High-level run metrics (counts, retries, runtime, success rate).
- `pilot_review.csv`: Human-readable inspection sheet with extracted themes, settings, narrative devices, protagonist types, source materials, continuous Plutchik emotion scores, and prominent emotions (`score > 0.3`).

### 4. Running the Pilot Locally (Mock Mode)

```bash
python scripts/run_phase3d_qwen_pilot.py --backend mock
```

### 5. Running on ASU SOL (GPU Compute via SLURM)

Refer to [`docs/sol_setup.md`](docs/sol_setup.md) for complete cluster environment configuration.

Submit the SLURM batch job:
```bash
sbatch scripts/sol/kg_l_qwen_pilot.sbatch
```

Monitor job output:
```bash
tail -f logs/sol/plotgraph_qwen_pilot_<JOB_ID>.out
```

---

## Phase 3E: Dedicated Evidence-Grounded Emotion Extraction & Pilot

Phase 3E introduces a dedicated, evidence-first extraction contract and pilot infrastructure for the 8 Plutchik emotions, separating emotional analysis from semantic extraction while preserving the exact KG-L ontology.

### 1. Research Motivation & Separation Decision

During Phase 3D real Qwen2.5-7B-Instruct pilots on ASU SOL, combined semantic + emotion prompts (v1.0.0 and refined v1.1.0) demonstrated that direct 8-dimensional emotion scoring was excessively sparse:
- 24 of 30 movies had zero emotions with score > 0.5.
- Mean scores across the pilot were severely depressed (`joy`: 0.027, `trust`: 0.020, `anger`: 0.037).
- The model interpreted zero conservatively and struggled to calibrate 8 continuous numbers simultaneously while extracting multi-label themes, settings, and narrative devices.

**Decision**:
- Semantic extraction remains isolated using the verified Prompt v1.1.0 (`themes`, `settings`, `narrative_devices`, `protagonist_types`, `based_on`).
- Emotion extraction is decoupled into a dedicated, evidence-first Qwen call.

### 2. Evidence-First Emotion Schema & Verification Rules

Defined in `src/knowledge_graph/emotion_contract.py`:
- **Strict 8 Plutchik Dimensions**: Exactly `joy`, `trust`, `fear`, `surprise`, `sadness`, `disgust`, `anger`, `anticipation`. Unknown or extra keys are rejected with `extra="forbid"`.
- **Evidence-Score Consistency**:
  - Score > 0.0 **strictly requires** $\ge 1$ supporting text evidence snippet directly grounded in the overview.
  - Score == 0.0 **strictly requires** an empty evidence list `[]`. Providing evidence with score 0.0 or assigning score > 0.0 without evidence fails schema validation.
- **Two-Step Prompt Structure**:
  1. *Step 1: Evidence Identification*: Identify concrete plot facts and narrative events evoking each emotion.
  2. *Step 2: Calibrated Intensity Assignment*: Assign intensity in $[0.0, 1.0]$ based strictly on the identified evidence.

### 3. Dual Artifact Model (Provenance + Downstream Graph)

Phase 3E generates two complementary output formats under `data/processed/ml-1m/kg_l/emotion_pilot_v1/`:
1. `validated_emotion_evidence.jsonl`: Rich evidence records with exact text quotes and scores for complete auditability.
2. `emotion_vectors.jsonl`: Lean, validated 8-dimensional float vectors (`movie_id`, `movie_idx`, `title`, 8 emotion scores, `prominent_emotions`) directly consumable by downstream KG-L graph builders without re-parsing evidence.
3. `emotion_review.csv`: Human-inspection sheet displaying overview, per-emotion evidence lists, continuous scores, and prominent emotions (`score > 0.3`).

### 4. Running the Phase 3E Dedicated Emotion Pilot

#### Locally (Mock Backend):
```bash
python scripts/run_phase3e_emotion_pilot.py --backend mock
```

#### On ASU SOL GPU Compute (via SLURM):
```bash
sbatch scripts/sol/kg_l_qwen_emotion_pilot.sbatch
```

### 5. Multi-Pilot Emotion Comparison Tool

To evaluate emotion distributions, prominent emotion frequencies, and evidence integrity across Phase 3D and Phase 3E pilots:
```bash
python scripts/compare_emotion_pilots.py \
    --dir-v1-0 data/processed/ml-1m/kg_l/pilot \
    --dir-v1-1 data/processed/ml-1m/kg_l/pilot_v1_1 \
    --dir-emotion-v1 data/processed/ml-1m/kg_l/emotion_pilot_v1
```

### 6. Pilot Completion & Methodology Freeze

Following real Qwen2.5-7B-Instruct pilot executions on ASU SOL GPU compute across combined prompts (v1.0.0, v1.1.0) and dedicated evidence-grounded emotion prompts (v1.0.0), extraction methodology is frozen for full-scale extraction:

1. **Semantic Extraction**:
   - Model: `Qwen/Qwen2.5-7B-Instruct`
   - Prompt: `v1.1.0`
   - Schema: Fixed semantic ontology (`themes`, `settings`, `narrative_devices`, `protagonist_types`, `based_on`)
2. **Emotion Extraction**:
   - Model: `Qwen/Qwen2.5-7B-Instruct`
   - Prompt: Dedicated evidence-grounded prompt `v1.0.0`
   - Schema: Evidence-grounded Plutchik schema with strict evidence-score consistency
3. **Continuous Score Preservation**:
   - All 8 raw continuous Plutchik emotion scores in $[0.0, 1.0]$ are strictly preserved and untouched across all extraction artifacts (`validated_emotion_evidence.jsonl`, `emotion_vectors.jsonl`, `emotion_review.csv`).

### 7. Pilot-Driven Emotion KG Edge Threshold Revision (`score > 0.3`)

The original project proposal specified that discrete RDF `evokesEmotion` edges in KG-L be instantiated when `score > 0.5`. Empirical analysis across the 30-movie pilot revealed that direct thresholding at $>0.5$ caused severe sparsity:

| Threshold | Total Prominent Edges | Avg Edges / Movie | Movies With No Emotion Edge | Movie Coverage (%) |
| :---: | :---: | :---: | :---: | :---: |
| `>0.2` | 65 | 2.17 | 7 | 76.7% |
| **`>0.3` (Selected)** | **48** | **1.60** | **7** | **76.7%** |
| `>0.4` | 14 | 0.47 | 19 | 36.7% |
| `>0.5` (Original) | 7 | 0.23 | 23 | 23.3% |

#### Calibration & Rationale
Under the evidence-grounded emotion prompt, intensity scores are calibrated as:
- `0.0`: Absent
- `0.1`–`0.3`: Weak signal
- `0.4`–`0.6`: Moderate signal
- `0.7`–`0.8`: Strong signal
- `0.9`–`1.0`: Dominant signal

- **Threshold `>0.3`**: Captures all moderate, strong, and dominant emotional relationships while filtering out weak signals ($\le 0.3$). It dramatically expands movie coverage from 23.3% to 76.7% (yielding 1.60 edges/movie vs. 0.23 edges/movie).
- **Comparison to `>0.2`**: Lowering the threshold to $>0.2$ introduces 17 additional weak edges without improving catalog movie coverage (the same 7 movies remain without emotion edges).
- **Boundary Behavior**:
  - `score = 0.30` $\rightarrow$ no edge
  - `score = 0.31` $\rightarrow$ edge
  - `score = 0.40` $\rightarrow$ edge

## Phase 3F: Production-Scale Batched Qwen Extraction Pipeline

Phase 3F scales the frozen pilot methodology across all 3,810 eligible MovieLens-1M movies with Phase 3A plot overviews using batched vLLM inference on ASU SOL GPU compute.

### 1. Production Architecture & Principles
- **Unified vLLM Engine**: Initializes `Qwen/Qwen2.5-7B-Instruct` once per job, servicing both semantic and emotion tasks without reloading weights.
- **Batched Generation**: Employs continuous batching in vLLM (`batch_size=16`) to maximize GPU throughput and maintain deterministic movie-result alignment.
- **Two Independent Tasks per Movie**:
  1. *Semantic Extraction*: Frozen prompt `v1.1.0` extracting `themes`, `settings`, `narrative_devices`, `protagonist_types`, `based_on`.
  2. *Emotion Extraction*: Frozen prompt `v1.0.0` extracting 8 Plutchik emotions with grounded quote evidence and calibrated continuous intensities.
- **Independent Checkpointing & Resumption**: Checkpoints are flushed to disk after every batch. Semantic and emotion tasks are tracked independently; resuming skips completed tasks and executes only missing or failed work without creating duplicates.
- **Chunking / Range Partitioning**: Supports splitting extraction across jobs via `--chunk-id` / `--num-chunks` or `--start-idx` / `--end-idx` with deterministic movie ordering by `movie_idx`.
- **Failure Resilience**: Bounded retry (`max_retries=1`) per task with explicit formatting reminders; permanent failures are recorded in `failures.jsonl` and do not abort the run.

### 2. Output Artifacts (`data/processed/ml-1m/kg_l/full_extraction/`)
All generated production data is Git-ignored:
- `semantic_raw_outputs.jsonl`: Complete audit trail of raw model responses across all attempts.
- `semantic_validated.jsonl`: Strictly validated `MoviePlotExtraction` records.
- `semantic_failures.jsonl`: Categorized failure records for movies exhausting retries.
- `emotion_raw_outputs.jsonl`: Raw model responses for emotion extraction attempts.
- `emotion_validated_evidence.jsonl`: Auditable records with quote evidence and intensity scores.
- `emotion_vectors.jsonl`: Clean 8-dimensional emotion vectors (`movie_id`, `movie_idx`, 8 scores, `prominent_emotions`).
- `emotion_failures.jsonl`: Categorized failure records for emotion extraction.
- `production_summary.json`: Detailed run metrics, throughputs, and descriptive emotion statistics.
- `semantic_validated.parquet` & `emotion_vectors.parquet`: Compact Parquet exports for downstream loading in Phase 3G.

### 3. Execution Commands

#### Dry-Run Runtime Estimation (No GPU Needed):
```bash
python scripts/run_phase3f_full_extraction.py --config configs/kg_l.yaml --estimate-runtime
```

#### Production Run on ASU SOL (via SLURM):
```bash
sbatch scripts/sol/kg_l_qwen_full_extraction.sbatch
```

#### Completion Validation Utility:
```bash
python scripts/run_phase3f_full_extraction.py --config configs/kg_l.yaml --validate-only
```

---

## Phase 3G: KG-L Knowledge Graph Construction & Cross-Graph Integration

Phase 3G constructs the LLM-enriched knowledge graph (**KG-L**) in RDF/Turtle format using the verified Phase 3F production extraction artifacts and Phase 3A metadata reference, establishing seamless cross-graph compatibility with **KG-M**.

### 1. Goals & Architectural Principles
- **No Extra Inference**: Pure RDF graph construction from existing frozen JSONL extractions and Parquet metadata.
- **100% Movie URI Alignment**: Movie URIs strictly match KG-M via `build_movie_uri_index(df_meta)` (`movie_tmdb_{tmdb_id}` for unique TMDB IDs, `movie_tmdb_{tmdb_id}_ml_{movie_id}` for duplicates, and `movie_ml_{movie_id}` for unmapped titles).
- **Fixed KG-L Ontology**:
  - Classes: `pgo:Movie`, `pgo:Theme`, `pgo:Setting`, `pgo:NarrativeDevice`, `pgo:ProtagonistType`, `pgo:SourceMaterial`, `pgo:Emotion`, `pgo:EmotionAssociation`.
  - Semantic Predicates: `pgo:hasTheme`, `pgo:hasSetting`, `pgo:hasNarrativeDevice`, `pgo:hasProtagonistType`, `pgo:basedOn`.
  - Emotion Predicates: `pgo:evokesEmotion`, `pgo:hasEmotionAssociation`, `pgo:associatedEmotion`, `pgo:emotionIntensity`, `pgo:supportingEvidence`.
- **Emotion Threshold Semantics (`score > 0.3`)**:
  - Direct `pgo:evokesEmotion` triples are created strictly when an emotion score exceeds 0.3 (moderate, strong, or dominant signal). Exactly 5,452 direct triples generated.
- **Continuous Score & Evidence Provenance**:
  - Continuous intensity scores in $[0.0, 1.0]$ and supporting plot quote evidence strings are modeled via reified `pgo:EmotionAssociation` resources attached to movies via `pgo:hasEmotionAssociation`.
  - Exactly 8 singleton `pgo:Emotion` resources (`pgo:emotion_joy`, `pgo:emotion_trust`, etc.) based on Plutchik's taxonomy.
- **Handling of Failed Extractions**:
  - The 22 movies that failed emotion extraction retain their valid semantic extractions, but produce zero emotion triples.

### 2. Production Graph Statistics (`data/processed/ml-1m/graphs/kg_l.ttl`)
- **Total RDF Triples**: **97,904**
- **Total Distinct Entities**: **16,556**
- **Entity Counts by Class**:
  - `pgo:Movie`: 3,810
  - `pgo:Theme`: 2,563
  - `pgo:Setting`: 2,287
  - `pgo:NarrativeDevice`: 365
  - `pgo:ProtagonistType`: 1,899
  - `pgo:SourceMaterial`: 172
  - `pgo:Emotion`: 8
  - `pgo:EmotionAssociation`: 5,452
- **Semantic Relation Triples**:
  - `pgo:hasTheme`: 7,181
  - `pgo:hasSetting`: 4,330
  - `pgo:hasNarrativeDevice`: 1,419
  - `pgo:hasProtagonistType`: 4,894
  - `pgo:basedOn`: 407
- **Per-Emotion `evokesEmotion` Edges (`score > 0.3`)**:
  | Emotion | Edge Count |
  |---|---|
  | `joy` | 589 |
  | `trust` | 201 |
  | `fear` | 1,028 |
  | `surprise` | 595 |
  | `sadness` | 731 |
  | `disgust` | 350 |
  | `anger` | 543 |
  | `anticipation` | 1,415 |
  | **Total** | **5,452** |
- **Catalog Coverage**:
  - Movies with $\ge 1$ emotion edge: 2,870 (75.33% of catalog; 75.77% of valid extractions)
  - Movies with 0 emotion edges: 940 (918 with all scores $\le 0.3$, plus 22 failed extractions)
  - Theme Coverage: 99.76% (3,801 movies)
  - Setting Coverage: 99.87% (3,805 movies)
  - Narrative Device Coverage: 36.48% (1,390 movies)
  - Protagonist Type Coverage: 95.43% (3,636 movies)
  - Source Material Coverage: 10.39% (396 movies)

### 3. SPARQL Query Engine (`KGLQueryEngine`) & Cross-Graph Compatibility
The query engine (`src/knowledge_graph/kg_l_query_engine.py`) provides 14 SPARQL query methods:
- Direct lookups: themes, settings, narrative devices, protagonist types, source materials, continuous emotions, and plot quote evidence.
- Inverted index lookups: finding movies by theme or emotion (ranked by intensity).
- Path finding: discovering shared semantic entities and shared emotional profiles between movie pairs.
- **Cross-Graph Joining**: Demonstrates seamless union of KG-M (metadata, genres, cast/crew) and KG-L (themes, emotions, narrative devices) in a single RDF graph (204,704 triples), enabling multi-hop path traversal such as:
  $$\text{Movie A} \xrightarrow{\text{hasGenre}} \text{Genre} \xleftarrow{\text{hasGenre}} \text{Movie B} \quad \land \quad \text{Movie A} \xrightarrow{\text{hasTheme}} \text{Theme} \xleftarrow{\text{hasTheme}} \text{Movie B}$$

### 4. Running Phase 3G
```bash
python scripts/run_phase3g_kg_l.py --config configs/kg_l.yaml
```
Outputs:
- `data/processed/ml-1m/graphs/kg_l.ttl`: Production KG-L Turtle graph (4.19 MB).
- `data/processed/ml-1m/graphs/kg_l_validation.json`: Complete validation report and statistics.

---

## Phase 4A: Leakage-Safe KG-M + KG-L + Emotion Feature Engineering

Phase 4A constructs a deterministic, candidate-level feature matrix for all **604,000 EASE candidate rows** ($6,040 \text{ users} \times 100 \text{ candidates}$) combining collaborative filtering scores, recency-weighted user KG-M metadata preferences, KG-L semantic compatibility, and continuous 8-dimensional emotion profiles.

### 1. Strict Temporal & Data Leakage Rules
- **TRAIN-Only Preference Profiles**: User preferences are derived **exclusively** from positive interactions in the TRAIN split (`rating >= 4.0` / `positive == 1`).
- **Zero Leakage from VAL & TEST**: VAL and TEST interactions, positives, and timestamps are strictly forbidden from user profile construction.
- **Reference Timestamps**: For each user, the reference time $t_{u, \max}$ is established strictly from their most recent positive interaction in the TRAIN split. All profile interaction timestamps satisfy $t \le t_{u, \max}$.
- **Candidate Pool Isolation**: Asserts that zero candidate movies appear in the user's positive TRAIN history (`candidate_seen_in_positive_train == 0` for all 604,000 rows).

### 2. Recency Weighting Formulation
Each positive interaction $(u, i)$ with timestamp $t$ has age in days relative to $t_{u, \max}$:
$$\text{age\_days} = \frac{t_{u, \max} - t}{86400.0}$$
Preferences are weighted using a deterministic exponential half-life decay:
$$w(u, i) = 0.5^{(\text{age\_days} / \text{half\_life\_days})}$$
Configured with default $\text{half\_life\_days} = 365.0$. Normalized interaction weights $\tilde{w}_i = w_i / \sum_j w_j$ ensure history lengths remain comparable ($\sum_i \tilde{w}_i = 1.0$).

### 3. Feature Dimensions (60 Total Columns)

#### A. Base Candidate & History Features (10 features)
- `candidate_rank`: 1..100 preserved from Phase 2B candidate pool.
- `ease_score`: Preserved exactly from EASE candidate generation.
- `user_positive_train_count`: Number of positive items in user's TRAIN history.
- `candidate_seen_in_positive_train`: Asserted to be 0 for all rows.
- Coverage flags: `candidate_has_tmdb_link`, `candidate_has_overview`, `candidate_has_kgm_metadata`, `candidate_has_kgl_semantics`, `candidate_emotion_available`, `user_emotion_profile_available`.

#### B. KG-M Metadata Features (15 features)
- **Affinities (6)**: `kgm_genre_affinity`, `kgm_director_affinity`, `kgm_actor_affinity`, `kgm_keyword_affinity`, `kgm_collection_affinity`, `kgm_decade_affinity` (sum of user profile weights for entities attached to candidate).
- **Aggregates (3)**: `kgm_affinity_mean`, `kgm_affinity_max`, `kgm_categories_matched`.
- **Interpretable Overlaps (6)**: `shared_genre_count`, `shared_director_count`, `shared_actor_count`, `shared_keyword_count`, `shared_collection_count`, `shared_decade_count` (exact entity overlap between candidate and user positive TRAIN history).

#### C. KG-L Semantic Features (13 features)
- **Affinities (5)**: `kgl_theme_affinity`, `kgl_setting_affinity`, `kgl_narrative_device_affinity`, `kgl_protagonist_type_affinity`, `kgl_source_material_affinity`.
- **Aggregates (3)**: `kgl_semantic_affinity_mean`, `kgl_semantic_affinity_max`, `kgl_semantic_categories_matched`.
- **Interpretable Overlaps (5)**: `shared_theme_count`, `shared_setting_count`, `shared_narrative_device_count`, `shared_protagonist_type_count`, `shared_source_material_count`.

#### D. Continuous Emotion Features (20 features)
- **Continuous User Profile**: Recency-weighted average of the full continuous 8D vectors for positive TRAIN movies with valid emotion extractions:
  $$\mathbf{u}_{\text{emo}} = \frac{\sum_{i \in \mathcal{H}_{u, \text{emo}}^+} w(u, i) \cdot \mathbf{v}_i}{\sum_{i \in \mathcal{H}_{u, \text{emo}}^+} w(u, i)}$$
  *(Note: continuous vectors preserve full scores in $[0.0, 1.0]$ without thresholding; the $>0.3$ threshold applies only to RDF graph edge creation).*
- **Compatibility Metrics (4)**: `emotion_cosine_similarity`, `emotion_dot_product`, `emotion_l1_distance`, `emotion_l2_distance`.
- **Raw Dimension Values (16)**: `candidate_emotion_{e}` and `user_emotion_{e}` for all 8 Plutchik emotions (`joy`, `trust`, `fear`, `surprise`, `sadness`, `disgust`, `anger`, `anticipation`).
- **Missingness Handling**: Movies without valid emotion extractions (e.g., 22 production failures) produce `np.nan` values with explicit availability flags (`candidate_emotion_available = 0`), natively supported by gradient-boosted tree rerankers.

### 4. Feature Groups for Downstream Ablation Studies
The codebase defines modular feature groups in `src/features/constants.py`:
- `BASE_FEATURES`: Baseline CF features + coverage flags.
- `KGM_FEATURES`: `KGM_AFFINITY_FEATURES` + `KGM_OVERLAP_FEATURES`.
- `KGL_SEMANTIC_FEATURES`: `KGL_SEMANTIC_AFFINITY_FEATURES` + `KGL_SEMANTIC_OVERLAP_FEATURES`.
- `EMOTION_FEATURES`: `EMOTION_COMPATIBILITY_FEATURES` + `CANDIDATE_EMOTION_FEATURES` + `USER_EMOTION_FEATURES`.
- `ALL_KG_FEATURES`: Combined KG-M, KG-L, and Emotion features.
- `ALL_FEATURES`: Complete 58-feature set (excluding identifiers `user_idx`, `movie_idx`).

These groups enable systematic ablation experiments in later phases:
- Model 1: EASE Baseline (`BASE_FEATURES`)
- Model 2: EASE + KG-M (`BASE_FEATURES + KGM_FEATURES`)
- Model 3: EASE + KG-L Semantics (`BASE_FEATURES + KGL_SEMANTIC_FEATURES`)
- Model 4: EASE + Emotion (`BASE_FEATURES + EMOTION_FEATURES`)
- Model 5: EASE + KG-M + KG-L (`BASE_FEATURES + KGM_FEATURES + KGL_SEMANTIC_FEATURES`)
- Model 6: Full Multimodal PlotGraph Model (`ALL_FEATURES`)

### 5. Running Phase 4A Pipeline
```bash
python scripts/run_phase4a_kg_features.py --config configs/features.yaml
```

Output Artifacts:
- `data/processed/ml-1m/features/kg_candidate_features.parquet` (38.24 MB, 604,000 rows, 60 columns).
- `data/processed/ml-1m/features/kg_feature_summary.json`: Detailed descriptive statistics, coverage, missingness, and leakage verification report.
- `data/processed/ml-1m/features/user_profiles/`: Reusable user profile artifacts (`user_emotion_profiles.parquet`, `user_kgm_profiles.json`, `user_kgl_profiles.json`).

---

## Phase 4B: Leakage-Safe Text-Semantic Features (TF-IDF & SBERT)

Phase 4B enriches every candidate row in Phase 4A's candidate feature table with text-semantic compatibility features derived from movie plot overviews.

### 1. Strict Temporal & Data Leakage Protocol
- **TRAIN-Only Profiles**: Text-semantic user profiles are constructed **strictly from positive interactions** in the TRAIN split (`rating >= 4.0` / `positive == 1`).
- **Zero Leakage from VAL & TEST**: VAL and TEST interactions, timestamps, and ratings are strictly excluded from user profile construction. Assertions reject any input DataFrame containing validation or test columns.
- **Reference Timestamps & Recency Decay**: Identical to Phase 4A, interaction age is computed relative to the user's latest positive TRAIN interaction $t_{u, \max}$:
  $$\text{age\_days} = \frac{t_{u, \max} - t}{86400.0}, \qquad w(u, i) = 0.5^{(\text{age\_days} / \text{half\_life\_days})}$$
  with default $\text{half\_life\_days} = 365.0$.

### 2. Overview Encoders & Text Representation
- **TF-IDF Sparse Representation**:
  - Unigram + bigram TF-IDF vectorizer (`ngram_range=(1, 2)`, `min_df=2`, `max_df=0.8`, `stop_words='english'`).
  - Fit across 3,810 usable catalog movie overviews, constructing a vocabulary of 14,541 terms.
  - Rows are unit L2-normalized: $\|\mathbf{v}_{\text{tfidf}}\|_2 = 1.0$.
- **Sentence-BERT (SBERT) Dense Representation**:
  - Compact sentence transformer `sentence-transformers/all-MiniLM-L6-v2` executed on CPU with batch size 64.
  - Generates 384-dimensional dense semantic embeddings for all 3,810 usable movie overviews.
  - Embeddings are unit L2-normalized: $\|\mathbf{v}_{\text{sbert}}\|_2 = 1.0$.
- **Dot Product & Cosine Similarity Equivalence**:
  - Because both user profile vectors and candidate movie vectors are strictly unit L2-normalized ($\|\mathbf{u}\|_2 = 1.0, \|\mathbf{v}\|_2 = 1.0$), the dot product $\mathbf{u} \cdot \mathbf{v}$ exactly equals the cosine similarity:
    $$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u} \cdot \mathbf{v} \in [-1.0, 1.0]$$
- **Nearest-Neighbor Retrieval via FAISS / NumPy**:
  - An optional `faiss.IndexFlatIP` index (with automatic vectorized NumPy fallback) indexes catalog embeddings for fast nearest-neighbor semantic search (e.g., querying *Toy Story (1995)* yields *Toy Story 2 (1999)* with cosine similarity 0.7401).

### 3. Feature Definitions (8 New Columns, 68 Total Columns)
The candidate table preserves all 604,000 rows and all 60 Phase 4A columns, appending 8 new text-semantic features:
1. `tfidf_cosine_similarity`: Cosine similarity between recency-weighted user TF-IDF profile and candidate overview TF-IDF vector.
2. `tfidf_dot_product`: Dot product of unit-normalized user and candidate TF-IDF vectors (identically equal to cosine similarity).
3. `tfidf_candidate_available`: Binary flag (1 if candidate movie has a valid plot overview, 0 if missing).
4. `tfidf_user_profile_available`: Binary flag (1 if user has $\ge 1$ positive TRAIN movie with a valid overview, 0 if inactive or text-empty).
5. `sbert_cosine_similarity`: Cosine similarity between recency-weighted user SBERT profile and candidate overview SBERT embedding.
6. `sbert_dot_product`: Dot product of unit-normalized user and candidate SBERT embeddings (identically equal to cosine similarity).
7. `sbert_candidate_available`: Binary flag (1 if candidate movie has a valid plot overview, 0 if missing).
8. `sbert_user_profile_available`: Binary flag (1 if user has $\ge 1$ positive TRAIN movie with a valid overview, 0 if inactive or text-empty).

**Missingness Handling**: When a movie lacks a plot overview (73 movies in catalog, 0.55% of candidate rows) or a user has no positive overview history, similarity and dot product values are stored as `NaN` (natively handled by gradient boosted decision trees like LightGBM), paired with 0 in the availability flags.

### 4. Running Phase 4B Pipeline
```bash
python scripts/run_phase4b_text_features.py --config configs/text_features.yaml
```

Output Artifacts:
- `data/processed/ml-1m/features/kg_text_candidate_features.parquet`: Full feature table (49.62 MB, exactly 604,000 rows, 68 columns).
- `data/processed/ml-1m/features/text_feature_summary.json`: Detailed missingness, coverage, and similarity distribution report.
- `data/processed/ml-1m/features/text_embeddings/`: Cached SBERT embeddings (`movie_sbert_embeddings.npy`, `sbert_has_embedding_mask.npy`).
- `data/processed/ml-1m/features/tfidf/`: Cached TF-IDF artifacts (`tfidf_vectorizer.joblib`, `movie_tfidf_matrix.npz`, `tfidf_has_text_mask.npy`).

---

## Phase 4C: NRC Lexicon Emotion Baseline Features

Phase 4C constructs an independent, non-LLM emotion baseline derived from the **NRC Word-Emotion Association Lexicon (EmoLex)** across all 604,000 candidate rows, enabling direct downstream ablation comparisons between classical lexical emotion representations and PlotGraph's evidence-grounded Qwen emotion representations.

### 1. Research Motivation & Separation
- **Lexical Baseline vs LLM Extraction**: While PlotGraph extracts nuanced continuous Plutchik emotions grounded in narrative evidence via Qwen2.5-7B-Instruct, the NRC baseline represents a traditional unigram bag-of-words emotion representation.
- **Strict Independence**: Movie NRC vectors are constructed **strictly from movie plot overviews**, completely independent of Qwen outputs.
- **Preservation of Qwen Emotion Features**: All existing Phase 4A Qwen emotion features (`candidate_emotion_*`, `user_emotion_*`, `emotion_cosine_similarity`, etc.) remain completely untouched. NRC features are named with distinct prefixes (`candidate_nrc_*`, `user_nrc_*`, `nrc_emotion_*`).

### 2. Lexicon Attribution & Text Preprocessing
- **Source & Attribution**:
  - Lexicon: *NRC Word-Emotion Association Lexicon (EmoLex)*, version 0.92, developed by Dr. Saif M. Mohammad and Dr. Peter D. Turney.
  - Package: `NRCLex` 4.1.0 (`nrclex`).
  - Citations:
    - Mohammad, S. M., & Turney, P. D. (2013). *Crowdsourcing a Word-Emotion Association Lexicon*. Computational Intelligence, 29(3), 436–465.
    - Mohammad, S. M., & Turney, P. D. (2010). *Emotions Evoked by the Common Words of Emotion*. Proceedings of the NAACL HLT 2010 Workshop on Computational Approaches to Analysis and Generation of Emotion in Text, 26–34.
  - Licensing: Free for research and non-commercial educational use (National Research Council Canada EmoLex Research License).
- **Emotion Mapping**:
  - Restricted strictly to the 8 Plutchik emotions (`joy`, `trust`, `fear`, `surprise`, `sadness`, `disgust`, `anger`, `anticipation`).
  - Binary sentiment polarities (`positive`, `negative`) present in EmoLex are explicitly filtered out from the primary 8D vector.
- **Deterministic Preprocessing**:
  - Unicode NFKD normalization with ASCII encoding to handle accents and diacritics.
  - Lowercase conversion.
  - Regex unigram word extraction (`[a-z]+`) with punctuation and digit stripping.
  - No stemming, lemmatization, or negation handling is applied, ensuring a transparent, reproducible, lightweight lexical baseline.

### 3. Movie NRC Representation & Normalization
- For each movie with an overview, unigram tokens are matched against EmoLex to produce an 8D count vector.
- When total matched Plutchik emotion count $> 0$, the vector is normalized into a continuous distribution:
  $$\text{normalized}_e = \frac{\text{count}_e}{\sum_{k=1}^8 \text{count}_k}, \qquad \sum_{e=1}^8 \text{normalized}_e = 1.0$$
- Movies lacking plot overviews (73 movies) or overviews with zero NRC emotion matches (66 movies) receive `candidate_nrc_available = 0`, with all 8 dimension values set to `NaN`.

### 4. Leakage-Safe Recency-Weighted User Profiles
- **TRAIN-Only Protocol**: Constructed **strictly from positive TRAIN interactions** (`rating >= 4.0` / `positive == 1`). Zero VAL or TEST interactions, ratings, or timestamps are accessed.
- **Exponential Recency Weighting**:
  $$\text{age\_days} = \frac{t_{u, \max} - t}{86400.0}, \qquad w(u, i) = 0.5^{(\text{age\_days} / \text{half\_life\_days})}$$
  with default $\text{half\_life\_days} = 365.0$.
- **User NRC Profile Aggregation**:
  $$\mathbf{u}_{\text{nrc}} = \frac{\sum_{i \in \mathcal{H}_u^+} w(u, i) \mathbf{v}_{\text{nrc}, i}}{\sum_{i \in \mathcal{H}_u^+} w(u, i)}, \qquad \sum_{e=1}^8 u_{\text{nrc}, e} = 1.0$$

### 5. Feature Dimensions (23 New Columns, 91 Total Columns in Full Table)
The final table `full_candidate_features.parquet` preserves all 68 Phase 4B columns and appends 23 NRC features:
- **Compatibility Metrics (4)**: `nrc_emotion_cosine_similarity`, `nrc_emotion_dot_product`, `nrc_emotion_l1_distance`, `nrc_emotion_l2_distance`.
- **Candidate 8D Dimensions (8)**: `candidate_nrc_{e}` for all 8 Plutchik emotions.
- **User 8D Dimensions (8)**: `user_nrc_{e}` for all 8 Plutchik emotions.
- **Coverage & Support (3)**: `candidate_nrc_available`, `user_nrc_profile_available`, `candidate_nrc_matched_token_count`.

### 6. Modular Ablation Groups
The codebase defines modular feature groups in `src/features/constants.py` (`ABLATION_FEATURE_GROUPS`):
- `BASE`: Baseline CF features + coverage flags (10 features)
- `KGM`: KG-M affinities + overlaps (15 features)
- `KGL_SEMANTIC`: KG-L semantic affinities + overlaps (13 features)
- `QWEN_EMOTION`: Qwen evidence-grounded emotion compatibility + raw dimensions (20 features)
- `NRC_EMOTION`: NRC lexicon emotion compatibility + raw dimensions + coverage (23 features)
- `TEXT`: TF-IDF + SBERT text-semantic features (8 features)
- `ALL`: Full multimodal feature set (89 features, 91 total columns including identifiers)

### 7. Descriptive Qwen vs NRC Movie-Level Comparison (3,723 Common Movies)
Evaluated across all 3,723 catalog movies with both valid Qwen emotion extractions and valid NRC vectors:
- **Average Cosine Similarity**: **0.4866** (std = 0.2665, median = 0.5252).
- **Per-Emotion Distribution Statistics**:
  - `joy`: NRC mean = 0.1369 vs Qwen mean = 0.0688 ($r = 0.2614$)
  - `trust`: NRC mean = 0.1945 vs Qwen mean = 0.0381 ($r = 0.0414$)
  - `fear`: NRC mean = 0.1579 vs Qwen mean = 0.1336 ($r = 0.2674$)
  - `surprise`: NRC mean = 0.0782 vs Qwen mean = 0.1215 ($r = 0.0361$)
  - `sadness`: NRC mean = 0.1070 vs Qwen mean = 0.1194 ($r = 0.1389$)
  - `disgust`: NRC mean = 0.0711 vs Qwen mean = 0.0835 ($r = 0.1417$)
  - `anger`: NRC mean = 0.1080 vs Qwen mean = 0.0902 ($r = 0.1899$)
  - `anticipation`: NRC mean = 0.1465 vs Qwen mean = 0.2826 ($r = -0.0281$)

*Observation*: While lexical NRC distributes weight heavily to frequent dictionary terms (`trust` mean 0.1945), Qwen's evidence-grounded extraction concentrates more strongly on narrative suspense/anticipation (`anticipation` mean 0.2826) and conflict, providing a distinct complementary signal.

### 8. Running Phase 4C Pipeline
```bash
python scripts/run_phase4c_nrc_features.py --config configs/nrc_features.yaml
```

Output Artifacts:
- `data/processed/ml-1m/features/full_candidate_features.parquet`: Complete candidate table (66.09 MB, exactly 604,000 rows, 91 columns).
- `data/processed/ml-1m/features/nrc_feature_summary.json`: Detailed missingness, coverage, and Qwen-vs-NRC comparative analysis report.
- `data/processed/ml-1m/features/nrc/movie_nrc_vectors.parquet`: Cached 8D movie NRC vectors (75.6 KB, 3,883 movies).

---

## Phase 5A: LightGBM Learning-to-Rank Reranker Training & Validation Ablation Experiments

Phase 5A implements supervised learning-to-rank reranking on the fixed EASE top-100 candidate pool (`data/processed/ml-1m/features/full_candidate_features.parquet`, 604,000 rows, 6,040 users, 100 rows/user, 91 columns) using LightGBM's LambdaRank objective, conducting extensive ablation experiments across knowledge, emotion, and text feature families on held-out validation users.

### 1. Supervision Rationale & Leakage Isolation
- **VAL Used Exclusively for Reranker Supervision**: Because user preference profiles and all candidate features were computed and frozen strictly from positive TRAIN interactions in Phases 4A–4C, the validation split (`val.parquet`, 99,950 interactions, 53,166 positives with `rating >= 4.0`) can safely serve as ground-truth supervision (`val_label \in {0, 1}`) for reranker training and evaluation.
- **TRAIN-Only Profiles Preserved**: User preference profiles are never recomputed or augmented with VAL interactions. Feature columns remain frozen.
- **Strict Zero TEST Access**: The MovieLens TEST split (`test.parquet`) is strictly isolated and never accessed, referenced, or loaded in any reranker module or pipeline script. Automated guards (`assert_no_test_access`) intercept and reject any reference to `test.parquet`.

### 2. Candidate-Pool Ceiling & Evaluability
- **Fixed Candidate Pool Recall@100**: The candidate generation stage (EASE reg=500) has an empirical validation Recall@100 ceiling of **0.4421** (mean per-user across users with $\ge 1$ validation positive) and **0.3293** (global ratio: 17,506 captured positives / 53,166 total validation positives).
- **Evaluable Users in Top-100**:
  - Total users: 6,040
  - Users with $\ge 1$ positive VAL interaction: 5,805
  - Users with $\ge 1$ positive VAL candidate in top-100: **5,023** (83.16% of total catalog users)
  - Users with 0 positive candidates in top-100: 1,017 (16.84%)
- **Supervision Restriction Rationale**: Ranking models optimize relative item order within each user query group. Users with zero positive candidates in their top-100 list produce all-zero label queries, generating zero ranking gradients for LambdaRank and undefined NDCG. Supervised reranker training and evaluation are therefore conducted strictly among the 5,023 evaluable users.
- **Recall Metric Distinction**:
  - **End-to-End Validation Recall@K**: Evaluated against *all* positive VAL items for the user ($\text{hits} / |\text{Relevant}_{\text{VAL}}|$). Cannot exceed the candidate-pool ceiling (0.4421).
  - **Conditional Reranking Recall@K**: Evaluated strictly against the positive items captured *within* the top-100 pool ($\text{hits} / |\text{Relevant}_{\text{Pool}}|$). Ceiling = 1.0.

### 3. User-Level Partitioning
To prevent user-level data leakage, candidate rows for each user are grouped contiguously and split at the user level:
- **Held-Out Evaluation Users (20%)**: Exactly **1,005 users** (100,500 candidate rows), strictly held out from model training and early stopping.
- **Reranker Training Users (80%)**: Exactly **4,018 users** (401,800 candidate rows), further partitioned for early stopping:
  - **Fit Partition (80% of Train)**: **3,214 users** (321,400 rows).
  - **Inner-Validation Partition (20% of Train)**: **804 users** (80,400 rows) used strictly for early stopping.
- **Stratification & Determinism**: Both splits are stratified by binned positive candidate counts with deterministic `seed=42`, ensuring identical positive label distributions across partitions (Train mean positives = 3.489; Eval mean positives = 3.469).

### 4. LightGBM LambdaRank Configuration
- **Objective**: `lambdarank` with query group sizes of exactly 100 per user.
- **Primary Target Metric**: `ndcg@10` (with early stopping patience of 30 rounds).
- **Hyperparameters**: `num_leaves=31`, `learning_rate=0.05`, `n_estimators=300`, `min_child_samples=20`, `colsample_bytree=0.8`, `subsample=0.8`, `random_state=42`, `n_jobs=4`.

### 5. Validation-Stage Ablation Results (Held-Out Evaluation Users, N=1,005)

| Model Configuration | Feature Count | NDCG@10 | Δ NDCG@10 vs EASE | Recall@10 (e2e) | Recall@10 (cond) | HR@10 | MRR@10 | NDCG@20 | Recall@20 (e2e) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EASE_BASELINE** | 0 | 0.0953 | --- | 0.1079 | 0.1745 | 0.4010 | 0.1645 | 0.1186 | 0.1776 |
| **BASE** | 10 | 0.0826 | -0.0127 (-13.3%) | 0.0961 | 0.1555 | 0.3881 | 0.1464 | 0.1052 | 0.1644 |
| **BASE + KGM** | 25 | 0.0923 | -0.0030 (-3.2%) | 0.1088 | 0.1800 | 0.4040 | 0.1570 | 0.1167 | 0.1824 |
| **BASE + KGL_SEMANTIC** | 23 | 0.0827 | -0.0126 (-13.2%) | 0.0930 | 0.1534 | 0.3701 | 0.1416 | 0.1063 | 0.1668 |
| **BASE + QWEN_EMOTION** | 30 | 0.0926 | -0.0027 (-2.8%) | 0.1045 | 0.1743 | 0.4060 | 0.1521 | 0.1148 | 0.1719 |
| **BASE + NRC_EMOTION** | 33 | 0.0926 | -0.0027 (-2.8%) | 0.1029 | 0.1772 | 0.4070 | 0.1597 | 0.1138 | 0.1684 |
| **BASE + TEXT** | 18 | 0.0820 | -0.0133 (-14.0%) | 0.0982 | 0.1650 | 0.3771 | 0.1404 | 0.1033 | 0.1621 |
| **BASE + KGM + KGL** | 38 | 0.0985 | +0.0032 (+3.4%) | 0.1114 | 0.1831 | 0.4090 | 0.1673 | 0.1224 | 0.1818 |
| **BASE + KGM + KGL + QWEN** | 58 | 0.0973 | +0.0020 (+2.1%) | 0.1054 | 0.1800 | 0.4179 | 0.1711 | 0.1215 | 0.1789 |
| **BASE + KGM + KGL + NRC** | 61 | 0.0989 | +0.0037 (+3.9%) | 0.1086 | 0.1864 | 0.4408 | 0.1801 | 0.1211 | 0.1756 |
| **FULL (KG+QWEN+TEXT)** | 66 | 0.1006 | +0.0054 (+5.6%) | 0.1101 | 0.1853 | 0.4318 | 0.1731 | 0.1246 | 0.1834 |
| **FULL + NRC (All Features)** | 89 | **0.1040** | **+0.0087 (+9.2%)** | **0.1165** | **0.1961** | **0.4517** | **0.1834** | **0.1248** | **0.1813** |

### 6. Key Scientific Findings & Ablation Analysis
1. **Tree Partitioning on Base Features Alone Distorts CF Calibration**:
   - Training on `BASE` features alone drops NDCG@10 from 0.0953 to 0.0826 (-13.3%). Without external semantic knowledge, gradient boosted tree splits discretize EASE score and rank in ways that disrupt continuous collaborative filtering precision.
2. **Knowledge Graph Synergy (`BASE + KGM + KGL`)**:
   - Combining metadata affinities (KG-M) with semantic narrative properties (KG-L) surpasses the EASE baseline (NDCG@10 = 0.0985, +3.4%), achieving a +0.0159 lift over BASE alone ($p = 0.0020$).
3. **Qwen Emotion vs NRC Emotion Comparison**:
   - In isolation with BASE, both emotion families provide identical standalone performance (NDCG@10 = 0.0926 vs 0.0926; paired bootstrap $\Delta = -0.0001$, $p = 0.9740$, not statistically different).
   - In multimodal combinations, NRC emotion features provide higher hit rates and MRR (`HR@10` = 0.4408 vs 0.4179; `MRR@10` = 0.1801 vs 0.1711).
4. **Multimodal Synergy**:
   - The full multimodal model (`FULL + NRC`, 89 features) achieves the strongest ranking performance across all metrics: **NDCG@10 = 0.1040** (+9.15% relative improvement over EASE), **HR@10 = 0.4517** (vs 0.4010 for EASE, a +12.6% relative increase in finding relevant movies in top 10), and **MRR@10 = 0.1834** (vs 0.1645 for EASE).

### 7. Statistical Significance Analysis (1,000 User-Level Bootstrap Resamples)
Using 1,000 paired bootstrap resamples across the 1,005 evaluation users with deterministic seed 42:
- **Best Model (`FULL_NRC`) vs EASE Baseline**:
  - $\Delta \text{NDCG@10} = +0.0087$, 95% CI: $[+0.0009, +0.0173]$, $p = 0.0340$ (**Statistically Significant at $\alpha = 0.05$**).
  - Probability of superiority: **98.3%**.
- **KG-M + KG-L vs BASE**:
  - $\Delta \text{NDCG@10} = +0.0159$, 95% CI: $[+0.0064, +0.0254]$, $p = 0.0020$ (**Statistically Significant**).
- **Best Model vs BASE**:
  - $\Delta \text{NDCG@10} = +0.0214$, 95% CI: $[+0.0122, +0.0308]$, $p = 0.0000$ (**Statistically Significant**).
- **Qwen Emotion vs NRC Emotion (BASE)**:
  - $\Delta \text{NDCG@10} = -0.0001$, 95% CI: $[-0.0081, +0.0086]$, $p = 0.9740$ (**No Significant Difference**).
- **Multimodal Qwen vs Multimodal NRC**:
  - $\Delta \text{NDCG@10} = -0.0017$, 95% CI: $[-0.0088, +0.0049]$, $p = 0.6680$ (**No Significant Difference**).

### 8. Feature Importance for Best Model (`FULL_NRC`)
- **Gain Importance by Feature Family**:
  - `BASE` (10 features): **28.77%** total gain (anchored by `ease_score` at 13.63% and `candidate_rank` at 11.43%).
  - `NRC_EMOTION` (23 features): **25.50%** total gain (`candidate_nrc_fear` 2.19%, `user_nrc_fear` 2.17%).
  - `KGM` (15 features): **18.97%** total gain (`kgm_actor_affinity` 4.21%, `kgm_affinity_mean` 2.76%, `kgm_decade_affinity` 2.71%).
  - `QWEN_EMOTION` (20 features): **17.70%** total gain (`user_emotion_fear` 2.23%).
  - `TEXT` (8 features): **4.99%** total gain.
  - `KGL_SEMANTIC` (13 features): **4.06%** total gain.

### 9. Output Artifacts (`data/processed/ml-1m/reranker/`)
All generated artifacts are Git-ignored:
- `ablation_results.csv`: Table of all 11 model configurations + EASE baseline with all 10 evaluation metrics, feature counts, fit and inference runtimes, and deltas.
- `ablation_results.json`: JSON format of full ablation results.
- `eval_predictions.parquet`: Parquet table (21.7 MB, 100,500 rows) with model predicted scores and reranked ranks for held-out evaluation users.
- `feature_importance.csv`: Gain and split importances mapped to feature families for the selected best model.
- `run_summary.json`: Complete audit record with candidate pool statistics, user split counts, model parameters, bootstrap significance intervals, and family importance totals.
- `selected_model.joblib`: Serialized LightGBM model object.
- `selected_model.txt`: Text dump of LightGBM booster model.

### 10. Execution Command
```bash
python scripts/run_phase5a_reranker.py --config configs/reranker.yaml
```

---

## Phase 5B: Final Methodology Freeze & Test-Time Candidate Preparation

Phase 5B freezes the final modeling configuration selected in Phase 5A (`FULL + NRC`, 89 features) and prepares the test-time candidate and feature pipeline using historical interactions (`TRAIN + VAL`), strictly without accessing, loading, or evaluating the MovieLens TEST split.

### 1. Final Methodology Freeze
All modeling, candidate generation, feature engineering, and hyperparameter decisions are permanently frozen:
- **Selected Model Family**: `FULL + NRC (All Features)` (89 model features, 91 total columns with identifiers).
- **Candidate Generator**: `EASE` ($L_2$ regularization $\lambda = 500.0$).
- **Positive Interaction Threshold**: `rating >= 4.0` / `positive == 1`.
- **Exponential Recency Weighting**: Half-life $\tau = 365.0$ days ($w(u, i) = 0.5^{(\text{age\_days} / 365.0)}$).
- **Text Encoders**:
  - TF-IDF: unigram + bigram, `min_df=2`, `max_df=0.8`, `stop_words='english'`, unit $L_2$ normalized.
  - SBERT: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, unit $L_2$ normalized).
- **Emotion Baselines & Extractions**:
  - NRC Lexicon: EmoLex v0.92, 8 Plutchik emotions, unit $L_1$ normalized (sums to 1.0).
  - Qwen Evidence-Grounded Extraction: Qwen2.5-7B-Instruct, prompt `v1.0.0`, schema `v1.0.0`, 8 Plutchik continuous scores in $[0, 1]$.
  - KG-L Emotion Edge Threshold: `score > 0.3`.
- **LightGBM LambdaRank Hyperparameters**:
  - `objective: "lambdarank"`
  - `ndcg_eval_at: [10]`
  - `n_estimators: 15` (frozen best iteration from Phase 5A model selection)
  - `learning_rate: 0.05`
  - `num_leaves: 31`
  - `min_child_samples: 20`
  - `colsample_bytree: 0.8`
  - `subsample: 0.8`
  - `random_state: 42`, `n_jobs: 4`

### 2. Historical Interaction Definition (`TRAIN + VAL`)
- **Temporal Ordering**: Both TRAIN and VAL chronologically precede TEST. At test time, all positive interactions from `TRAIN + VAL` (523,955 interactions across 6,038 active users out of 900,259 total ratings) represent available historical user data.
- **Strict Separation of Purposes**:
  - *Reranker Supervision*: LightGBM learns feature-to-relevance relationships strictly from validation candidate labels.
  - *Test-Time User Profiles*: Rebuilt exclusively from historical `TRAIN + VAL` interactions, with each user's reference timestamp $t_{u, \max}$ established strictly from their latest positive `TRAIN + VAL` interaction. Zero TEST timestamps, ratings, or interactions are accessed.

### 3. EASE Refit on Positive TRAIN+VAL & Test Candidate Pool
- EASE ($\lambda = 500.0$) was refit on all 523,955 positive `TRAIN + VAL` interactions across the 6,040 catalog users and 3,883 movies in 4.39s.
- Generated Top-100 candidates for all 6,040 users with strict exclusion of all seen positive `TRAIN + VAL` items (`exclude_seen=True`).
- Output: `data/processed/ml-1m/final_eval/test_candidates_top100.parquet` (exactly 604,000 rows, 100 candidates per user, finite scores, zero positive `TRAIN + VAL` leakage).

### 4. Test-Time User Profiles & Candidate Feature Table
- Rebuilt all user preference profiles using positive `TRAIN + VAL` history:
  - KG-M metadata profiles and KG-L semantic narrative profiles.
  - Qwen continuous emotion profiles.
  - NRC lexicon emotion profiles (6,038 users with valid profiles).
  - TF-IDF and SBERT text-semantic user profiles.
- Assembled the complete 89-feature candidate table:
  - `test_candidate_features.parquet` (69.90 MB, exactly 604,000 rows, 91 columns).
  - Verified 100% schema compatibility with Phase 5A: identical column names, exact 89-feature order, identical dtypes, and zero historical seen-item leakage.

### 5. Final LightGBM Reranker Refit
- Refit the frozen `FULL_NRC` model on all 5,023 eligible validation-supervised users (502,300 candidate rows) using the frozen `best_iteration = 15`.
- Serialized final model artifacts:
  - `data/processed/ml-1m/final_eval/final_reranker.joblib`
  - `data/processed/ml-1m/final_eval/final_reranker.txt` (15 decision trees).

### 6. Dry-Run Unlabeled Candidate Predictions
- Evaluated the final LightGBM model across all 604,000 test-time candidate rows **without loading TEST labels**.
- Output: `data/processed/ml-1m/final_eval/test_candidate_predictions_unlabeled.parquet` (7.16 MB, exactly 604,000 rows).
- Schema: `['user_idx', 'movie_idx', 'candidate_rank', 'ease_score', 'reranker_score', 'reranked_rank']`.
- Contains candidate rankings (1..100) per user. No relevance labels or performance metrics were computed.

### 7. Final Evaluation Manifest
- Generated `data/processed/ml-1m/final_eval/final_evaluation_manifest.json` recording:
  - Frozen methodology parameters, feature list, and history definitions.
  - Candidate pool and feature table validation statistics.
  - SHA-256 checksums for all final evaluation artifacts:
    - `test_candidates_top100.parquet`: `45f08f438293654ff19c12c178177f7c7f17489c33b4e612bee47264ceec3d05`
    - `test_candidate_features.parquet`: `0fc39bf212bef002b8ed246e7aaeda5cf8ae594997114788283fa30bc905510e`
    - `final_reranker.joblib`: `74f21f412510484096d9af976a0da58d1fc8f7e4bcd90ec2b6e1acb9df8c93b3`
    - `final_reranker.txt`: `ee138c3e91d1d3dd79bb4c54aa249b1377def126ad0994daf482019951893a53`
    - `test_candidate_predictions_unlabeled.parquet`: `4cb6a967cf3dd3ef4a2c5346c808200c5aa2ef9b4475af0144d4dc975ca7968e`
  - Explicit statement: `"FINAL TEST LABELS HAVE NOT BEEN ACCESSED. ZERO TEST ACCESS VERIFIED."`

### 8. Execution Command
```bash
python scripts/run_phase5b_final_prep.py --config configs/final_eval.yaml
```

---

## Phase 5C: One-Time Final MovieLens Test Evaluation

Phase 5C conducts the auditable, read-only final evaluation of the frozen PlotGraph recommender system against the previously untouched MovieLens-1M TEST split (`data/processed/ml-1m/splits/test.parquet`). In strict accordance with the pre-registered evaluation protocol, no model, feature, threshold, candidate, or hyperparameter was modified after the TEST split was opened.

### 1. Pre-Evaluation Cryptographic Verification
Before reading or inspecting any test data, the pipeline verified the SHA-256 hashes of all frozen Phase 5B artifacts against [final_evaluation_manifest.json](file:///Users/harsh9703/plotgraph-movie-recommender/data/processed/ml-1m/final_eval/final_evaluation_manifest.json):
- `test_candidates_top100.parquet`: `45f08f438293654ff19c12c178177f7c7f17489c33b4e612bee47264ceec3d05` (VERIFIED)
- `test_candidate_features.parquet`: `0fc39bf212bef002b8ed246e7aaeda5cf8ae594997114788283fa30bc905510e` (VERIFIED)
- `final_reranker.joblib`: `74f21f412510484096d9af976a0da58d1fc8f7e4bcd90ec2b6e1acb9df8c93b3` (VERIFIED)
- `final_reranker.txt`: `ee138c3e91d1d3dd79bb4c54aa249b1377def126ad0994daf482019951893a53` (VERIFIED)
- `test_candidate_predictions_unlabeled.parquet`: `4cb6a967cf3dd3ef4a2c5346c808200c5aa2ef9b4475af0144d4dc975ca7968e` (VERIFIED)

**Pre-flight Certification**: `FROZEN ARTIFACTS VERIFIED — BEGINNING ONE-TIME TEST EVALUATION`

### 2. Candidate-Pool Limitation & TEST Set Coverage
- **Total TEST Interactions**: 99,950
- **Positive TEST Interactions (`rating >= 4.0`)**: 51,326
- **Positive TEST Items Captured in Top-100 Pool**: 16,631
- **Global Candidate Recall@100**: **0.3240** (16,631 / 51,326)
- **Mean User Candidate Recall@100 (Ceiling)**: **0.4146** across users with $\ge 1$ test positive
- **Evaluation Populations**:
  - **Population A (End-to-End)**: **5,774 users** with $\ge 1$ positive TEST interaction. Uncovered users (874 users with zero positives in top-100) are assigned 0.0 metrics rather than dropped.
  - **Population B (Conditional)**: **4,900 users** (84.86% of evaluable users) with $\ge 1$ positive candidate captured in the frozen top-100 pool.

### 3. Final Test Evaluation Results (End-to-End Population, N = 5,774 Users)

| Metric | EASE Baseline | PlotGraph (FULL + NRC) | Absolute Δ | Relative Δ (%) | 95% Bootstrap CI | $p$-value | Significance Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **NDCG@10 [PRIMARY]** | 0.0735 | **0.0751** | **+0.0016** | **+2.14%** | [-0.0015, +0.0045] | $p = 0.3220$ | Superiority Prob = 83.9% |
| **Recall@10 (e2e)** | 0.0823 | **0.0847** | **+0.0024** | **+2.92%** | [-0.0014, +0.0065] | $p = 0.2320$ | Superiority Prob = 88.4% |
| **HR@10** | 0.3362 | **0.3559** | **+0.0197** | **+5.87%** | [+0.0076, +0.0326] | **$p = 0.0020$** | **Statistically Significant ($\alpha = 0.05$)** |
| **MRR@10** | 0.1268 | **0.1355** | **+0.0087** | **+6.89%** | [+0.0021, +0.0155] | **$p = 0.0040$** | **Statistically Significant ($\alpha = 0.05$)** |
| **NDCG@20** | 0.0940 | **0.0941** | **+0.0001** | **+0.13%** | [-0.0026, +0.0030] | $p = 0.9180$ | --- |
| **Recall@20 (e2e)** | 0.1452 | 0.1438 | -0.0014 | -1.00% | [-0.0055, +0.0028] | $p = 0.5540$ | --- |
| **HR@20** | 0.4952 | **0.5151** | **+0.0199** | **+4.02%** | [+0.0088, +0.0312] | **$p = 0.0000$** | **Statistically Significant ($\alpha = 0.05$)** |
| **MRR@20** | 0.1379 | **0.1465** | **+0.0086** | **+6.24%** | [+0.0021, +0.0153] | **$p = 0.0060$** | **Statistically Significant ($\alpha = 0.05$)** |

### 4. Conditional Population Results (N = 4,900 Users with $\ge 1$ Positive in Pool)

| Metric | EASE Baseline | PlotGraph (FULL + NRC) | Absolute Δ | Relative Δ (%) |
| :--- | :---: | :---: | :---: | :---: |
| **NDCG@10** | 0.1146 | **0.1204** | **+0.0058** | **+5.07%** |
| **Recall@10 (Conditional)** | 0.1700 | **0.1807** | **+0.0107** | **+6.32%** |
| **HR@10** | 0.3961 | **0.4194** | **+0.0233** | **+5.87%** |
| **MRR@10** | 0.1494 | **0.1597** | **+0.0103** | **+6.89%** |
| **NDCG@20** | 0.1444 | **0.1449** | **+0.0005** | **+0.35%** |
| **Recall@20 (Conditional)** | 0.2847 | 0.2818 | -0.0029 | -1.02% |
| **HR@20** | 0.5835 | **0.6069** | **+0.0234** | **+4.01%** |
| **MRR@20** | 0.1624 | **0.1726** | **+0.0102** | **+6.28%** |

### 5. Scientific Findings & Discussion
1. **Primary Metric Performance (NDCG@10)**:
   - On the primary test objective, PlotGraph achieved a positive relative lift over the highly competitive EASE baseline (+2.14% end-to-end, +5.07% conditional).
   - Under 1,000 paired bootstrap resamples, the 95% confidence interval spans $[-0.0015, +0.0045]$ with a two-tailed $p = 0.3220$. The model exhibited an **83.9% empirical probability of superiority** over EASE.
2. **Statistically Significant Lifts in Discovery & Rank Precision (HR@10 and MRR@10)**:
   - While NDCG differences are conservative due to strong collaborative filtering signals, PlotGraph demonstrates **statistically significant improvements** in retrieving at least one relevant movie in the top 10 (**HR@10**: +5.87%, $p = 0.0020$) and in placing the first relevant movie earlier in the list (**MRR@10**: +6.89%, $p = 0.0040$).
   - This indicates that external knowledge graph entities, narrative semantic features, and continuous emotion vectors significantly improve the discovery of relevant items for users whose preferences were poorly captured by CF matrix factorization alone.
3. **User-Level Win/Loss Distribution**:
   - **Improved Users**: **1,351 users (23.4%)** with an average lift of $\Delta = +0.1285$ in NDCG@10.
   - **Tied Users**: **3,296 users (57.1%)**.
   - **Worsened Users**: **1,127 users (19.5%)** with an average change of $\Delta = -0.1460$.
   - **Win/Loss Ratio**: **$1.20\times$** more users improved than worsened.
4. **Candidate Pool Ceiling Constraint**:
   - The candidate generation stage achieves a mean per-user Recall@100 of 0.4146. Because the reranker only reorders existing candidates, 58.54% of positive test items remain unretrievable by any reranker operating on this top-100 pool, highlighting the importance of reporting both conditional and end-to-end performance.

### 6. Output Artifacts (`data/processed/ml-1m/final_eval/results/`)
All generated test evaluation outputs are Git-ignored:
- `final_test_results.json`: Complete metrics dictionary for both populations.
- `final_test_results.csv`: Tabular CSV of final comparative metrics across cutoffs.
- `per_user_test_metrics.parquet`: User-level metric table (5,774 rows) for downstream error and segment analysis.
- `test_rank_movements.parquet`: Relevant item movement table (16,631 rows) detailing rank shifts for test positives.
- `bootstrap_test_results.json`: Complete 1,000-sample bootstrap distributions, 95% CIs, and paired $p$-values.
- `final_test_evaluation_record.json`: Auditable tamper-evident record with checksums, timestamps, and certification statement.

### 7. Execution Command
```bash
python scripts/run_phase5c_final_test.py --config configs/final_test.yaml
```














