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









