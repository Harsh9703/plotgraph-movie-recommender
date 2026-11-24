# PlotGraph Codebase Organization

Course: **CSE 573: Semantic Web Mining**  
Institution: **Arizona State University** (Fall 2026)  
Project: **PlotGraph: Knowledge-Graph Movie Recommendation System**

---

## Directory Overview

To conform with standard Python packaging standards ([PEP 517 / PEP 621](https://peps.python.org/pep-0621/)), editable package installation (`pip install -e .`), automated test discovery (`pytest`), and the interactive Streamlit runtime, the software implementation is organized across established top-level directories.

This document serves as the guide for navigating the source code, scripts, configuration, and test suites.

```
plotgraph/
├── code/                      # Submission-facing navigation guide (this directory)
│   └── README.md              # Code architecture and component directory
├── src/                       # Core Python library implementation
│   ├── candidate_gen/         # Candidate generators (Popularity, ItemKNN, ALS, BPR, EASE)
│   ├── data/                  # Data loaders, ID mappers, chronological splitters
│   ├── demo/                  # Streamlit cold-start demo backend & visualization
│   ├── explanations/          # Grounded multi-hop explanation engine & validator
│   ├── features/              # Feature engineering (KG, Text/SBERT, Qwen/NRC emotion)
│   ├── kg/                    # Knowledge graph builders (KG-M metadata, KG-L semantics)
│   ├── models/                # LightGBM LambdaRank reranker training & inference
│   └── utils/                 # Configuration loaders, graph helpers, logging
├── scripts/                   # Executable end-to-end pipeline & reproduction scripts
├── configs/                   # YAML configuration files for models, features, and graphs
├── tests/                     # 212 automated unit, integration, and contract tests
└── streamlit_app.py           # Interactive multi-tab web application (Cold-Start Demo)
```

---

## Component Guide

### 1. Core Source Code (`src/`)

| Package / Module | Description | Key Source Files |
| :--- | :--- | :--- |
| **`src/candidate_gen/`** | Classical collaborative filtering and shallow autoencoder candidate generators. Evaluates full-catalog retrieval against validation positives. | [`ease.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/candidate_gen/ease.py), [`als.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/candidate_gen/als.py), [`bpr.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/candidate_gen/bpr.py), [`item_knn.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/candidate_gen/item_knn.py), [`evaluator.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/candidate_gen/evaluator.py) |
| **`src/data/`** | Dataset ingestion, encoding verification, ID mapping (0-indexed consecutive IDs), and strict chronological per-user train/val/test splitting. | [`loader.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/data/loader.py), [`id_mapper.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/data/id_mapper.py), [`splitter.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/data/splitter.py) |
| **`src/kg/`** | RDF triple generation, SPARQL query interface, and ontology management. Constructs KG-M (metadata) and KG-L (LLM plot semantics). | [`kg_m.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/kg/kg_m.py), [`kg_l_builder.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/kg/kg_l_builder.py), [`ontology.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/kg/ontology.py) |
| **`src/features/`** | Feature extraction pipeline computing 89 tabular features across BASE, KG-M, KG-L, Qwen continuous emotions, NRC lexical emotions, and TF-IDF/SBERT text embeddings. | [`kg_features.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/features/kg_features.py), [`text_features.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/features/text_features.py), [`nrc_features.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/features/nrc_features.py), [`user_profiler.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/features/user_profiler.py) |
| **`src/models/`** | LightGBM LambdaRank reranking framework with group-aware pairwise ranking, early stopping, and NDCG@10 optimization. | [`reranker.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/models/reranker.py), [`metrics.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/models/metrics.py) |
| **`src/explanations/`** | Deterministic, grounded explanation engine finding multi-hop paths across KG-M and KG-L, emotion profile alignment, tree contribution decomposition, and 100% automated evidence validation. | [`engine.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/explanations/engine.py), [`path_finder.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/explanations/path_finder.py), [`validator.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/explanations/validator.py), [`schemas.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/explanations/schemas.py) |
| **`src/demo/`** | Backend service adapter for the interactive Streamlit cold-start demo, handling catalog lookup, on-the-fly candidate scoring, feature vectorization, and Graphviz rendering. | [`recommender.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/demo/recommender.py), [`cold_start.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/demo/cold_start.py), [`explanation_adapter.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/demo/explanation_adapter.py), [`visualizations.py`](file:///Users/harsh9703/plotgraph-movie-recommender/src/demo/visualizations.py) |

---

### 2. Execution Scripts (`scripts/`)

All pipeline stages are automated through reproducible Python scripts:

| Script | Purpose |
| :--- | :--- |
| `download_ml1m.py` | Automated download and extraction of GroupLens MovieLens-1M dataset. |
| `run_phase1_pipeline.py` | Chronological train/val/test splitting (80/10/10) with leakage checks. |
| `run_phase2a_baselines.py` | Baseline candidate generation (Popularity and ItemKNN). |
| `run_phase2b_models.py` | ALS, BPR, and EASE grid search; exports frozen Top-100 candidate pool. |
| `run_phase3a_metadata.py` | TMDB metadata ingestion, cleaning, deduplication, and MovieLens linkage. |
| `run_phase3b_kg_m.py` | Builds KG-M (129,658 triples) into Turtle (`kg_m.ttl`). |
| `run_phase3f_full_extraction.py` | Orchestrates Qwen2.5-7B structured semantic & emotion extraction. |
| `run_phase3g_kg_l.py` | Builds KG-L (97,904 triples) into Turtle (`kg_l.ttl`). |
| `run_phase4a_kg_features.py` | Computes KG-M, KG-L, and Qwen emotion candidate features. |
| `run_phase4b_text_features.py` | Computes TF-IDF and SBERT semantic compatibility features. |
| `run_phase4c_nrc_features.py` | Computes NRC Lexicon emotion features as an independent baseline. |
| `run_phase5a_reranker.py` | Trains 11 LightGBM reranker ablations on validation split; selects `FULL_NRC`. |
| `run_phase5b_final_prep.py` | Freezes production configuration and trains final model on Train+Val. |
| `run_phase5c_final_test.py` | One-time read-only evaluation against untouched MovieLens-1M TEST split. |
| `run_phase6_explanations.py` | Generates 60,400 grounded explanations and performs 100% automated audit. |
| `prepare_demo_artifacts.py` | Verifies and pre-caches demo artifacts for sub-second cold-start latency. |

---

### 3. Automated Test Suite (`tests/`)

The repository includes **212 tests** covering every component:

- `test_loader.py`, `test_id_mapper.py`, `test_splitter.py`: Data ingestion, ID mappings, and zero-leakage splits.
- `test_popularity.py`, `test_item_knn.py`, `test_als.py`, `test_bpr.py`, `test_ease.py`: Candidate generator mechanics.
- `test_kg_m.py`, `test_kg_l_builder.py`, `test_kg_l_contract.py`: RDF triple construction and SPARQL queries.
- `test_feature_matrix.py`, `test_text_features.py`, `test_nrc_features.py`: Feature mathematical correctness and leakage isolation.
- `test_reranker.py`, `test_ranking_metrics.py`, `test_final_test_eval.py`: Pairwise ranking evaluation and bootstrap tests.
- `test_explanations.py`: Explanation grounding and anti-hallucination validation.
- `test_demo.py`: Cold-start recommendation pipeline, Graphviz generation, and session-state adapters.

Run the test suite with:
```bash
pytest -q
```

---

### 4. Interactive Web Application (`streamlit_app.py`)

The presentation demo is a standalone interactive application:

```bash
python -m streamlit run streamlit_app.py
```

It contains three comprehensive tabs:
1. **Recommend**: Cold-start movie selection, real-time EASE candidate generation, 89-feature LightGBM reranking, and rank promotion badges.
2. **Explain**: Interactive multi-hop Graphviz knowledge graph diagram, KG-M vs KG-L path breakdown, continuous Qwen 8D Plutchik emotion radar/bar comparison, and LightGBM tree contribution decomposition.
3. **Research**: Complete offline experimental findings, including the 12-configuration validation ablation chart, final untouched TEST performance metrics ($p$-values and 95% CIs), and zero-hallucination explanation audit statistics.
