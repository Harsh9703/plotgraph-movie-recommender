# PlotGraph Data Architecture & Ingestion Guide

Course: **CSE 573: Semantic Web Mining**  
Institution: **Arizona State University** (Fall 2026)  
Project: **PlotGraph: Knowledge-Graph Movie Recommendation System**

---

## 1. External Datasets Used

PlotGraph builds a hybrid recommendation system by linking collaborative filtering interaction data with knowledge graph ontologies, textual semantic embeddings, and LLM-extracted emotional vectors. Three primary external data sources are utilized:

### A. MovieLens-1M
- **Source**: GroupLens Research, University of Minnesota ([https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/))
- **Role**: Foundational interaction dataset and primary catalog definition.
- **Key Statistics**:
  - **1,000,209 ratings** on a 1–5 integer scale.
  - **6,040 users** (each user having at least 20 ratings).
  - **3,883 unique catalog movies**.
  - Interaction timestamps used for temporal ordering.
- **Processing**:
  - Partitioned using strict chronological per-user splitting: **80% TRAIN**, **10% VAL**, and **10% TEST**.
  - Positive interaction threshold: $\text{rating} \ge 4.0$ defines user ground-truth preference.
- *Note*: Only MovieLens-1M was used in this research project; MovieLens-20M was not used.

### B. The Movies Dataset (TMDB Metadata)
- **Source**: Kaggle / The Movie Database (TMDB) ([https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset))
- **Role**: Semantic entity linking and metadata enrichment for Knowledge Graph M (KG-M) and plot semantic analysis for Knowledge Graph L (KG-L).
- **Files Utilized**:
  - `links.csv`: Maps MovieLens `movieId` to TMDB `id`.
  - `movies_metadata.csv`: Plot overview, genres, release date, runtime, budget, revenue, and collection membership.
  - `credits.csv`: Cast members (actors) and crew (directors).
  - `keywords.csv`: Thematic and stylistic plot keywords.
- **Coverage**:
  - **3,828 movies linked** out of 3,883 catalog items (**98.58%**).
  - **3,810 movies with non-empty plot overviews** (**98.12%**) eligible for LLM semantic and emotion extraction.
  - **3,803 movies with directors** (**97.94%**), **3,768 with cast** (**97.04%**), and **3,342 with keywords** (**86.07%**).

### C. NRC Word-Emotion Association Lexicon (EmoLex)
- **Source**: National Research Council Canada, Mohammad & Turney (2013) ([http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm))
- **Role**: Non-LLM lexical emotion baseline mapping unigram tokens from plot overviews into Plutchik's 8 basic emotional dimensions (*joy, trust, fear, surprise, sadness, disgust, anger, anticipation*).
- **Package**: Loaded via `NRCLex` 4.1.0 to establish a comparative baseline against continuous Qwen2.5-7B-Instruct emotion vectors.

---

## 2. Expected Local Directory Structure

Raw datasets and large reproducible binary tables are excluded from version control to maintain a lightweight repository. To replicate the pipeline locally, place raw datasets in the following structure:

```
data/
├── raw/
│   ├── ml-1m/
│   │   ├── movies.dat         # MovieID::Title::Genres (latin-1)
│   │   ├── ratings.dat        # UserID::MovieID::Rating::Timestamp
│   │   ├── users.dat          # UserID::Gender::Age::Occupation::Zip-code
│   │   └── README
│   └── the-movies-dataset/
│       ├── movies_metadata.csv
│       ├── credits.csv
│       ├── keywords.csv
│       └── links.csv
├── samples/                   # Lightweight committed sample artifacts
│   ├── coverage_report.json   # TMDB catalog coverage metrics
│   └── sample_summary.json    # 30-movie sample schema summary
└── processed/                 # Generated artifacts (Git-ignored)
    └── ml-1m/
        ├── splits/            # train.parquet, val.parquet, test.parquet
        ├── id_maps/           # user_id_map.parquet, movie_id_map.parquet
        ├── metadata/          # movies_with_tmdb.parquet, coverage_report.json
        ├── candidates/        # top100_candidates.parquet (604,000 rows)
        ├── graphs/            # kg_m.ttl (129,658 triples), kg_l.ttl (97,904 triples)
        ├── kg_l/              # Qwen extraction results, prompt pilot checkpoints
        ├── features/          # full_candidate_features.parquet (89 features)
        ├── reranker/          # ablation_results.csv, selected_model.joblib
        ├── final_eval/        # final_reranker.joblib, ease_item_matrix.npy
        └── explanations/      # recommendation_explanations.parquet, explanation_summary.json
```

---

## 3. Generated Artifacts & Reproduction Pipeline

All intermediate and final artifacts are reproducible using the automated scripts in `scripts/`:

| Pipeline Stage | Command | Output Artifact | Description |
| :--- | :--- | :--- | :--- |
| **Data Ingestion & Splits** | `python scripts/download_ml1m.py`<br>`python scripts/run_phase1_pipeline.py` | `data/processed/ml-1m/splits/` | Chronological 80/10/10 split with zero temporal leakage. |
| **Candidate Generation** | `python scripts/run_phase2b_models.py` | `data/processed/ml-1m/candidates/top100_candidates.parquet` | EASE ($\lambda=500$) Top-100 pool (604,000 rows; Recall@100 = 0.4421). |
| **Metadata Graph (KG-M)** | `python scripts/run_phase3a_metadata.py`<br>`python scripts/run_phase3b_kg_m.py` | `data/processed/ml-1m/graphs/kg_m.ttl` | 129,658 RDF triples capturing director, actor, genre, keyword, collection, decade. |
| **Plot Semantics Graph (KG-L)** | `python scripts/run_phase3f_full_extraction.py`<br>`python scripts/run_phase3g_kg_l.py` | `data/processed/ml-1m/graphs/kg_l.ttl` | 97,904 RDF triples capturing themes, settings, devices, protagonist types, and emotions. |
| **Feature Engineering** | `python scripts/run_phase4a_kg_features.py`<br>`python scripts/run_phase4b_text_features.py`<br>`python scripts/run_phase4c_nrc_features.py` | `data/processed/ml-1m/features/full_candidate_features.parquet` | 604,000 candidate rows with 89 features (KG-M, KG-L, Qwen, NRC, TF-IDF, SBERT). |
| **Reranker Training** | `python scripts/run_phase5a_reranker.py`<br>`python scripts/run_phase5b_final_prep.py` | `data/processed/ml-1m/final_eval/final_reranker.joblib` | LightGBM LambdaRank trained on Train+Val with frozen hyperparams. |
| **Final Test Evaluation** | `python scripts/run_phase5c_final_test.py` | `data/processed/ml-1m/final_eval/results/` | One-time evaluation on untouched TEST split ($N=5,774$ evaluable users). |
| **Grounded Explanations** | `python scripts/run_phase6_explanations.py` | `data/processed/ml-1m/explanations/explanation_summary.json` | 60,400 explanations audited for 100% factual evidence consistency. |
| **Demo Artifact Cache** | `python scripts/prepare_demo_artifacts.py` | `data/processed/ml-1m/final_eval/ease_item_matrix.npy` | Pre-computes 57.5 MB EASE matrix for sub-second cold-start recommendations. |

---

## 4. Committed Sample Data

To facilitate code inspection without downloading external datasets, lightweight summary files are committed under [`data/samples/`](file:///Users/harsh9703/plotgraph-movie-recommender/data/samples/):
- **`coverage_report.json`**: Entity linking statistics and TMDB attribute distributions across the MovieLens catalog.
- **`sample_summary.json`**: Schema structure for pilot semantic extractions.
- Quantitative evaluation metrics across all experimental phases are committed under [`evaluations/`](file:///Users/harsh9703/plotgraph-movie-recommender/evaluations/).
