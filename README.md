# PlotGraph: Knowledge-Graph Movie Recommendation System

**A hybrid movie recommendation architecture combining collaborative filtering, knowledge graphs, LLM-extracted plot semantics, continuous emotional vectors, learning-to-rank, and grounded multi-hop explainability.**

---

### Course Information
- **Course**: CSE 573: Semantic Web Mining
- **Institution**: Arizona State University
- **Semester**: Fall 2026
- **Project**: Software & Research Implementation

---

## 1. Overview & Problem Statement

Recommender systems in streaming and media platforms face a fundamental trade-off:
- **Collaborative Filtering (CF)** methods (such as matrix factorization, ItemKNN, or shallow autoencoders like EASE) excel at capturing behavioral user-item interaction signals, yet they treat items as opaque indices with zero intrinsic understanding of plot, narrative tropes, or emotional arcs.
- **Content-Based & Knowledge Graph (KG)** approaches offer rich semantic relations (directors, genres, themes) and explainability, but frequently underperform behavioral CF in pure ranking precision and catalog coverage.
- **Large Language Models (LLMs)** demonstrate unprecedented narrative comprehension and common-sense reasoning, but naive prompt-based recommenders suffer from severe latency, high inference costs, lack of global collaborative context, and hallucinations.

**PlotGraph** resolves this dichotomy through a principled multi-tier architecture:
1. High-throughput collaborative filtering generates a candidate pool from behavioral interactions.
2. Dual semantic knowledge graphs (**KG-M** for metadata and **KG-L** for LLM plot semantics) enrich items with structured entities and continuous emotional vectors.
3. A gradient-boosted pairwise learning-to-rank reranker (**LightGBM LambdaRank**) fuses collaborative scores, graph affinities, textual embeddings, and continuous Plutchik emotion dimensions.
4. A deterministic, zero-hallucination explanation engine synthesizes human-readable, auditable multi-hop paths connecting historical user choices to recommended items.
5. An interactive Streamlit demo provides cold-start personalized recommendations and live graph visualizations.

---

## 2. System Architecture

The PlotGraph pipeline processes raw interactions, extracts semantic graphs, constructs ranking features, and serves recommendations through the following end-to-end data flow:

```
[ User Interaction History (MovieLens-1M) ]
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Tier 1: High-Recall Candidate Generator (EASE λ=500)   │
│  - Inverts closed-form item-item Gram matrix            │
│  - Generates Top-100 candidates per user (Recall=0.442) │
└────────────────────────────┬────────────────────────────┘
                             │ Top-100 Candidates
                             ▼
┌─────────────────────────────────────────────────────────┐
│  Tier 2: Multi-Modal Semantic Representation & Graphs   │
│  ├── KG-M: Metadata Graph (129,658 triples)             │
│  │   └── Director, Cast, Genre, Keyword, Collection     │
│  ├── KG-L: LLM Plot Semantics (97,904 triples)          │
│  │   └── Themes, Settings, Devices, Protagonist Types   │
│  ├── Qwen2.5-7B-Instruct Continuous 8D Emotion Vectors  │
│  ├── NRC Lexicon 8D Emotion Baseline                    │
│  └── Plot Text Semantics (TF-IDF unigram+bigram, SBERT) │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│  Tier 3: Feature Engineering (89 Tabular Features)      │
│  - Candidate signals, recency-weighted user profiles,   │
│    graph Jaccard affinities, cosine & dot similarities  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│  Tier 4: Pairwise Reranker (LightGBM LambdaRank)        │
│  - 89 features, MART pairwise objective, 15 estimators  │
│  - Reranks candidates into final personalized Top-K     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│  Tier 5: Grounded Explanation Engine & Web Demo         │
│  - Multi-hop Graphviz graph traversal (KG-M & KG-L)     │
│  - 100% automated validation (Zero Hallucination)       │
│  - Interactive Streamlit cold-start interface           │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Key Components

1. **Candidate Generator (`EASE`)**: Embarrassingly Shallow Autoencoders ($\lambda = 500.0$) compute an item-item weight matrix $B = -P \cdot (\text{diag}(P))^{-1}$ with a zero diagonal. It provides collaborative filtering retrieval with a validation Recall@100 of 0.4421.
2. **Metadata Knowledge Graph (`KG-M`)**: An RDF knowledge graph constructed from TMDB metadata containing 129,658 triples across 3,883 movies, capturing directors, actors, genres, thematic keywords, franchise collections, and release decades.
3. **LLM Plot Semantics Graph (`KG-L`)**: An RDF knowledge graph populated by structured extraction using Qwen2.5-7B-Instruct over movie plot overviews. Contains 97,904 triples covering 2,563 narrative themes, 2,287 settings, 365 narrative devices, 1,899 protagonist types, 172 source material types, and 5,452 prominent emotion edges.
4. **Continuous Plutchik Emotion Profiles**: Quantitative 8-dimensional vectors in $[0, 1]$ representing Plutchik's primary emotions (*joy, trust, fear, surprise, sadness, disgust, anger, anticipation*), extracted from narrative evidence via Qwen2.5-7B-Instruct.
5. **Lexical Emotion Baseline (`NRC EmoLex`)**: An independent non-LLM emotion baseline derived from the NRC Word-Emotion Association Lexicon to contrast keyword counts against contextual LLM understanding.
6. **Text-Semantic Representations**: Unigram+bigram TF-IDF sparse vectors (14,541 features) and dense sentence embeddings (`all-MiniLM-L6-v2`, 384 dimensions) capturing semantic plot compatibility between user history and candidates.
7. **Learning-to-Rank Reranker (`LightGBM LambdaRank`)**: A pairwise LambdaMART reranker trained on 89 features spanning 6 feature families (`BASE`, `KGM`, `KGL_SEMANTIC`, `QWEN_EMOTION`, `NRC_EMOTION`, `TEXT`).
8. **Grounded Explanation Engine**: Traverses shared entity paths across KG-M and KG-L, compares continuous emotion alignments, extracts LightGBM tree split contributions, and outputs structured, zero-hallucination natural language explanations audited against ground-truth history.
9. **Interactive Streamlit Web Demo**: A cold-start interactive web application enabling new users to select movies, generate EASE candidates, run LightGBM reranking, and inspect multi-hop graph diagrams in real time.

---

## 4. Dataset & Preprocessing

The primary interaction dataset is **MovieLens-1M**, enriched with metadata from **The Movies Dataset** (TMDB):

| Dataset | Total Records | Linked Catalog | Characteristics |
| :--- | :---: | :---: | :--- |
| **MovieLens-1M Interactions** | 1,000,209 ratings | 3,883 movies | 6,040 users with $\ge 20$ ratings; scale 1–5 |
| **MovieLens Split (Chronological)** | 80% Train, 10% Val, 10% Test | Per-user split | Strict temporal ordering; zero future interaction leakage |
| **Positive Threshold** | $r \ge 4.0$ | Ground-truth likes | Defines positive user preference for profile construction |
| **TMDB Metadata Linkage** | 3,828 movies (98.58%) | 3,810 overviews (98.12%) | Directors (97.9%), cast (97.0%), keywords (86.1%), collections (17.2%) |

*Note*: MovieLens-1M was used exclusively; MovieLens-20M was not used. Detailed ingestion and schema documentation is provided in [`data/README.md`](file:///Users/harsh9703/plotgraph-movie-recommender/data/README.md).

---

## 5. Knowledge Graphs

PlotGraph defines formal OWL/RDFS ontologies stored in W3C Turtle (`.ttl`) format and queried via SPARQL:

```
Combined Graph Triples: 227,562
├── KG-M (Metadata Graph): 129,658 triples (3,883 movies, 22,510 total entities)
└── KG-L (Plot Semantics Graph): 97,904 triples (3,810 movies, 16,556 total entities)
```

- **KG-M Triples (`129,658`)**:
  - `hasKeyword`: 25,591 | `starsActor`: 18,502 | `hasGenre`: 9,120 | `title`: 3,883 | `directedBy`: 3,803 | `releaseDecade`: 3,824 | `inCollection`: 669 | RDF/RDFS labels & types: 45,020.
- **KG-L Triples (`97,904`)**:
  - `hasTheme`: 7,181 (99.8% coverage) | `hasSetting`: 4,330 (99.9% coverage) | `hasProtagonistType`: 4,894 (95.4% coverage) | `hasNarrativeDevice`: 1,419 (36.5% coverage) | `basedOn`: 407 | `evokesEmotion`: 5,452 prominent edges (threshold $> 0.3$) | `hasEmotionAssociation`: 5,452 reified continuous emotion resources.

---

## 6. Empirical Evaluation

All evaluations follow strict temporal isolation: validation ablations guided model selection without touching test interactions, followed by a one-time final test evaluation.

### A. Candidate Generation Validation (Phase 2B)
Evaluated across **5,805 validation users** on full-catalog retrieval against held-out positive items ($r \ge 4.0$):

| Candidate Model | Configuration | NDCG@10 | Recall@10 | HR@10 | MRR@10 | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Popularity** | Non-personalized frequency | 0.0490 | 0.0432 | 0.2305 | 0.0954 | 0.135s |
| **BPR** | factors=32, reg=0.01, iters=50 | 0.0592 | 0.0704 | 0.2734 | 0.1009 | 4.625s |
| **ItemKNN** | $k=40$ neighbors, cosine | 0.0679 | 0.0772 | 0.3077 | 0.1161 | 2.011s |
| **ALS** | factors=32, reg=0.01, iters=15 | 0.0722 | 0.0844 | 0.3276 | 0.1216 | 1.396s |
| **EASE (Selected)** | $\lambda = 500.0$ | **0.0776** | **0.0909** | **0.3414** | **0.1295** | 3.491s |

*Selection*: EASE ($\lambda=500.0$) achieved the highest validation NDCG@10 and was frozen as the candidate generator.

### B. Validation-Stage Feature Ablations (Phase 5A)
Evaluated across **5,023 validation users** to select the final LightGBM LambdaRank reranker configuration:

| Model Configuration | Feature Families | Total Features | Val NDCG@10 | Relative Lift vs EASE |
| :--- | :--- | :---: | :---: | :---: |
| **EASE Baseline** | Candidate scores only | 0 | 0.0953 | — |
| `BASE` | EASE score & candidate counts | 10 | 0.0826 | -13.33% |
| `BASE_TEXT` | BASE + TF-IDF + SBERT | 18 | 0.0820 | -13.95% |
| `BASE_KGL_SEMANTIC` | BASE + KG-L themes/settings | 23 | 0.0827 | -13.19% |
| `BASE_KGM` | BASE + KG-M metadata | 25 | 0.0923 | -3.18% |
| `BASE_QWEN_EMOTION` | BASE + Qwen 8D emotions | 30 | 0.0926 | -2.87% |
| `BASE_NRC_EMOTION` | BASE + NRC 8D emotions | 33 | 0.0926 | -2.81% |
| `BASE_KGM_KGL` | BASE + KG-M + KG-L | 38 | 0.0985 | +3.35% |
| `BASE_KGM_KGL_QWEN` | BASE + KG-M + KG-L + Qwen | 58 | 0.0973 | +2.09% |
| `BASE_KGM_KGL_NRC` | BASE + KG-M + KG-L + NRC | 61 | 0.0989 | +3.83% |
| `FULL` | BASE + KGM + KGL + QWEN + TEXT | 66 | 0.1006 | +5.63% |
| **`FULL_NRC` (Selected)** | **All 6 feature families** | **89** | **0.1040** | **+9.23%** |

*Selection*: `FULL_NRC` achieved the highest validation performance (+9.2% relative lift, 95% bootstrap CI $[+0.0009, +0.0173]$) and was frozen for production evaluation.

### C. Final Untouched TEST Results (Phase 5C)
One-time read-only evaluation on the untouched TEST split ($N = 5,774$ evaluable users, 1,000 bootstrap iterations):

| Metric | EASE Baseline | PlotGraph (FULL+NRC) | Relative Lift | Paired $p$-value | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NDCG@10** | 0.0735 | **0.0751** | **+2.14%** | $p = 0.322$ | Not statistically significant (95% CI $[-0.0015, +0.0045]$) |
| **Recall@10** | 0.0823 | **0.0847** | **+2.92%** | $p = 0.232$ | Not statistically significant |
| **HR@10** | 0.3362 | **0.3559** | **+5.87%** | **$p = 0.002$** | **Statistically Significant** (95% CI $[+0.0076, +0.0326]$) |
| **MRR@10** | 0.1268 | **0.1355** | **+6.89%** | **$p = 0.004$** | **Statistically Significant** (95% CI $[+0.0021, +0.0155]$) |
| **HR@20** | 0.4952 | **0.5151** | **+4.02%** | **$p < 0.001$** | **Statistically Significant** |
| **MRR@20** | 0.1379 | **0.1465** | **+6.24%** | **$p = 0.006$** | **Statistically Significant** |

*Key Takeaway*: PlotGraph provides statistically significant gains in discovery and retrieval rank (**HR@10 +5.87%**, **MRR@10 +6.89%**). While the NDCG@10 lift was positive (+2.14%), it did not achieve statistical significance, demonstrating that semantic knowledge graph reranking functions primarily as an effective high-precision discovery filter. Complete evaluation tables and figures are located in [`evaluations/`](file:///Users/harsh9703/plotgraph-movie-recommender/evaluations/).

---

## 7. Grounded Explainability & Audit

Rather than relying on unconstrained generative language models at inference time, PlotGraph employs a **deterministic, grounded explanation engine** audited across all **60,400 top-10 recommendations**:

- **Zero-Hallucination Audit**: **100.00% validation pass rate** across all 60,400 recommendations. Every cited entity, director, actor, theme, or emotion is strictly traced to ground-truth KG triples.
- **Temporal Integrity**: **0.00% test leakage**. Explanations reference only positive training and validation history.
- **Evidence Coverage**:
  - **99.95%** cite verified KG-M metadata links.
  - **93.71%** cite verified KG-L plot semantic themes and narrative devices.
  - **80.20%** align continuous Plutchik emotional dimensions.
  - **0.04%** fall back to collaborative-only explanations (only 26 out of 60,400 recommendations lacked KG links).
- **Average Evidence Density**: **9.49 grounded paths** per recommendation.

---

## 8. Interactive Cold-Start Demo

The repository includes a web application for real-time presentation and user interaction:

```bash
python -m streamlit run streamlit_app.py
```

### Cold-Start User Workflow
1. **Catalog Search & Selection**: A new user (with no prior MovieLens ID) selects 3 to 10 favorite movies from the 3,883-title catalog or chooses a preset (*Mainstream Mixed*, *Sci-Fi / Adventure*, *Crime / Drama*).
2. **On-the-Fly Profile Construction**: An equal-weight user preference profile ($w_i = 1/N$) is synthesized across metadata, semantics, text, and emotions.
3. **Sub-Second Reranking**: The precomputed EASE matrix ($\lambda=500$) scores all 3,883 items, retrieves the top 100 candidates (excluding selected movies), vectorizes 89 features, and reranks them with LightGBM in sub-second execution (~700–800 ms warm latency).
4. **Interactive Multi-Hop Explanations**: The user can inspect any recommendation to view:
   - Dynamic **Graphviz SVG network diagram** linking selected movies to recommendations via shared entities.
   - Side-by-side **continuous Plutchik emotion comparison** (user profile vs recommendation).
   - LightGBM tree contribution decomposition revealing additive feature influences.
5. **Research Tab**: Embedded interactive charts displaying the 12-configuration validation ablation study, final test metrics, and zero-hallucination audit statistics.

*Scope Note*: The demo supports cold-start **users** over the fixed MovieLens-1M catalog; it does not process unseen external movie titles.

---

## 9. Repository Structure

The repository is organized into submission-facing and implementation directories:

```
plotgraph/
├── code/                      # Submission-facing navigation guide
│   └── README.md              # Code architecture and component directory
├── data/                      # Dataset ingestion guide and sample metadata
│   ├── README.md              # Ingestion instructions and pipeline reproduction
│   └── samples/               # Lightweight committed coverage and sample summaries
├── evaluations/               # Committed experimental results and figures
│   ├── README.md              # Detailed evaluation report
│   ├── candidate_models.csv   # Phase 2B candidate generator metrics
│   ├── validation_ablations.csv # Phase 5A held-out validation ablations
│   ├── final_test_results.csv # Phase 5C final TEST metrics
│   ├── bootstrap_test_results.json # Paired bootstrap significance data
│   ├── feature_importance.csv # LightGBM 89-feature importance
│   ├── kg_statistics.json     # KG-M and KG-L graph statistics
│   └── figures/               # Presentation-ready PNG evaluation figures
├── src/                       # Core Python library implementation
│   ├── candidate_gen/         # EASE, ALS, BPR, ItemKNN, Popularity
│   ├── data/                  # Loaders, ID mappers, chronological splitters
│   ├── demo/                  # Demo backend, recommender adapter, visualizations
│   ├── explanations/          # Grounded path finder, explanation engine, validator
│   ├── features/              # 89 tabular feature pipelines (KG, Text, Emotion)
│   ├── kg/                    # RDF knowledge graph builders (KG-M & KG-L)
│   ├── models/                # LightGBM LambdaRank training & inference
│   └── utils/                 # Configuration loaders, graph helpers
├── scripts/                   # Executable reproduction and evaluation scripts
├── configs/                   # YAML configurations for candidate models and features
├── tests/                     # 212 automated unit, integration, and contract tests
├── docs/                      # Technical documentation and demo presentation guide
├── streamlit_app.py           # Interactive Streamlit cold-start application
├── pyproject.toml             # Python package specifications and dependencies
└── .gitignore                 # Strict ignore rules for large/reproducible data
```

---

## 10. Installation & Setup

### Environment Configuration
```bash
# 1. Clone repository
git clone https://github.com/your-username/plotgraph-movie-recommender.git
cd plotgraph-movie-recommender

# 2. Create and activate Python virtual environment (Python 3.10+ recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install project in editable mode with development dependencies
pip install -e .
pip install streamlit graphviz matplotlib altair
```

*Note*: Large raw datasets and generated binary tables (>500 MB) are excluded from version control. Precomputed evaluation summaries are committed under [`evaluations/`](file:///Users/harsh9703/plotgraph-movie-recommender/evaluations/). To reconstruct binary feature tables or retrain models locally, refer to [`data/README.md`](file:///Users/harsh9703/plotgraph-movie-recommender/data/README.md).

---

## 11. Running & Reproduction

### Automated Test Suite
Run all 212 automated tests:
```bash
pytest -q
```

### Interactive Cold-Start Demo
Launch the Streamlit web application:
```bash
python -m streamlit run streamlit_app.py
```
*(If `data/processed/ml-1m/final_eval/ease_item_matrix.npy` is missing, regenerate it in ~4 seconds via `python scripts/prepare_demo_artifacts.py`).*

### Core Pipeline Reproduction
To re-run individual experimental stages, execute the corresponding script from `scripts/`:
- **Ingestion & Splits**: `python scripts/run_phase1_pipeline.py`
- **Candidate Generation**: `python scripts/run_phase2b_models.py`
- **Knowledge Graphs**: `python scripts/run_phase3b_kg_m.py && python scripts/run_phase3g_kg_l.py`
- **Feature Engineering**: `python scripts/run_phase4a_kg_features.py && python scripts/run_phase4b_text_features.py && python scripts/run_phase4c_nrc_features.py`
- **Reranker Training**: `python scripts/run_phase5a_reranker.py`
- **Grounded Explanations**: `python scripts/run_phase6_explanations.py`

---

## 12. Reproducibility & Integrity Guarantees

PlotGraph enforces strict scientific integrity:
- **Temporal Splitting**: 80/10/10 per-user chronological split with zero future interaction leakage.
- **Test Isolation**: All model selections, ablation studies, and hyperparameter tuning were completed exclusively on the validation split prior to a single, read-only final test evaluation.
- **Deterministic Randomness**: Fixed random seeds (`seed=42`) across all sampling, splitting, matrix factorizations, and bootstrap evaluations.
- **Zero Generative Hallucination**: Explanations are deterministic graph queries audited against ground-truth RDF triples rather than unconstrained generative text.

---

## 13. Technologies Used

- **Language & Core Libraries**: Python 3.11, NumPy, pandas, SciPy.
- **Knowledge Graphs & Semantic Web**: RDFlib, SPARQL, W3C Turtle, OWL/RDFS ontologies.
- **Machine Learning & Ranking**: LightGBM (LambdaRank), scikit-learn, Implicit.
- **Natural Language Processing & Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`), scikit-learn (TF-IDF), NRCLex (Word-Emotion Association Lexicon).
- **Large Language Models**: Qwen2.5-7B-Instruct via vLLM (offline structured semantic & emotion extraction).
- **Interactive Presentation & Visualization**: Streamlit, Graphviz, Altair, Matplotlib.
- **Quality Assurance**: Pytest (212 unit, integration, and contract tests).

---

## 14. Limitations & Future Work

1. **Candidate Pool Ceiling**: The reranker operates on the Top-100 candidates produced by EASE. Relevant items outside this top-100 window cannot be recovered by the reranker (validation Recall@100 = 0.4421).
2. **Statistical Significance on NDCG**: While Hit Rate (**HR@10 +5.87%**, $p=0.002$) and Reciprocal Rank (**MRR@10 +6.89%**, $p=0.004$) showed statistically significant improvements on the untouched test set, the NDCG@10 lift (+2.14%) did not reach statistical significance ($p=0.322$).
3. **Cold-Start Catalog Scope**: The interactive demo is designed for cold-start *users* across the catalog of 3,883 represented movies. Unseen external movies require metadata extraction before inclusion.
4. **Plot Overview Dependency**: LLM semantic and emotion extraction is bounded by the quality, length, and detail of TMDB plot overviews.

---

## 15. Citations & Attribution

- **MovieLens-1M**: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
- **EASE**: Harald Steck. 2019. *Embarrassingly Shallow Autoencoders for Sparse Data*. In The World Wide Web Conference (WWW '19). ACM, 3251–3257.
- **NRC Emotion Lexicon**: Saif M. Mohammad and Peter D. Turney. 2013. *Crowdsourcing a Word-Emotion Association Lexicon*. Computational Intelligence, 29(3): 436–465.
- **Qwen2.5**: Qwen Team, Alibaba Cloud. 2024. *Qwen2.5: A Comprehensive Technical Report*.
- **The Movie Database (TMDB)**: Metadata provided by TMDB under non-commercial community license.
