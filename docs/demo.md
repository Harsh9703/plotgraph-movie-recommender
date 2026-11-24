# PlotGraph: Interactive Streamlit Demo Guide

This guide documents the local setup, execution, architecture, and live presentation sequence for the **PlotGraph Interactive Cold-Start Recommendation Demo**.

---

## 1. Overview & Demo Philosophy

The PlotGraph Streamlit application demonstrates cold-start movie recommendation for a brand-new person with no prior MovieLens history.

### Core Principles
- **Real Inference**: Recommendations are computed in real time using the frozen EASE item-item matrix ($\lambda = 500.0$), the exact 89-feature candidate schema, and the frozen LightGBM LambdaRank reranker.
- **No Training at Demo Time**: The demo never trains models, runs Qwen inference, rebuilds graphs, or downloads external data. All expensive work is precomputed.
- **Cold-Start Semantics**: The user's selected movies function as their positive preference history with equal weighting ($w_i = 1/N$). No historical timestamps or ratings are fabricated.
- **Zero Hallucination Explanations**: Multi-hop paths and emotion alignments are strictly audited against verified knowledge graphs and continuous vectors. No generative LLM is queried at inference time.
- **Completely Offline on macOS**: No internet connection, GPU, or ASU SOL cluster required.

---

## 2. Prerequisites & Environment Setup

### 1. Python Environment
Ensure your Python 3.11 virtual environment is active:

```bash
source .venv/bin/activate
```

### 2. Dependencies
Streamlit and Graphviz are installed in the project environment:
- `streamlit >= 1.30.0`
- `graphviz >= 0.20.0`
- `altair >= 5.0.0`
- `lightgbm`, `pandas`, `scipy`, `pyarrow`, `joblib`

---

## 3. Required Frozen Artifacts

The application requires the following precomputed artifacts:

| Artifact | Location | Purpose |
| :--- | :--- | :--- |
| **Catalog Metadata** | `data/processed/ml-1m/metadata/movies_with_tmdb.parquet` | Movie titles, TMDB links, credits, genres |
| **Frozen EASE Matrix** | `data/processed/ml-1m/final_eval/ease_item_matrix.npy` | 3,883 $\times$ 3,883 item-item weight matrix |
| **Final Reranker** | `data/processed/ml-1m/final_eval/final_reranker.joblib` | Frozen LightGBM LambdaRank model (89 features) |
| **Plot Semantics (KG-L)** | `data/processed/ml-1m/kg_l/full_extraction/semantic_validated.parquet` | Qwen extracted themes, settings, devices |
| **Emotion Vectors** | `data/processed/ml-1m/kg_l/full_extraction/emotion_vectors.parquet` | Continuous 8D Plutchik vectors in $[0, 1]$ |
| **NRC Emotion Vectors** | `data/processed/ml-1m/features/nrc/movie_nrc_vectors.parquet` | Lexicon-based baseline emotion features |
| **TF-IDF Artifacts** | `data/processed/ml-1m/features/tfidf/` | Fitted vectorizer and catalog matrix |
| **SBERT Artifacts** | `data/processed/ml-1m/features/text_embeddings/` | Precomputed dense MiniLM-L6 embeddings |
| **Validation Results** | `data/processed/ml-1m/reranker/ablation_results.json` | 11-model ablation results from Phase 5A |
| **Final TEST Results** | `data/processed/ml-1m/final_eval/results/final_test_results.json` | Frozen one-time test evaluation metrics |
| **Explanation Summary** | `data/processed/ml-1m/explanations/explanation_summary.json` | Grounding metrics across 60,400 explanations |

### Offline Preparation Command
If `ease_item_matrix.npy` is ever missing, reconstruct it with the one-time preparation script:

```bash
python scripts/prepare_demo_artifacts.py
```
*(Runs in ~4 seconds on positive TRAIN+VAL history without tuning).*

---

## 4. Running the Demo

Launch the Streamlit app locally:

```bash
streamlit run streamlit_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 5. Recommended Live Presentation Sequence (2–3 Minutes)

The demo is optimized for classroom presentations or stakeholder walkthroughs:

```
[ Step 1: Recommend Tab ]
 ├── Choose "Preset A: Mainstream Mixed" (Toy Story, The Matrix, Titanic)
 ├── Click "Get Recommendations"
 └── Highlight the interactive / sub-second latency in observed warm runs (measured dynamically)

[ Step 2: Reranking & Lift ]
 ├── Show top recommendation cards (e.g. Forrest Gump promoted +17 positions from EASE #18)
 ├── Point out: EASE candidate rank vs final PlotGraph rank
 └── Read the concise grounded "Why?" bullets

[ Step 3: Explain Tab ]
 ├── Click "Inspect Details" on the top recommendation
 ├── Show the interactive Graphviz multi-hop Knowledge Graph diagram
 ├── Contrast KG-M (metadata: director, collection) vs KG-L (plot semantics: themes, settings)
 └── Inspect the continuous Qwen 8D emotion profile comparison (radar/bar chart)

[ Step 4: Research Tab ]
 ├── Switch to the Research tab
 ├── Review the held-out validation ablation chart (+9.2% lift for FULL+NRC)
 ├── Present final frozen TEST improvements (HR@10 +5.87% p=0.002, MRR@10 +6.89% p=0.004)
 └── Emphasize the 100.00% automated validation pass rate (zero hallucinated claims)
```

---

## 6. Demo Presets

Three built-in presets allow instant demonstration without typing:

- **Preset A — Mainstream Mixed**:
  - *Toy Story (1995)*
  - *Matrix, The (1999)*
  - *Titanic (1997)*
  - *Demonstrates*: Rich cross-genre hybrid reranking with franchise, director, and emotional profiles.
- **Preset B — Sci-Fi / Adventure**:
  - *Star Wars: Episode IV - A New Hope (1977)*
  - *Jurassic Park (1993)*
  - *Terminator 2: Judgment Day (1991)*
  - *Back to the Future (1985)*
  - *Demonstrates*: Franchise continuity, Spielberg/Cameron/Zemeckis director overlaps, and high-anticipation plot themes.
- **Preset C — Crime / Drama**:
  - *Godfather, The (1972)*
  - *Pulp Fiction (1994)*
  - *GoodFellas (1990)*
  - *Shawshank Redemption, The (1994)*
  - *Demonstrates*: Auteur-driven recommendations, gritty narrative devices, and dark dramatic themes.

---

## 7. Screenshot Guide for Slides

For presentations or reports, three primary screenshots capture the entire project:

1. **Recommendation Screen**:
   - Tab: **Recommend**
   - State: Preset A selected, recommendations generated.
   - Captures: Searchable selection box, sub-second latency badge, and recommendation cards displaying EASE $\rightarrow$ PlotGraph rank movements.
2. **Knowledge Graph & Emotion Explanation Screen**:
   - Tab: **Explain**
   - State: Top recommendation selected.
   - Captures: Multi-hop Graphviz diagram connecting selected movies to the candidate through explicit theme and director nodes, alongside the 8D continuous emotion profile comparison.
3. **Empirical Research Findings Screen**:
   - Tab: **Research**
   - State: Scrolled to validation ablation and final TEST result sections.
   - Captures: Validation NDCG@10 ablation bar chart, final TEST comparison chart, and statistical significance badges.

---

## 8. Troubleshooting

- **Port in use**: If port 8501 is busy, specify an alternate port:
  ```bash
  streamlit run streamlit_app.py --server.port 8502
  ```
- **Graphviz visual warning**: If system Graphviz binaries (`dot`) are not installed on the host Mac, install via Homebrew (`brew install graphviz`). The app gracefully degrades to textual path breakdowns if the system binary is absent.
- **Resetting state**: Refresh the browser tab or click any Preset button to reset session state.
