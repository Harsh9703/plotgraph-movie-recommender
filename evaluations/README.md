# PlotGraph Empirical Evaluations & Experimental Artifacts

Course: **CSE 573: Semantic Web Mining**  
Institution: **Arizona State University** (Fall 2026)  
Project: **PlotGraph: Knowledge-Graph Movie Recommendation System**

---

## 1. Overview & Directory Contents

This directory contains the committed, verified evaluation records and summary artifacts from all experimental phases of PlotGraph. All numbers are derived from frozen evaluation runs and reproduce the findings presented in the project report and demo presentation.

```
evaluations/
├── README.md                      # Comprehensive experimental report (this file)
├── candidate_models.csv           # Phase 2B full-catalog candidate generation metrics
├── validation_ablations.csv       # Phase 5A held-out validation reranker ablations (12 configs)
├── final_test_results.csv         # Phase 5C end-to-end and conditional TEST metrics
├── final_test_results.json        # Machine-readable final TEST evaluation record
├── bootstrap_test_results.json    # Paired bootstrap significance testing (1,000 samples)
├── feature_importance.csv         # LightGBM feature gain and split counts for all 89 features
├── feature_family_importance.csv  # Feature importance aggregated by feature family
├── kg_statistics.json             # KG-M (129,658 triples) and KG-L (97,904 triples) metrics
├── qwen_vs_nrc.json               # Descriptive alignment metrics between Qwen and NRC emotions
├── explanation_summary.json       # Phase 6 explanation audit and grounded coverage statistics
└── figures/                       # Presentation-ready evaluation figures (PNG)
    ├── candidate_ndcg_comparison.png
    ├── validation_ablations_ndcg.png
    ├── final_test_comparison.png
    ├── feature_family_importance.png
    └── explanation_coverage.png
```

---

## 2. Candidate Generation Experiments (Phase 2B)

Five candidate generators were evaluated on full-catalog retrieval across **5,805 validation users** who had at least one positive interaction ($r \ge 4.0$) in the validation split. Training was strictly isolated to positive training history (`positive == 1`, $r \ge 4.0$).

| Model | Hyperparameters | NDCG@10 | Recall@10 | HR@10 | MRR@10 | NDCG@20 | Recall@20 | HR@20 | MRR@20 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity** | non-personalized | 0.0490 | 0.0432 | 0.2305 | 0.0954 | 0.0617 | 0.0857 | 0.3611 | 0.1044 |
| **BPR** | factors=32, reg=0.01, iters=50 | 0.0592 | 0.0704 | 0.2734 | 0.1009 | 0.0761 | 0.1215 | 0.4169 | 0.1107 |
| **ItemKNN** | $k=40$ neighbors, cosine | 0.0679 | 0.0772 | 0.3077 | 0.1161 | 0.0883 | 0.1383 | 0.4711 | 0.1275 |
| **ALS** | factors=32, reg=0.01, iters=15 | 0.0722 | 0.0844 | 0.3276 | 0.1216 | 0.0916 | 0.1433 | 0.4844 | 0.1323 |
| **EASE** | $\lambda = 500.0$ | **0.0776** | **0.0909** | **0.3414** | **0.1295** | **0.0984** | **0.1540** | **0.5104** | **0.1410** |

- **Selection**: **EASE ($\lambda = 500.0$)** achieved the highest validation NDCG@10 (**0.0776**, +14.28% relative lift over ItemKNN and +7.48% over ALS).
- **Candidate Pool**: EASE exported exactly 100 candidates per user ($6,040 \times 100 = 604,000$ candidate rows) with **zero positive training leakage** and validation **Recall@100 = 0.4421** (mean per user) / 0.3293 (global).

---

## 3. Validation Reranker Ablations (Phase 5A)

During model development, 11 LightGBM LambdaRank feature configurations were trained on training interactions and evaluated across **5,023 validation users** (having candidate positives) to select the optimal model before accessing the TEST split.

| Model / Configuration | Feature Families Included | Total Features | Val NDCG@10 | Rel. Lift vs EASE | Val HR@10 | Val MRR@10 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EASE Baseline** | Candidate generator rank/score only | 0 | 0.0953 | — | 0.4010 | 0.1645 |
| `BASE` | EASE score, candidate/history counts | 10 | 0.0826 | -13.33% | 0.3881 | 0.1464 |
| `BASE_TEXT` | BASE + TF-IDF + SBERT | 18 | 0.0820 | -13.95% | 0.3771 | 0.1404 |
| `BASE_KGL_SEMANTIC` | BASE + KG-L themes/settings/devices | 23 | 0.0827 | -13.19% | 0.3701 | 0.1416 |
| `BASE_KGM` | BASE + KG-M director/actor/genre/keywords | 25 | 0.0923 | -3.18% | 0.4040 | 0.1570 |
| `BASE_QWEN_EMOTION` | BASE + Qwen 8D continuous emotions | 30 | 0.0926 | -2.87% | 0.4060 | 0.1521 |
| `BASE_NRC_EMOTION` | BASE + NRC 8D lexical baseline | 33 | 0.0926 | -2.81% | 0.4070 | 0.1597 |
| `BASE_KGM_KGL` | BASE + KG-M + KG-L | 38 | 0.0985 | +3.35% | 0.4090 | 0.1673 |
| `BASE_KGM_KGL_QWEN` | BASE + KG-M + KG-L + Qwen Emotions | 58 | 0.0973 | +2.09% | 0.4179 | 0.1711 |
| `BASE_KGM_KGL_NRC` | BASE + KG-M + KG-L + NRC Emotions | 61 | 0.0989 | +3.83% | 0.4408 | 0.1801 |
| `FULL` | BASE + KGM + KGL + QWEN + TEXT | 66 | 0.1006 | +5.63% | 0.4318 | 0.1731 |
| **`FULL_NRC` (Selected)** | **All 6 feature families** | **89** | **0.1040** | **+9.23%** | **0.4517** | **0.1834** |

- **Validation Selection**: `FULL_NRC` achieved the highest validation NDCG@10 (**0.1040**, **+9.2% relative lift** over EASE, paired bootstrap 95% CI $[+0.0009, +0.0173]$).
- All 89 features and hyperparameters were **frozen** prior to evaluating on the final TEST split.

---

## 4. Final Untouched MovieLens-1M TEST Results (Phase 5C)

A one-time read-only evaluation was performed against the previously untouched TEST split across **5,774 evaluable test users** (users with $\ge 1$ positive interaction in the test window).

### End-to-End Test Performance ($N = 5,774$ Users)

| Metric | EASE Baseline | PlotGraph (FULL+NRC) | Absolute $\Delta$ | Relative Lift | Paired $p$-value | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **NDCG@10** | 0.0735 | **0.0751** | +0.0016 | **+2.14%** | $p = 0.322$ | Not statistically significant (95% CI $[-0.0015, +0.0045]$) |
| **Recall@10** | 0.0823 | **0.0847** | +0.0024 | **+2.92%** | $p = 0.232$ | Not statistically significant |
| **HR@10** | 0.3362 | **0.3559** | +0.0197 | **+5.87%** | **$p = 0.002$** | **Statistically Significant** (95% CI $[+0.0076, +0.0326]$) |
| **MRR@10** | 0.1268 | **0.1355** | +0.0087 | **+6.89%** | **$p = 0.004$** | **Statistically Significant** (95% CI $[+0.0021, +0.0155]$) |
| **NDCG@20** | 0.0940 | **0.0941** | +0.0001 | +0.13% | $p = 0.918$ | Not statistically significant |
| **Recall@20** | 0.1452 | 0.1438 | -0.0015 | -1.00% | $p = 0.612$ | Not statistically significant |
| **HR@20** | 0.4952 | **0.5151** | +0.0199 | **+4.02%** | **$p < 0.001$** | **Statistically Significant** |
| **MRR@20** | 0.1379 | **0.1465** | +0.0086 | **+6.24%** | **$p = 0.006$** | **Statistically Significant** |

### Scientific Interpretation
1. **Discovery & Conversion**: PlotGraph significantly improves Hit Rate (**HR@10 +5.87%**, $p=0.002$) and Mean Reciprocal Rank (**MRR@10 +6.89%**, $p=0.004$). The reranker successfully promotes relevant items into top viewing slots.
2. **NDCG@10 Lift**: While positive (+2.14%, superiority probability 83.9%), the NDCG@10 lift did not reach the $\alpha = 0.05$ threshold on the one-time test set. This establishes that knowledge-graph reranking acts primarily as a high-precision discovery and filtering mechanism rather than a sheer score scaler.

---

## 5. Knowledge Graph Statistics

Both graphs are stored in standard W3C Turtle format and queried via SPARQL:

```
Combined Graph Triples: 227,562
├── KG-M (Metadata Knowledge Graph): 129,658 triples (3,883 movies)
└── KG-L (Plot Semantics & Emotions): 97,904 triples (3,810 movies)
```

| Dimension | KG-M (Metadata Graph) | KG-L (Plot Semantics & Emotions Graph) |
| :--- | :--- | :--- |
| **Triples** | **129,658** | **97,904** |
| **Movie Entities** | 3,883 | 3,810 |
| **Other Entities** | 10,417 persons, 7,815 keywords, 364 collections, 20 genres, 11 decades | 2,563 themes, 2,287 settings, 1,899 protagonist types, 365 narrative devices, 172 source materials, 8 Plutchik emotions, 5,452 emotion associations |
| **Primary Relations** | `directedBy` (3,803), `starsActor` (18,502), `hasGenre` (9,120), `hasKeyword` (25,591), `inCollection` (669), `releaseDecade` (3,824) | `hasTheme` (7,181), `hasSetting` (4,330), `hasNarrativeDevice` (1,419), `hasProtagonistType` (4,894), `basedOn` (407), `evokesEmotion` (5,452) |
| **Coverage** | 98.58% linked to TMDB metadata | 99.76% theme coverage, 99.87% setting coverage, 75.33% prominent emotion coverage |

---

## 6. Qwen2.5-7B vs NRC Lexicon Analysis

Across 3,723 movies sharing both representations:
- **Cosine Similarity**: Mean = 0.4866, Median = 0.5252, Std = 0.2665.
- **Correlation by Emotion**:
  - Highest correlation: *fear* ($r = 0.2674$) and *joy* ($r = 0.2614$).
  - Moderate correlation: *anger* ($r = 0.1899$), *disgust* ($r = 0.1417$), *sadness* ($r = 0.1389$).
  - Lowest correlation: *surprise* ($r = 0.0361$) and *trust* ($r = 0.0414$).
- **Conclusion**: Qwen captures contextual, narrative-grounded emotional nuance from sentences, whereas NRC counts surface word frequencies. Neither represents absolute ground truth, but their combination in `FULL_NRC` provided the strongest ranking signal.

---

## 7. Grounded Explanation Evaluation (Phase 6)

The explanation engine was evaluated across **60,400 recommendations** (Top-10 for all 6,040 MovieLens users):

| Evaluation Metric | Measured Result | Interpretation |
| :--- | :---: | :--- |
| **Total Explanations Evaluated** | 60,400 | Complete top-10 coverage for all 6,040 users |
| **Automated Validation Pass Rate** | **100.00%** | Zero fabricated entities, relations, or historical items |
| **Test Split Contamination** | **0.00%** | Explanations strictly reference positive training/validation history |
| **KG-M Evidence Coverage** | **99.95%** | 60,370 / 60,400 recommendations cite verified metadata links |
| **KG-L Evidence Coverage** | **93.71%** | 56,601 / 60,400 recommendations cite verified plot semantics |
| **Qwen Continuous Emotion Coverage** | **80.20%** | 48,443 / 60,400 recommendations align continuous Plutchik dimensions |
| **Text Semantic Coverage** | **51.82%** | SBERT/TF-IDF plot overview compatibility |
| **Collaborative-Only Fallback** | **0.04%** | Only 26 recommendations lacked KG paths |
| **Average Grounded Paths / Recommendation** | **9.49 paths** | Rich multi-hop explanation graphs |
