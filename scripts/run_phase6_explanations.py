#!/usr/bin/env python3
"""Phase 6: Grounded, Auditable Recommendation Explanations.

Generates concise, human-readable explanations grounded in:
1. Actual user historical positive interactions (TRAIN + VAL only)
2. Actual KG-M relationships
3. Actual KG-L relationships
4. Actual continuous Qwen emotion vectors
5. Actual text/NRC compatibility features
6. Actual frozen LightGBM feature contributions (pred_contrib=True)

Validates 100% of explanations with automated zero-hallucination auditor.
Exports recommendation_explanations.parquet and explanation_summary.json.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

from src.explanations.engine import ExplanationEngine, FEATURE_FAMILY_MAP
from src.explanations.path_finder import MovieKnowledgeIndex
from src.explanations.schemas import StructuredExplanation
from src.explanations.validator import ExplanationValidator
from src.final_eval.history import load_train_val_history


def parse_args():
    parser = argparse.ArgumentParser(description="Generate grounded recommendation explanations.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/explanations.yaml",
        help="Path to explanations configuration YAML file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    outputs = config["outputs"]
    params = config["parameters"]

    k_top = int(params.get("k_top", 10))
    min_emotion_threshold = float(params.get("min_emotion_threshold", 0.2))
    half_life_days = float(params.get("recency_half_life_days", 365.0))

    out_dir = Path(outputs["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PLOTGRAPH PHASE 6: GROUNDED, AUDITABLE RECOMMENDATION EXPLANATIONS")
    print("=" * 80)

    # 1. Load frozen final model
    print(f"\n[1/7] Loading frozen final reranker from {inputs['model_path']}...")
    model = joblib.load(inputs["model_path"])
    feature_names = model.booster_.feature_name()
    print(f"  Model loaded successfully with {len(feature_names)} features.")

    # 2. Load unlabeled test predictions and filter for top-k
    print(f"\n[2/7] Loading test candidate predictions from {inputs['predictions_path']}...")
    preds_df = pd.read_parquet(inputs["predictions_path"])
    print(f"  Total candidate rows: {len(preds_df):,}")

    top_recs_df = preds_df[preds_df["reranked_rank"] <= k_top].copy()
    top_recs_df.sort_values(["user_idx", "reranked_rank"], inplace=True)
    top_indices = top_recs_df.index.tolist()
    print(f"  Filtered top-{k_top} recommendations: {len(top_recs_df):,} rows ({top_recs_df['user_idx'].nunique():,} users)")

    # 3. Load candidate features for top-k rows
    print(f"\n[3/7] Loading candidate features from {inputs['features_path']}...")
    t0 = time.time()
    feats_df = pd.read_parquet(inputs["features_path"])
    top_feats_df = feats_df.iloc[top_indices].copy()
    print(f"  Candidate features sliced in {time.time()-t0:.2f}s (shape: {top_feats_df.shape})")

    # 4. Load TRAIN + VAL positive history (strictly zero test leakage)
    print(f"\n[4/7] Loading TRAIN+VAL historical interactions...")
    history_df = load_train_val_history(
        train_path=inputs["train_path"],
        val_path=inputs["val_path"],
    )
    pos_history_df = history_df[history_df["positive"] == 1]
    print(f"  Total positive history interactions: {len(pos_history_df):,}")

    user_pos_history: Dict[int, List[Tuple[int, float, int]]] = {}
    user_pos_sets: Dict[int, Set[int]] = {}
    for uid, group in pos_history_df.groupby("user_idx"):
        tuples = list(zip(group["movie_idx"], group["rating"], group["timestamp"]))
        # Sort chronologically
        tuples.sort(key=lambda x: x[2])
        user_pos_history[int(uid)] = tuples
        user_pos_sets[int(uid)] = set(group["movie_idx"])

    # Optional test set for post-hoc audit/leakage checks and case studies
    test_sets: Dict[int, Set[int]] = {}
    test_pos_pairs: Set[Tuple[int, int]] = set()
    test_path = Path(inputs.get("test_path", "data/processed/ml-1m/splits/test.parquet"))
    if test_path.exists():
        test_df = pd.read_parquet(test_path)
        test_pos = test_df[test_df["positive"] == 1]
        for uid, group in test_pos.groupby("user_idx"):
            test_sets[int(uid)] = set(group["movie_idx"])
        test_pos_pairs = set(zip(test_pos["user_idx"], test_pos["movie_idx"]))
        print(f"  Loaded {len(test_pos):,} TEST positive interactions for descriptive post-hoc analysis.")

    # 5. Build in-memory Knowledge Graph index
    print(f"\n[5/7] Building in-memory MovieKnowledgeIndex...")
    t0 = time.time()
    knowledge_index = MovieKnowledgeIndex(
        metadata_path=inputs["metadata_path"],
        semantic_path=inputs["semantic_path"],
        emotion_vectors_path=inputs["emotion_vectors_path"],
        emotion_evidence_path=inputs.get("emotion_evidence_path"),
    )
    print(f"  Index built in {time.time()-t0:.2f}s ({len(knowledge_index.movie_titles):,} catalog movies)")

    # 6. Compute LightGBM feature contributions
    print(f"\n[6/7] Computing LightGBM tree feature contributions (pred_contrib=True)...")
    t0 = time.time()
    X_top = top_feats_df[feature_names].to_numpy()
    # model.predict(X, pred_contrib=True) returns shape (N, num_features + 1)
    contribs_matrix = model.predict(X_top, pred_contrib=True)
    tree_feat_contribs = contribs_matrix[:, :-1]  # (N, 89)
    base_values = contribs_matrix[:, -1]          # (N,)
    print(f"  Decomposed {len(top_recs_df):,} predictions in {time.time()-t0:.2f}s (matrix shape: {contribs_matrix.shape})")

    # Verify mathematical exactness
    reconstructed_preds = np.sum(contribs_matrix, axis=1)
    max_recon_diff = np.max(np.abs(reconstructed_preds - top_recs_df["reranker_score"].to_numpy()))
    print(f"  Max tree reconstruction difference vs reranker_score: {max_recon_diff:.2e} (Exact match: {max_recon_diff < 1e-4})")

    # 7. Generate and Validate all 60,400 explanations
    print(f"\n[7/7] Generating and auditing grounded explanations for all {len(top_recs_df):,} recommendations...")
    engine = ExplanationEngine(
        knowledge_index=knowledge_index,
        min_emotion_threshold=min_emotion_threshold,
    )
    validator = ExplanationValidator(knowledge_index=knowledge_index)

    explanations: List[StructuredExplanation] = []
    explanation_rows: List[Dict[str, Any]] = []

    type_counts: Dict[str, int] = {}
    kgm_count = 0
    kgl_count = 0
    emotion_count = 0
    text_count = 0
    fallback_count = 0
    total_paths = 0
    total_support_movies = 0
    validation_failures = 0

    t_start = time.time()
    num_recs = len(top_recs_df)

    for i in range(num_recs):
        rec_row = top_recs_df.iloc[i]
        feat_row = top_feats_df.iloc[i]

        uidx = int(rec_row["user_idx"])
        midx = int(rec_row["movie_idx"])
        ease_rank = int(rec_row["candidate_rank"])
        plotgraph_rank = int(rec_row["reranked_rank"])
        reranker_score = float(rec_row["reranker_score"])

        feat_vals = X_top[i]
        feat_contribs = tree_feat_contribs[i]

        user_hist = user_pos_history.get(uidx, [])
        user_pos_set = user_pos_sets.get(uidx, set())
        test_set = test_sets.get(uidx, set())

        # Generate structured explanation
        exp = engine.generate_explanation(
            user_idx=uidx,
            movie_idx=midx,
            ease_rank=ease_rank,
            plotgraph_rank=plotgraph_rank,
            reranker_score=reranker_score,
            feature_row=feat_row,
            feature_names=feature_names,
            feature_values=feat_vals,
            contrib_values=feat_contribs,
            user_positive_history=user_hist,
        )

        # Audit explanation
        val_res = validator.validate(
            explanation=exp,
            user_pos_history_set=user_pos_set,
            test_movies_set=test_set,
            feature_row=feat_row,
        )

        if not val_res.is_valid:
            validation_failures += 1
            print(f"Validation failure for user {uidx}, movie {midx}: {val_res.errors}")

        explanations.append(exp)

        # Track statistics
        exp_type = exp.explanation_type
        type_counts[exp_type] = type_counts.get(exp_type, 0) + 1

        if exp.kgm_paths:
            kgm_count += 1
        if exp.kgl_paths:
            kgl_count += 1
        if exp.emotion_evidence and exp.emotion_evidence.top_aligned_emotions:
            emotion_count += 1
        if exp.text_evidence and exp.text_evidence.sbert_cosine_similarity > 0.4:
            text_count += 1
        if exp_type == "collaborative-fallback":
            fallback_count += 1

        total_paths += len(exp.kgm_paths) + len(exp.kgl_paths)
        total_support_movies += len(exp.historical_support_movies)

        explanation_rows.append({
            "user_idx": uidx,
            "movie_idx": midx,
            "title": exp.title,
            "plotgraph_rank": plotgraph_rank,
            "ease_rank": ease_rank,
            "reranker_score": reranker_score,
            "rank_movement": exp.rank_movement,
            "explanation_type": exp_type,
            "explanation_text": exp.explanation_text,
            "structured_evidence": exp.to_json(),
        })

        if (i + 1) % 15000 == 0 or (i + 1) == num_recs:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  Processed {i+1:,}/{num_recs:,} explanations ({rate:.0f} recs/s)...")

    total_time = time.time() - t_start
    print(f"\nGenerated and audited {num_recs:,} explanations in {total_time:.2f}s ({num_recs/total_time:.0f} recs/s).")

    # Create summary metrics
    summary_data = {
        "total_explanations": num_recs,
        "total_users": top_recs_df["user_idx"].nunique(),
        "coverage_kgm_pct": round(100.0 * kgm_count / num_recs, 2),
        "coverage_kgl_pct": round(100.0 * kgl_count / num_recs, 2),
        "coverage_emotion_pct": round(100.0 * emotion_count / num_recs, 2),
        "coverage_text_pct": round(100.0 * text_count / num_recs, 2),
        "coverage_collaborative_fallback_pct": round(100.0 * fallback_count / num_recs, 2),
        "explanation_type_distribution": {
            k: {"count": v, "percentage": round(100.0 * v / num_recs, 2)}
            for k, v in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        },
        "average_evidence_paths_per_rec": round(total_paths / num_recs, 2),
        "average_support_movies_per_rec": round(total_support_movies / num_recs, 2),
        "validation_metrics": {
            "total_validated": num_recs,
            "validation_failures": validation_failures,
            "validation_pass_rate": round(1.0 - (validation_failures / num_recs), 4),
            "zero_hallucination_verified": (validation_failures == 0),
        },
    }

    # Save output artifacts
    parquet_path = Path(outputs["explanations_parquet"])
    print(f"\nSaving explanations table to {parquet_path}...")
    explanations_df = pd.DataFrame(explanation_rows)
    explanations_df.to_parquet(parquet_path, index=False)
    print(f"  Saved {len(explanations_df):,} rows.")

    summary_path = Path(outputs["summary_json"])
    print(f"Saving summary to {summary_path}...")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Print summary report
    print("\n" + "=" * 80)
    print("PHASE 6 EXPLANATION SUMMARY")
    print("=" * 80)
    print(f"Total Explanations:               {summary_data['total_explanations']:,}")
    print(f"KG-M Metadata Coverage:           {summary_data['coverage_kgm_pct']}%")
    print(f"KG-L Semantic Coverage:           {summary_data['coverage_kgl_pct']}%")
    print(f"Continuous Emotion Coverage:      {summary_data['coverage_emotion_pct']}%")
    print(f"Text Compatibility Coverage:      {summary_data['coverage_text_pct']}%")
    print(f"Collaborative Fallback Coverage:  {summary_data['coverage_collaborative_fallback_pct']}%")
    print(f"Average Grounded Paths/Rec:       {summary_data['average_evidence_paths_per_rec']}")
    print(f"Average Support Movies/Rec:       {summary_data['average_support_movies_per_rec']}")
    print(f"Automated Validation Pass Rate:   {summary_data['validation_metrics']['validation_pass_rate']*100:.2f}%")
    print(f"Validation Failures:              {summary_data['validation_metrics']['validation_failures']}")

    print("\nExplanation Type Breakdown:")
    for etype, data in summary_data["explanation_type_distribution"].items():
        print(f"  - {etype:<26}: {data['count']:>6,} ({data['percentage']:>5.1f}%)")

    # Extract Representative Case Studies
    print("\n" + "=" * 80)
    print("REPRESENTATIVE CASE STUDIES")
    print("=" * 80)

    # Case A: Promoted recommendation (large rank jump)
    promoted = [e for e in explanations if e.rank_movement >= 15 and e.explanation_type != "collaborative-fallback"]
    case_a = promoted[0] if promoted else explanations[0]
    print("\n[Case Study A: Promoted Recommendation]")
    print(f"User: {case_a.user_idx} | Movie: {case_a.title} (idx: {case_a.movie_idx})")
    print(f"Rank Movement: EASE #{case_a.ease_rank} -> PlotGraph #{case_a.plotgraph_rank} (Lift: +{case_a.rank_movement} positions)")
    print(f"Type: {case_a.explanation_type}")
    print(f"Explanation: \"{case_a.explanation_text}\"")
    if case_a.top_feature_contributions:
        top_fc = case_a.top_feature_contributions[0]
        print(f"Top Model Feature: {top_fc.feature_name} (Family: {top_fc.feature_family}, Contrib: +{top_fc.contribution:.4f})")

    # Case B: KG-M driven
    kgm_cases = [e for e in explanations if e.explanation_type == "metadata-driven" and any(p.relation_type in {"directedBy", "inCollection"} for p in e.kgm_paths)]
    case_b = kgm_cases[0] if kgm_cases else [e for e in explanations if e.kgm_paths][0]
    print("\n[Case Study B: KG-M Metadata Driven]")
    print(f"User: {case_b.user_idx} | Movie: {case_b.title} (idx: {case_b.movie_idx})")
    print(f"Type: {case_b.explanation_type}")
    print(f"Explanation: \"{case_b.explanation_text}\"")

    # Case C: KG-L semantic theme driven
    kgl_cases = [e for e in explanations if e.explanation_type == "semantic-theme-driven" and e.kgl_paths]
    case_c = kgl_cases[0] if kgl_cases else [e for e in explanations if e.kgl_paths][0]
    print("\n[Case Study C: KG-L Semantic Theme Driven]")
    print(f"User: {case_c.user_idx} | Movie: {case_c.title} (idx: {case_c.movie_idx})")
    print(f"Type: {case_c.explanation_type}")
    print(f"Explanation: \"{case_c.explanation_text}\"")

    # Case D: Emotion driven
    emo_cases = [e for e in explanations if e.explanation_type == "emotion-driven" and e.emotion_evidence]
    case_d = emo_cases[0] if emo_cases else [e for e in explanations if e.emotion_evidence and e.emotion_evidence.top_aligned_emotions][0]
    print("\n[Case Study D: Emotion Driven]")
    print(f"User: {case_d.user_idx} | Movie: {case_d.title} (idx: {case_d.movie_idx})")
    print(f"Type: {case_d.explanation_type}")
    print(f"Explanation: \"{case_d.explanation_text}\"")

    # Case E: Hybrid
    hybrid_cases = [e for e in explanations if e.explanation_type == "hybrid"]
    case_e = hybrid_cases[0] if hybrid_cases else explanations[0]
    print("\n[Case Study E: Hybrid Grounded Evidence]")
    print(f"User: {case_e.user_idx} | Movie: {case_e.title} (idx: {case_e.movie_idx})")
    print(f"Type: {case_e.explanation_type}")
    print(f"Explanation: \"{case_e.explanation_text}\"")

    # Case F: Fallback
    fallback_cases = [e for e in explanations if e.explanation_type == "collaborative-fallback"]
    case_f = fallback_cases[0] if fallback_cases else explanations[-1]
    print("\n[Case Study F: Collaborative Fallback]")
    print(f"User: {case_f.user_idx} | Movie: {case_f.title} (idx: {case_f.movie_idx})")
    print(f"Type: {case_f.explanation_type}")
    print(f"Explanation: \"{case_f.explanation_text}\"")

    # Case G: Post-hoc descriptive TEST hit
    hit_cases = [e for e in explanations if (e.user_idx, e.movie_idx) in test_pos_pairs]
    if hit_cases:
        case_g = hit_cases[0]
        print("\n[Case Study G: Post-Hoc Descriptive TEST Hit]")
        print(f"User: {case_g.user_idx} | Movie: {case_g.title} (idx: {case_g.movie_idx})")
        print(f"PlotGraph Rank: #{case_g.plotgraph_rank} (EASE Rank: #{case_g.ease_rank})")
        print(f"Type: {case_g.explanation_type}")
        print(f"Explanation: \"{case_g.explanation_text}\"")
        print("Note: TEST relevance was evaluated post-hoc and did NOT influence explanation generation.")

    print("\n" + "=" * 80)
    print("PHASE 6 EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
