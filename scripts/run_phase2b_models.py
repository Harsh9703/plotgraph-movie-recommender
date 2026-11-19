"""Script to tune, compare, select, and export top-100 candidates for Phase 2B."""

import argparse
from pathlib import Path
import sys
import time
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.candidate_gen.popularity import PopularityRecommender
from src.candidate_gen.item_knn import ItemKNNRecommender
from src.candidate_gen.als import ALSRecommender
from src.candidate_gen.bpr import BPRRecommender
from src.candidate_gen.ease import EASERecommender
from src.candidate_gen.evaluator import (
    evaluate_candidate_model,
    tune_grid,
    generate_and_save_candidate_pool,
    evaluate_candidate_pool_recall,
)


def run_phase2b(config_path: str = "configs/candidates.yaml") -> None:
    print(f"=== Running Phase 2B Classical Models & Tuning with config: {config_path} ===")
    config = load_config(config_path)

    train_path = Path(config["paths"]["train_split"])
    val_path = Path(config["paths"]["val_split"])
    movie_map_path = Path(config["paths"]["movie_id_map"])
    user_map_path = Path(config["paths"]["user_id_map"])

    for p in [train_path, val_path, movie_map_path, user_map_path]:
        if not p.is_file():
            print(f"[ERROR] Required file not found: {p.resolve()}")
            print("Please run Phase 1 pipeline first: python scripts/run_phase1_pipeline.py")
            sys.exit(1)

    print("\n[1/6] Loading data splits and ID mappings...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    movie_map_df = pd.read_parquet(movie_map_path)
    user_map_df = pd.read_parquet(user_map_path)

    num_users = len(user_map_df)
    num_items = len(movie_map_df)
    all_users = list(range(num_users))
    cutoffs = config["evaluation"]["cutoffs"]
    exclude_seen = config["evaluation"]["exclude_seen"]
    top_k_candidates = config["evaluation"].get("top_k_candidates", 100)

    train_pos = train_df[train_df["positive"] == 1]
    val_pos = val_df[val_df["positive"] == 1]

    print(f"      Universe: {num_users:,} users, {num_items:,} catalog movies")
    print(f"      TRAIN: {len(train_df):,} total ratings | {len(train_pos):,} positive interactions (r >= 4.0)")
    print(f"      VAL  : {len(val_df):,} total ratings | {len(val_pos):,} positive interactions (r >= 4.0)")
    print("      RESEARCH INTEGRITY CHECK: TEST split is strictly excluded. Tag Genome is NOT loaded.")

    final_comparison = []

    # --- 1. Popularity Baseline ---
    print("\n[2/6] Evaluating Popularity baseline...")
    pop = PopularityRecommender(name="Popularity")
    t0 = time.perf_counter()
    pop.fit(train_df, num_users=num_users, num_items=num_items)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    pop_eval = evaluate_candidate_model(pop, val_df, cutoffs=cutoffs, exclude_seen=exclude_seen)
    t_eval = time.perf_counter() - t0
    pop_eval["train_time"] = t_train
    pop_eval["eval_time"] = t_eval
    pop_eval["config_str"] = "non-personalized"
    final_comparison.append(pop_eval)
    print(f"      Popularity: NDCG@10 = {pop_eval['metrics']['ndcg@10']:.4f} | Recall@10 = {pop_eval['metrics']['recall@10']:.4f}")

    # --- 2. ItemKNN Baseline ---
    print("\n[3/6] Evaluating ItemKNN baseline (k_neighbors=40)...")
    knn = ItemKNNRecommender(
        name="ItemKNN",
        k_neighbors=config["models"]["item_knn"]["k_neighbors"],
        shrinkage=config["models"]["item_knn"].get("shrinkage", 0.0),
    )
    t0 = time.perf_counter()
    knn.fit(train_df, num_users=num_users, num_items=num_items)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    knn_eval = evaluate_candidate_model(knn, val_df, cutoffs=cutoffs, exclude_seen=exclude_seen)
    t_eval = time.perf_counter() - t0
    knn_eval["train_time"] = t_train
    knn_eval["eval_time"] = t_eval
    knn_eval["config_str"] = f"k_neighbors={knn.k_neighbors}"
    final_comparison.append(knn_eval)
    print(f"      ItemKNN: NDCG@10 = {knn_eval['metrics']['ndcg@10']:.4f} | Recall@10 = {knn_eval['metrics']['recall@10']:.4f}")

    # --- 3. Implicit ALS Tuning ---
    print("\n[4/6] Tuning Implicit ALS across validation grid...")
    als_grid = config["models"]["als"]["grid"]
    best_als, best_als_params, best_als_metric, als_results = tune_grid(
        model_class=ALSRecommender,
        param_grid=als_grid,
        train_df=train_df,
        val_df=val_df,
        num_users=num_users,
        num_items=num_items,
        target_metric="ndcg@10",
        cutoffs=cutoffs,
        exclude_seen=exclude_seen,
        base_name="ALS",
    )
    print("      ALS Grid Search Results:")
    for res in als_results:
        p_str = ", ".join(f"{k}={v}" for k, v in res["params"].items())
        print(f"        - [{p_str}] -> NDCG@10: {res['metrics']['ndcg@10']:.4f}, Recall@10: {res['metrics']['recall@10']:.4f} (fit: {res['train_time']:.2f}s)")

    best_als_eval = evaluate_candidate_model(best_als, val_df, cutoffs=cutoffs, exclude_seen=exclude_seen)
    best_als_eval["train_time"] = [r["train_time"] for r in als_results if r["params"] == best_als_params][0]
    best_als_eval["eval_time"] = [r["eval_time"] for r in als_results if r["params"] == best_als_params][0]
    best_als_eval["config_str"] = ", ".join(f"{k}={v}" for k, v in best_als_params.items())
    final_comparison.append(best_als_eval)

    # --- 4. BPR-MF Tuning ---
    print("\n[5/6] Tuning BPR-MF across validation grid...")
    bpr_grid = config["models"]["bpr"]["grid"]
    best_bpr, best_bpr_params, best_bpr_metric, bpr_results = tune_grid(
        model_class=BPRRecommender,
        param_grid=bpr_grid,
        train_df=train_df,
        val_df=val_df,
        num_users=num_users,
        num_items=num_items,
        target_metric="ndcg@10",
        cutoffs=cutoffs,
        exclude_seen=exclude_seen,
        base_name="BPR",
    )
    print("      BPR Grid Search Results:")
    for res in bpr_results:
        p_str = ", ".join(f"{k}={v}" for k, v in res["params"].items())
        print(f"        - [{p_str}] -> NDCG@10: {res['metrics']['ndcg@10']:.4f}, Recall@10: {res['metrics']['recall@10']:.4f} (fit: {res['train_time']:.2f}s)")

    best_bpr_eval = evaluate_candidate_model(best_bpr, val_df, cutoffs=cutoffs, exclude_seen=exclude_seen)
    best_bpr_eval["train_time"] = [r["train_time"] for r in bpr_results if r["params"] == best_bpr_params][0]
    best_bpr_eval["eval_time"] = [r["eval_time"] for r in bpr_results if r["params"] == best_bpr_params][0]
    best_bpr_eval["config_str"] = ", ".join(f"{k}={v}" for k, v in best_bpr_params.items())
    final_comparison.append(best_bpr_eval)

    # --- 5. EASE Tuning ---
    print("\n[6/6] Tuning EASE across regularization grid...")
    ease_grid = config["models"]["ease"]["grid"]
    best_ease, best_ease_params, best_ease_metric, ease_results = tune_grid(
        model_class=EASERecommender,
        param_grid=ease_grid,
        train_df=train_df,
        val_df=val_df,
        num_users=num_users,
        num_items=num_items,
        target_metric="ndcg@10",
        cutoffs=cutoffs,
        exclude_seen=exclude_seen,
        base_name="EASE",
    )
    print("      EASE Grid Search Results:")
    for res in ease_results:
        p_str = ", ".join(f"{k}={v}" for k, v in res["params"].items())
        print(f"        - [{p_str}] -> NDCG@10: {res['metrics']['ndcg@10']:.4f}, Recall@10: {res['metrics']['recall@10']:.4f} (fit: {res['train_time']:.2f}s)")

    best_ease_eval = evaluate_candidate_model(best_ease, val_df, cutoffs=cutoffs, exclude_seen=exclude_seen)
    best_ease_eval["train_time"] = [r["train_time"] for r in ease_results if r["params"] == best_ease_params][0]
    best_ease_eval["eval_time"] = [r["eval_time"] for r in ease_results if r["params"] == best_ease_params][0]
    best_ease_eval["config_str"] = ", ".join(f"{k}={v}" for k, v in best_ease_params.items())
    final_comparison.append(best_ease_eval)

    # --- Final Five-Model Comparison Table ---
    print("\n==========================================================================================================")
    print("FINAL FIVE-MODEL VALIDATION COMPARISON (Full Catalog Ranking, 5,805 Evaluated Users)")
    print("==========================================================================================================")
    header = (
        f"{'Model':<12} | {'Selected Configuration':<24} | "
        f"{'NDCG@10':<8} {'Recall@10':<10} {'HR@10':<8} {'MRR@10':<8} | "
        f"{'NDCG@20':<8} {'Recall@20':<10} {'HR@20':<8} {'MRR@20':<8} | "
        f"{'Train(s)':<8} {'Eval(s)':<8}"
    )
    print(header)
    print("-" * len(header))

    for r in final_comparison:
        m = r["metrics"]
        row = (
            f"{r['model_name']:<12} | {r['config_str']:<24} | "
            f"{m['ndcg@10']:<8.4f} {m['recall@10']:<10.4f} {m['hr@10']:<8.4f} {m['mrr@10']:<8.4f} | "
            f"{m['ndcg@20']:<8.4f} {m['recall@20']:<10.4f} {m['hr@20']:<8.4f} {m['mrr@20']:<8.4f} | "
            f"{r['train_time']:<8.3f} {r['eval_time']:<8.3f}"
        )
        print(row)
    print("==========================================================================================================")

    # --- Best Model Selection ---
    best_overall = max(final_comparison, key=lambda x: x["metrics"]["ndcg@10"])
    best_name = best_overall["model_name"]
    best_score = best_overall["metrics"]["ndcg@10"]
    print(f"\n[SELECTION] Best candidate generator selected by validation NDCG@10: {best_name} ({best_score:.4f})")
    print(f"            Configuration: {best_overall['config_str']}")

    # Map name to model instance
    model_map = {
        "Popularity": pop,
        "ItemKNN": knn,
        "ALS": best_als,
        "BPR": best_bpr,
        "EASE": best_ease,
    }
    selected_model = model_map[best_name]

    # --- Top-100 Candidate Generation & Export ---
    out_dir = Path(config["paths"]["candidates_output_dir"])
    out_file = out_dir / "top100_candidates.parquet"
    print(f"\n[CANDIDATE EXPORT] Generating Top-{top_k_candidates} candidates for all {num_users:,} users with {selected_model.name}...")

    t0 = time.perf_counter()
    candidates_df = generate_and_save_candidate_pool(
        model=selected_model,
        all_users=all_users,
        k=top_k_candidates,
        output_path=out_file,
        exclude_seen=exclude_seen,
    )
    export_time = time.perf_counter() - t0
    print(f"                  Exported {len(candidates_df):,} candidate rows to {out_file.resolve()} ({export_time:.2f}s).")

    # --- Recall@100 Coverage Evaluation ---
    print("\n[COVERAGE EVALUATION] Evaluating candidate pool Recall@100 on validation positives...")
    recall_stats = evaluate_candidate_pool_recall(
        candidates_df=candidates_df,
        val_df=val_df,
        train_df=train_df,
        k=top_k_candidates,
    )
    print(f"      Validation Recall@{top_k_candidates}           : {recall_stats[f'recall@{top_k_candidates}']:.4f}")
    print(f"      Evaluated Validation Users       : {recall_stats['eval_users']:,} (with >= 1 positive)")
    print(f"      Total Users in Candidate Pool    : {recall_stats['total_users_in_pool']:,}")
    print(f"      Average Candidates per User      : {recall_stats['avg_candidates_per_user']:.1f}")
    print(f"      Seen Positive Train Item Leakage : {recall_stats['seen_train_leakage_count']} (must be 0)")

    if recall_stats["seen_train_leakage_count"] == 0:
        print("      [PASSED] Strict isolation verified: Zero seen positive training items in candidate pool.")
    else:
        print("      [FAILED] Seen positive training items leaked into candidate pool!")
        sys.exit(1)

    print("\n[SUCCESS] Phase 2B completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2B models, tuning, and candidate export.")
    parser.add_argument("--config", type=str, default="configs/candidates.yaml", help="Path to YAML configuration file.")
    args = parser.parse_args()
    run_phase2b(args.config)
