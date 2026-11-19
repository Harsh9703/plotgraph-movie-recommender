"""Script to train and evaluate Phase 2A baselines (Popularity and ItemKNN)."""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.candidate_gen.popularity import PopularityRecommender
from src.candidate_gen.item_knn import ItemKNNRecommender
from src.candidate_gen.evaluator import evaluate_candidate_model


def run_phase2a(config_path: str = "configs/candidates.yaml") -> None:
    print(f"=== Running Phase 2A Baselines with config: {config_path} ===")
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

    print("[1/4] Loading processed data splits...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    movie_map_df = pd.read_parquet(movie_map_path)
    user_map_df = pd.read_parquet(user_map_path)

    num_users = len(user_map_df)
    num_items = len(movie_map_df)
    cutoffs = config["evaluation"]["cutoffs"]
    exclude_seen = config["evaluation"]["exclude_seen"]

    train_pos = train_df[train_df["positive"] == 1]
    val_pos = val_df[val_df["positive"] == 1]

    print(f"      Universe: {num_users:,} users, {num_items:,} catalog movies")
    print(f"      TRAIN: {len(train_df):,} total ratings, {len(train_pos):,} positive interactions (r >= 4.0)")
    print(f"      VAL  : {len(val_df):,} total ratings, {len(val_pos):,} positive interactions (r >= 4.0)")
    print("      TEST split is strictly excluded from this evaluation.")

    results = []

    # --- 1. Popularity Baseline ---
    print("\n[2/4] Training & Evaluating Popularity baseline...")
    pop_model = PopularityRecommender(name=config["models"]["popularity"]["name"])

    t0 = time.perf_counter()
    pop_model.fit(train_df, num_users=num_users, num_items=num_items)
    t_train_pop = time.perf_counter() - t0

    t0 = time.perf_counter()
    pop_eval = evaluate_candidate_model(pop_model, val_df, cutoffs=cutoffs, exclude_seen=exclude_seen)
    t_eval_pop = time.perf_counter() - t0

    pop_eval["train_time"] = t_train_pop
    pop_eval["eval_time"] = t_eval_pop
    results.append(pop_eval)
    print(f"      Fitted in {t_train_pop:.3f}s | Evaluated in {t_eval_pop:.3f}s")

    # --- 2. ItemKNN Recommender ---
    k_nn = config["models"]["item_knn"]["k_neighbors"]
    shrinkage = config["models"]["item_knn"].get("shrinkage", 0.0)
    print(f"\n[3/4] Training & Evaluating ItemKNN (k_neighbors={k_nn}, shrinkage={shrinkage})...")
    knn_model = ItemKNNRecommender(
        name=config["models"]["item_knn"]["name"],
        k_neighbors=k_nn,
        shrinkage=shrinkage,
    )

    t0 = time.perf_counter()
    knn_model.fit(train_df, num_users=num_users, num_items=num_items)
    t_train_knn = time.perf_counter() - t0

    t0 = time.perf_counter()
    knn_eval = evaluate_candidate_model(knn_model, val_df, cutoffs=cutoffs, exclude_seen=exclude_seen)
    t_eval_knn = time.perf_counter() - t0

    knn_eval["train_time"] = t_train_knn
    knn_eval["eval_time"] = t_eval_knn
    results.append(knn_eval)
    print(f"      Fitted in {t_train_knn:.3f}s | Evaluated in {t_eval_knn:.3f}s")

    # --- 3. Results Summary Table ---
    print("\n[4/4] Validation Results Comparison (Full-Catalog Ranking, No Negative Sampling):")
    print(f"      Total validation users: {pop_eval['total_val_users']:,}")
    print(f"      Validation users with >=1 positive item: {pop_eval['eval_val_users']:,} ({pop_eval['eval_coverage_pct']:.2f}%)\n")

    header = (
        f"{'Model':<12} | "
        f"{'NDCG@10':<8} {'Recall@10':<10} {'HR@10':<8} {'MRR@10':<8} | "
        f"{'NDCG@20':<8} {'Recall@20':<10} {'HR@20':<8} {'MRR@20':<8} | "
        f"{'Train(s)':<8} {'Eval(s)':<8}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for r in results:
        m = r["metrics"]
        row = (
            f"{r['model_name']:<12} | "
            f"{m['ndcg@10']:<8.4f} {m['recall@10']:<10.4f} {m['hr@10']:<8.4f} {m['mrr@10']:<8.4f} | "
            f"{m['ndcg@20']:<8.4f} {m['recall@20']:<10.4f} {m['hr@20']:<8.4f} {m['mrr@20']:<8.4f} | "
            f"{r['train_time']:<8.3f} {r['eval_time']:<8.3f}"
        )
        print(row)
    print(separator)

    # Compute relative lifts of ItemKNN over Popularity
    p_m = results[0]["metrics"]
    k_m = results[1]["metrics"]
    ndcg10_lift = ((k_m["ndcg@10"] - p_m["ndcg@10"]) / p_m["ndcg@10"]) * 100.0
    recall10_lift = ((k_m["recall@10"] - p_m["recall@10"]) / p_m["recall@10"]) * 100.0
    print(f"\n[Relative Lift: ItemKNN vs Popularity] NDCG@10: +{ndcg10_lift:.2f}% | Recall@10: +{recall10_lift:.2f}%")
    print("[SUCCESS] Phase 2A baselines executed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2A recommendation baselines.")
    parser.add_argument("--config", type=str, default="configs/candidates.yaml", help="Path to YAML configuration file.")
    args = parser.parse_args()
    run_phase2a(args.config)
