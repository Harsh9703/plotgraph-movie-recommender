"""Visualization helpers for PlotGraph Streamlit Demo (Altair, Matplotlib, Graphviz)."""

from typing import Any, Dict, List, Optional
import altair as alt
import numpy as np
import pandas as pd

from src.features.constants import PLUTCHIK_EMOTIONS


def plot_emotion_radar_or_bar(
    user_emotions: Dict[str, float],
    candidate_emotions: Dict[str, float],
    candidate_title: str,
) -> alt.Chart:
    """Create a side-by-side comparative bar chart for the 8 Plutchik emotions."""
    rows = []
    for emo in PLUTCHIK_EMOTIONS:
        rows.append({
            "Emotion": emo.capitalize(),
            "Profile": "Selected Movies Profile",
            "Score": round(float(user_emotions.get(emo, 0.0)), 3),
        })
        rows.append({
            "Emotion": emo.capitalize(),
            "Profile": f"Recommendation ({candidate_title[:20]}...)" if len(candidate_title) > 20 else f"Recommendation ({candidate_title})",
            "Score": round(float(candidate_emotions.get(emo, 0.0)), 3),
        })

    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Emotion:N", title="Plutchik Dimension", sort=None),
            y=alt.Y("Score:Q", title="Continuous Score [0, 1]", scale=alt.Scale(domain=[0, 1.0])),
            color=alt.Color(
                "Profile:N",
                scale=alt.Scale(range=["#1E88E5", "#FF8F00"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            xOffset="Profile:N",
            tooltip=["Emotion", "Profile", "Score"],
        )
        .properties(height=280)
    )
    return chart


def plot_validation_ablations_chart(ablation_data: Dict[str, Any]) -> alt.Chart:
    """Create an Altair bar chart showing held-out validation NDCG@10 across feature ablations."""
    rows = []
    for model_name, metrics in ablation_data.items():
        if isinstance(metrics, dict) and "ndcg_at_10" in metrics:
            ndcg = metrics["ndcg_at_10"]
            lift = metrics.get("relative_lift_pct", 0.0)
            is_best = (model_name == "FULL_NRC")
            is_baseline = (model_name == "EASE")
            category = "Selected Best (+9.2%)" if is_best else ("EASE Baseline" if is_baseline else "Ablation Variant")
            rows.append({
                "Configuration": model_name,
                "NDCG@10": round(ndcg, 4),
                "Relative Lift (%)": round(lift, 2),
                "Category": category,
            })

    df = pd.DataFrame(rows)
    # Sort by NDCG@10 ascending
    df = df.sort_values("NDCG@10")

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("Configuration:N", title=None, sort=None),
            x=alt.X("NDCG@10:Q", title="Validation NDCG@10", scale=alt.Scale(domain=[0.09, 0.106])),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(
                    domain=["EASE Baseline", "Ablation Variant", "Selected Best (+9.2%)"],
                    range=["#78909C", "#42A5F5", "#2E7D32"],
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["Configuration", "NDCG@10", "Relative Lift (%)"],
        )
        .properties(height=350)
    )
    return chart


def plot_final_test_comparison_chart() -> alt.Chart:
    """Create a grouped bar chart for final MovieLens-1M TEST results (EASE vs PlotGraph)."""
    rows = [
        {"Metric": "NDCG@10", "Model": "EASE Baseline", "Value": 0.0735, "Sig": "p = 0.322 (NS)"},
        {"Metric": "NDCG@10", "Model": "PlotGraph (FULL+NRC)", "Value": 0.0751, "Sig": "+2.14%"},
        {"Metric": "HR@10", "Model": "EASE Baseline", "Value": 0.3362, "Sig": "p = 0.002 (*)"},
        {"Metric": "HR@10", "Model": "PlotGraph (FULL+NRC)", "Value": 0.3559, "Sig": "+5.87% (p=0.002)"},
        {"Metric": "MRR@10", "Model": "EASE Baseline", "Value": 0.1268, "Sig": "p = 0.004 (*)"},
        {"Metric": "MRR@10", "Model": "PlotGraph (FULL+NRC)", "Value": 0.1355, "Sig": "+6.89% (p=0.004)"},
    ]
    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title=None, sort=["NDCG@10", "HR@10", "MRR@10"]),
            y=alt.Y("Value:Q", title="Test Score", scale=alt.Scale(domain=[0, 0.40])),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(range=["#90A4AE", "#1565C0"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            xOffset="Model:N",
            tooltip=["Metric", "Model", "Value", "Sig"],
        )
        .properties(height=280)
    )
    return chart
