"""
ChurnSignal — Explainability & Factor Attribution Engine (SHAP)
==============================================================
Provides game-theoretic model interpretability for the Champion LightGBM classifier:
- TreeExplainer initialization and additivity validation
- Global SHAP feature importance across the customer population
- Local customer-level waterfall attributions (risk drivers vs. protective factors)
- Human-readable translation for Customer Success operational playbooks
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap


def get_figures_dir() -> Path:
    """Returns the reports/figures directory."""
    fig_dir = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def build_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Constructs a TreeExplainer for tree-based models (LightGBM/XGBoost)."""
    return shap.TreeExplainer(model)


def compute_global_feature_importance(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Computes global feature importance based on mean absolute SHAP values: E[|phi_j|].
    """
    shap_vals = explainer(X)
    mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    return importance_df


def explain_single_customer(
    explainer: shap.TreeExplainer,
    customer_vector: np.ndarray,
    feature_names: List[str],
    top_n: int = 3
) -> Dict[str, Any]:
    """
    Generates local feature attribution for an individual customer account.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        Fitted SHAP tree explainer.
    customer_vector : np.ndarray
        1D or 2D single customer feature vector.
    feature_names : List[str]
        List of 30 feature names.
    top_n : int
        Number of top drivers to return.

    Returns
    -------
    Dict[str, Any]
        base_value, total_shap_sum, top_risk_drivers, top_protective_factors
    """
    vec_2d = customer_vector.reshape(1, -1) if customer_vector.ndim == 1 else customer_vector
    shap_vals = explainer(vec_2d)
    
    base_val = float(explainer.expected_value)
    values = shap_vals.values[0]
    
    # Pair feature names with their local SHAP contribution
    attribs = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": values,
        "Feature_Input": vec_2d[0]
    })
    
    # Risk drivers (positive SHAP pushes probability up toward churn)
    risk_drivers = (
        attribs[attribs["SHAP_Value"] > 0]
        .sort_values(by="SHAP_Value", ascending=False)
        .head(top_n)
        .to_dict(orient="records")
    )
    
    # Protective factors (negative SHAP pulls probability down toward retention)
    protective_factors = (
        attribs[attribs["SHAP_Value"] < 0]
        .sort_values(by="SHAP_Value", ascending=True)
        .head(top_n)
        .to_dict(orient="records")
    )
    
    return {
        "base_value": base_val,
        "raw_margin": float(base_val + values.sum()),
        "risk_drivers": risk_drivers,
        "protective_factors": protective_factors
    }


def plot_shap_global_summary(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
    feature_names: List[str],
    filename: str = "13_shap_global_summary.png"
) -> Path:
    """Plots global SHAP bar chart ranking feature impact across the population."""
    importance_df = compute_global_feature_importance(explainer, X, feature_names).head(15)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.barplot(data=importance_df, y="Feature", x="Mean_Abs_SHAP", hue="Feature", palette="Blues_r", legend=False, ax=ax, edgecolor="black")
    
    ax.set_title("ChurnSignal — Global Feature Attribution (Mean |SHAP Value|)", fontweight="bold", fontsize=12, pad=12)
    ax.set_xlabel("Mean Absolute SHAP Value (Population-Level Impact)", fontweight="bold")
    ax.set_ylabel("Customer Metrics & Features", fontweight="bold")
    
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.4f}", (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha="left", va="center", xytext=(5, 0), textcoords="offset points", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_path = get_figures_dir() / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_customer_waterfall(
    explainer: shap.TreeExplainer,
    customer_vector: np.ndarray,
    feature_names: List[str],
    customer_id: str = "U0000042",
    filename: str = "14_customer_shap_waterfall.png"
) -> Path:
    """Plots a local waterfall-style bar attribution for a single customer."""
    explanation = explain_single_customer(explainer, customer_vector, feature_names, top_n=5)
    
    drivers = explanation["risk_drivers"] + explanation["protective_factors"]
    df_plot = pd.DataFrame(drivers).sort_values(by="SHAP_Value", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in df_plot["SHAP_Value"]]
    
    bars = ax.barh(df_plot["Feature"], df_plot["SHAP_Value"], color=colors, edgecolor="black", alpha=0.85, height=0.55)
    ax.axvline(0, color="black", linestyle="--", lw=1.2)
    
    ax.set_title(f"ChurnSignal — Local SHAP Diagnosis: Customer {customer_id}\n(Red = Pushes Churn Risk UP | Blue = Protects Against Churn)",
                 fontweight="bold", fontsize=11, pad=12)
    ax.set_xlabel("SHAP Impact on Log-Odds of Churn", fontweight="bold")
    
    for bar in bars:
        w = bar.get_width()
        ha = "left" if w > 0 else "right"
        offset = 5 if w > 0 else -5
        ax.annotate(f"{w:+.3f}", (w, bar.get_y() + bar.get_height() / 2.),
                    ha=ha, va="center", xytext=(offset, 0), textcoords="offset points", fontweight="bold", fontsize=9)

    plt.tight_layout()
    save_path = get_figures_dir() / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path
