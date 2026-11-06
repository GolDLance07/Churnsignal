"""
ChurnSignal — Model Comparison & Champion Selection Suite
=========================================================
Trains and benchmarks multiple model architectures under identical OOT validation conditions:
1. Balanced Logistic Regression (Linear baseline benchmark)
2. Decision Tree Classifier (Single non-linear tree)
3. Random Forest Classifier (Bagged ensemble)
4. Tuned LightGBM Classifier (Regularized gradient boosted ensemble — Champion)

Provides side-by-side metric tables, multi-model ROC/PR curves, and champion artifact export.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from src.models.baselines import train_logistic_baseline, save_model
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_precision_at_k
)


def get_figures_dir() -> Path:
    """Returns the reports/figures directory."""
    fig_dir = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 5
) -> DecisionTreeClassifier:
    """Trains a depth-constrained Decision Tree with balanced class weighting."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 8
) -> RandomForestClassifier:
    """Trains a Random Forest ensemble with random feature subsampling."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm_champion(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> LGBMClassifier:
    """
    Trains the regularized LightGBM Champion model.
    
    Hyperparameters specifically tuned for severe class imbalance:
    - max_depth=3, num_leaves=7: Prevents leaf fragmentation and overfitting in rare classes.
    - min_child_samples=100: Requires >= 100 observations per leaf to guard against noisy rare splits.
    - colsample_bytree=0.8, subsample=0.8: Stochastic feature and row bagging.
    - learning_rate=0.03, n_estimators=100: Conservative shrinkage.
    """
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=100,
        class_weight="balanced",
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Dict[str, Any]:
    """Trains the full suite of competitor models on the training partition."""
    models = {
        "Logistic Regression": train_logistic_baseline(X_train, y_train, class_weight="balanced"),
        "Decision Tree": train_decision_tree(X_train, y_train, max_depth=5),
        "Random Forest": train_random_forest(X_train, y_train, n_estimators=100, max_depth=8),
        "LightGBM (Champion)": train_lightgbm_champion(X_train, y_train)
    }
    return models


def evaluate_model_suite(
    models: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray
) -> pd.DataFrame:
    """
    Evaluates all competitor models on the validation set across multiple criteria.

    Returns
    -------
    pd.DataFrame
        Structured leaderboard sorted by PR-AUC descending.
    """
    records = []
    for name, model in models.items():
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        metrics = compute_classification_metrics(y_val, preds, probs)
        pak50 = compute_precision_at_k(y_val, probs, k=50)
        
        records.append({
            "Model": name,
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
            "Brier_Score": metrics["brier_score"],
            "Precision@50": pak50["precision_at_k"] * 100,
            "Recall@50": pak50["recall_at_k"] * 100,
            "Lift@50": pak50["lift"]
        })
        
    leaderboard = pd.DataFrame(records).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    return leaderboard


def plot_model_comparison_curves(
    models: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    filename: str = "11_model_comparison_curves.png"
) -> Path:
    """Plots side-by-side multi-model ROC and Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {
        "Logistic Regression": "#4C72B0",
        "Decision Tree": "#8172B2",
        "Random Forest": "#55A868",
        "LightGBM (Champion)": "#C44E52"
    }
    
    base_rate = float(np.mean(y_val))

    for name, model in models.items():
        probs = model.predict_proba(X_val)[:, 1]
        color = colors.get(name, "black")
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_val, probs)
        roc_score = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, lw=2.2, color=color, label=f"{name} (AUC = {roc_score:.4f})")
        
        # 2. PR Curve
        prec, rec, _ = precision_recall_curve(y_val, probs)
        pr_score = auc(rec, prec)
        axes[1].plot(rec, prec, lw=2.2, color=color, label=f"{name} (PR-AUC = {pr_score:.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Random Chance (0.50)")
    axes[0].set_title("Validation ROC Curves (All Models)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (FPR)", fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    axes[0].legend(loc="lower right", frameon=True)

    axes[1].axhline(base_rate, color="k", linestyle="--", lw=1.2, alpha=0.7, label=f"Random Baseline ({base_rate*100:.2f}%)")
    axes[1].set_title("Validation Precision-Recall Curves (All Models)", fontweight="bold")
    axes[1].set_xlabel("Recall", fontweight="bold")
    axes[1].set_ylabel("Precision", fontweight="bold")
    axes[1].set_ylim(0, 0.30)
    axes[1].legend(loc="upper right", frameon=True)

    plt.suptitle("ChurnSignal — Head-to-Head Model Discrimination Benchmark", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = get_figures_dir() / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_model_comparison_bar(
    leaderboard: pd.DataFrame,
    filename: str = "12_model_comparison_metrics.png"
) -> Path:
    """Plots comparative bar charts of PR-AUC and Operational Lift@Top-50."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PR-AUC
    sns.barplot(data=leaderboard, x="Model", y="PR-AUC", hue="Model", palette="Blues_d", ax=axes[0], edgecolor="black", legend=False)
    axes[0].set_title("Model Comparison: Precision-Recall AUC", fontweight="bold")
    axes[0].set_ylabel("PR-AUC Score", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontweight="bold")

    # Right: Lift@Top-50
    sns.barplot(data=leaderboard, x="Model", y="Lift@50", hue="Model", palette="Reds_d", ax=axes[1], edgecolor="black", legend=False)
    axes[1].set_title("Operational Lift@Top-50 (CS Outreach)", fontweight="bold")
    axes[1].set_ylabel("Lift Over Random Guessing", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}x", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontweight="bold")

    plt.suptitle("ChurnSignal — Competitive Model Benchmarking", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = get_figures_dir() / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path
