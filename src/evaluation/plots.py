"""
ChurnSignal — Evaluation Plotting Suite
======================================
Publication-grade visualizations for binary classification under class imbalance:
- Confusion Matrix with percentages and annotations
- Receiver Operating Characteristic (ROC) curve
- Precision-Recall (PR) curve with baseline reference
- Threshold Trade-off curves (Precision & Recall crossover)
- Precision@Top-K Lift curves
"""

from pathlib import Path
from typing import List, Optional
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from src.evaluation.metrics import compute_precision_at_k, compute_threshold_curve


def get_figures_dir() -> Path:
    """Returns the reports/figures directory."""
    fig_dir = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    filename: str = "07_confusion_matrix.png"
) -> Path:
    """Plots an annotated 2x2 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm / cm.sum() * 100

    labels = [["True Negatives (TN)", "False Positives (FP)"],
              ["False Negatives (FN)", "True Positives (TP)"]]

    annot = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{labels[i][j]}\n{cm[i, j]:,}\n({cm_pct[i, j]:.2f}%)"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Pred: Retain (0)", "Pred: Churn (1)"],
                yticklabels=["Actual: Retain (0)", "Actual: Churn (1)"], ax=ax)
    
    ax.set_title(title, fontweight="bold", fontsize=12, pad=12)
    plt.tight_layout()
    
    save_path = get_figures_dir() / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Balanced Logistic Regression",
    filename: str = "08_roc_and_pr_curves.png"
) -> Path:
    """Plots side-by-side ROC Curve and Precision-Recall Curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_score = auc(fpr, tpr)

    axes[0].plot(fpr, tpr, color="#2B5C8F", lw=2.5, label=f"{model_name} (AUC = {roc_score:.4f})")
    axes[0].plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (FPR)", fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
    axes[0].legend(loc="lower right", frameon=True)

    # 2. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_score = auc(rec, prec)
    base_rate = float(np.mean(y_true))

    axes[1].plot(rec, prec, color="#C44E52", lw=2.5, label=f"{model_name} (PR-AUC = {pr_score:.4f})")
    axes[1].axhline(base_rate, color="grey", lw=1.5, linestyle="--", label=f"Random Baseline ({base_rate*100:.2f}%)")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, max(max(prec) * 1.15, 0.25)])
    axes[1].set_xlabel("Recall (Fraction of True Churners Caught)", fontweight="bold")
    axes[1].set_ylabel("Precision (Accuracy of Churn Alerts)", fontweight="bold")
    axes[1].set_title("Precision-Recall (PR) Curve", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True)

    plt.suptitle(f"ChurnSignal — Discrimination Curves ({model_name})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = get_figures_dir() / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_threshold_tradeoffs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal_tau: float = 0.5,
    filename: str = "09_threshold_tradeoffs.png"
) -> Path:
    """Plots Precision and Recall as a function of the decision threshold."""
    thresh_df = compute_threshold_curve(y_true, y_prob)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    ax1.plot(thresh_df["threshold"], thresh_df["precision"], color="#4C72B0", lw=2.5, label="Precision")
    ax1.plot(thresh_df["threshold"], thresh_df["recall"], color="#C44E52", lw=2.5, label="Recall")
    ax1.plot(thresh_df["threshold"], thresh_df["f1"], color="#55A868", lw=2, linestyle="-.", label="F1 Score")
    
    ax1.axvline(optimal_tau, color="purple", linestyle="--", lw=1.8, label=f"Optimal F1 Threshold ($\\tau = {optimal_tau:.2f}$)")
    
    ax1.set_xlabel("Decision Threshold ($\\tau$)", fontweight="bold")
    ax1.set_ylabel("Score", fontweight="bold")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="center right", frameon=True)

    ax2 = ax1.twinx()
    ax2.plot(thresh_df["threshold"], thresh_df["flagged_accounts"], color="grey", lw=1.2, linestyle=":", alpha=0.6, label="Flagged Accounts")
    ax2.set_ylabel("Accounts Flagged for Outreach", color="grey", fontweight="bold")
    ax2.grid(False)

    plt.title("ChurnSignal — Decision Threshold Tuning: Precision vs. Recall Trade-Off", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    save_path = get_figures_dir() / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_precision_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_list: Optional[List[int]] = None,
    filename: str = "10_precision_at_k.png"
) -> Path:
    """Plots Precision@Top-K and Lift for business capacity planning."""
    if k_list is None:
        k_list = [25, 50, 100, 200, 500, 1000]

    results = [compute_precision_at_k(y_true, y_prob, k) for k in k_list]
    k_vals = [r["k"] for r in results]
    precs = [r["precision_at_k"] * 100 for r in results]
    lifts = [r["lift"] for r in results]
    base_rate = results[0]["base_rate"] * 100

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    bars = ax1.bar([str(k) for k in k_vals], precs, color="#4C72B0", width=0.45, edgecolor="black", alpha=0.85, label="Precision@K (%)")
    ax1.axhline(base_rate, color="red", linestyle="--", lw=1.8, label=f"Random Base Rate ({base_rate:.2f}%)")
    ax1.set_xlabel("Top K Accounts Flagged for CS Outreach", fontweight="bold")
    ax1.set_ylabel("Precision@K (% True Churners)", color="#4C72B0", fontweight="bold")
    ax1.set_ylim(0, max(precs) * 1.35)

    for bar, lift in zip(bars, lifts):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%\n({lift:.1f}x Lift)", ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax1.legend(loc="upper right", frameon=True)
    plt.title("ChurnSignal — Operational Precision@Top-K & Actionable Lift", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    save_path = get_figures_dir() / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path

