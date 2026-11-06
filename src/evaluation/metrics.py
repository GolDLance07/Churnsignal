"""
ChurnSignal — Imbalance-Aware Evaluation Metrics
================================================
Comprehensive evaluation suite tailored for rare-event customer churn prediction:
- Standard metrics: Accuracy, Precision, Recall, F1
- Threshold-agnostic ranking metrics: ROC-AUC, PR-AUC (Average Precision)
- Probability calibration quality: Brier Score
- Operational business metrics: Precision@Top-K, Recall@Top-K, and Lift over random chance
- Decision threshold optimization curves
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes a comprehensive dictionary of binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_pred : np.ndarray
        Hard binary predictions (0 or 1) at a chosen decision threshold.
    y_prob : Optional[np.ndarray]
        Continuous predicted churn probabilities in [0, 1].

    Returns
    -------
    Dict[str, float]
        Dictionary of metric scores.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    
    # Calculate confusion matrix entries
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    return metrics


def compute_precision_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int = 50
) -> Dict[str, Any]:
    """
    Computes Precision@Top-K and Recall@Top-K.
    
    Ranks customers by predicted risk probability descending, flags the top K accounts,
    and calculates how many of them actually churned.

    Parameters
    ----------
    y_true : np.ndarray
        True binary churn labels.
    y_prob : np.ndarray
        Continuous predicted churn probabilities.
    k : int
        Number of top-ranked accounts to inspect (e.g. weekly CS outreach capacity).

    Returns
    -------
    Dict[str, Any]
        k, true_positives_at_k, precision_at_k, recall_at_k, lift_over_random
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if k > len(y_true):
        k = len(y_true)
        
    # Sort indices by probability descending
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    tp_at_k = int(y_true[top_k_indices].sum())
    
    precision_at_k = tp_at_k / k
    total_positives = int(y_true.sum())
    recall_at_k = tp_at_k / total_positives if total_positives > 0 else 0.0
    
    base_rate = float(y_true.mean())
    lift = precision_at_k / base_rate if base_rate > 0 else 1.0

    return {
        "k": int(k),
        "tp_at_k": tp_at_k,
        "precision_at_k": float(precision_at_k),
        "recall_at_k": float(recall_at_k),
        "base_rate": float(base_rate),
        "lift": float(lift)
    }


def compute_threshold_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Evaluates precision, recall, f1, and flagged count across candidate decision thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
        
    records = []
    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        flagged_count = int(y_pred.sum())
        
        records.append({
            "threshold": tau,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "flagged_accounts": flagged_count,
            "flagged_pct": float(flagged_count / len(y_true) * 100)
        })
        
    return pd.DataFrame(records)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float]:
    """
    Finds the probability threshold that maximizes a specified evaluation metric.

    Returns
    -------
    Tuple[float, float]
        (optimal_threshold, best_metric_score)
    """
    thresh_df = compute_threshold_curve(y_true, y_prob)
    best_row = thresh_df.loc[thresh_df[metric].idxmax()]
    return float(best_row["threshold"]), float(best_row[metric])

