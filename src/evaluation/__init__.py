"""Evaluation package for ChurnSignal."""

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_precision_at_k,
    compute_threshold_curve,
    find_optimal_threshold
)
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    plot_threshold_tradeoffs,
    plot_precision_at_k
)

__all__ = [
    "compute_classification_metrics",
    "compute_precision_at_k",
    "compute_threshold_curve",
    "find_optimal_threshold",
    "plot_confusion_matrix",
    "plot_roc_and_pr_curves",
    "plot_threshold_tradeoffs",
    "plot_precision_at_k"
]

