"""
Unit tests for imbalance-aware evaluation metrics, threshold tuning, and evaluation plots.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
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


@pytest.fixture
def sample_predictions():
    """Generates synthetic ground truth and predicted probabilities for testing."""
    np.random.seed(42)
    # 1000 samples with ~5% positive class
    y_true = np.zeros(1000, dtype=int)
    y_true[:50] = 1
    np.random.shuffle(y_true)
    
    # Probabilities correlated with y_true
    y_prob = np.clip(0.05 + 0.5 * y_true + np.random.normal(0, 0.15, 1000), 0.0, 1.0)
    y_pred = (y_prob >= 0.3).astype(int)
    
    return y_true, y_pred, y_prob


def test_compute_classification_metrics(sample_predictions):
    """Verify dictionary of metrics is properly calculated and bounded."""
    y_true, y_pred, y_prob = sample_predictions
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
        
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == len(y_true)


def test_precision_at_k_calculation(sample_predictions):
    """Verify Precision@K ranking and lift logic."""
    y_true, _, y_prob = sample_predictions
    res_25 = compute_precision_at_k(y_true, y_prob, k=25)
    
    assert res_25["k"] == 25
    assert 0 <= res_25["tp_at_k"] <= 25
    assert res_25["precision_at_k"] == res_25["tp_at_k"] / 25
    assert res_25["lift"] >= 1.0  # Should be better than random


def test_threshold_curve_and_optimal_finder(sample_predictions):
    """Verify optimal threshold search returns valid bounded value."""
    y_true, _, y_prob = sample_predictions
    thresh_df = compute_threshold_curve(y_true, y_prob)
    
    assert len(thresh_df) > 0
    assert "threshold" in thresh_df.columns
    assert "f1" in thresh_df.columns
    
    opt_tau, best_f1 = find_optimal_threshold(y_true, y_prob, metric="f1")
    assert 0.0 < opt_tau < 1.0
    assert 0.0 < best_f1 <= 1.0


def test_evaluation_plot_generation(sample_predictions, tmp_path):
    """Verify evaluation plots export valid non-empty files."""
    y_true, y_pred, y_prob = sample_predictions
    
    p1 = plot_confusion_matrix(y_true, y_pred, filename="test_cm.png")
    p2 = plot_roc_and_pr_curves(y_true, y_prob, filename="test_curves.png")
    p3 = plot_threshold_tradeoffs(y_true, y_prob, optimal_tau=0.3, filename="test_thresholds.png")
    p4 = plot_precision_at_k(y_true, y_prob, k_list=[25, 50], filename="test_pak.png")
    
    for p in [p1, p2, p3, p4]:
        assert p.exists()
        assert p.stat().st_size > 1000
        p.unlink()  # Clean up test artifacts

