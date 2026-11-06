"""
Unit tests for model comparison suite (Decision Tree, Random Forest, LightGBM, and benchmark leaderboard).
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from src.data.loader import load_prediction_eligible_data
from src.features.engineer import engineer_all_features
from src.features.pipeline import chronological_split, ChurnFeaturePipeline
from src.models.comparison import (
    train_decision_tree,
    train_random_forest,
    train_lightgbm_champion,
    train_all_models,
    evaluate_model_suite,
    plot_model_comparison_curves,
    plot_model_comparison_bar
)


@pytest.fixture(scope="module")
def prepared_sample_data():
    """Generates a small partition of real data for fast comparison testing."""
    df = load_prediction_eligible_data()
    # Take a subsample for fast unit testing
    sample_df = df[df["month"] <= "2024-06-01"].head(2000)
    df_feat = engineer_all_features(sample_df)
    
    pipeline = ChurnFeaturePipeline()
    X = pipeline.fit_transform(df_feat)
    y = df_feat["churned_next_month"].values
    
    # Split 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Ensure at least one positive in val
    if y_val.sum() == 0:
        y_val[0] = 1
        
    return X_train, y_train, X_val, y_val


def test_individual_models_training(prepared_sample_data):
    """Verify each model trains and predicts probabilities in [0, 1]."""
    X_train, y_train, X_val, y_val = prepared_sample_data
    
    dt = train_decision_tree(X_train, y_train, max_depth=3)
    rf = train_random_forest(X_train, y_train, n_estimators=10, max_depth=3)
    lgb = train_lightgbm_champion(X_train, y_train)
    
    for m in [dt, rf, lgb]:
        probs = m.predict_proba(X_val)[:, 1]
        assert len(probs) == len(X_val)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_evaluate_model_suite_structure(prepared_sample_data):
    """Verify evaluation leaderboard contains all models and correct metrics."""
    X_train, y_train, X_val, y_val = prepared_sample_data
    models = train_all_models(X_train, y_train)
    
    leaderboard = evaluate_model_suite(models, X_val, y_val)
    
    assert len(leaderboard) == 4
    expected_cols = ["Model", "ROC-AUC", "PR-AUC", "Brier_Score", "Precision@50", "Recall@50", "Lift@50"]
    for col in expected_cols:
        assert col in leaderboard.columns
        
    # Check that leaderboard is sorted descending by PR-AUC
    assert leaderboard["PR-AUC"].is_monotonic_decreasing


def test_comparison_plot_generation(prepared_sample_data, tmp_path):
    """Verify multi-model comparison curves export valid images."""
    X_train, y_train, X_val, y_val = prepared_sample_data
    models = train_all_models(X_train, y_train)
    leaderboard = evaluate_model_suite(models, X_val, y_val)
    
    p1 = plot_model_comparison_curves(models, X_val, y_val, filename="test_comp_curves.png")
    p2 = plot_model_comparison_bar(leaderboard, filename="test_comp_bar.png")
    
    for p in [p1, p2]:
        assert p.exists()
        assert p.stat().st_size > 1000
        p.unlink()

