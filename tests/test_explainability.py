"""
Unit tests for ChurnSignal explainability engine (SHAP).
"""

import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from src.explainability.explainer import (
    build_tree_explainer,
    compute_global_feature_importance,
    explain_single_customer,
    plot_shap_global_summary,
    plot_customer_waterfall,
)


@pytest.fixture
def trained_toy_lgb():
    """Trains a quick toy LightGBM model on synthetic data."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] * 2 + X[:, 1] - X[:, 2] > 0).astype(int)
    
    clf = lgb.LGBMClassifier(n_estimators=10, max_depth=3, random_state=42, verbose=-1)
    clf.fit(X, y)
    feature_names = [f"feat_{i}" for i in range(5)]
    return clf, X, feature_names


def test_tree_explainer_additivity(trained_toy_lgb):
    """Verifies that sum of SHAP values plus base value equals raw margin output."""
    clf, X, feature_names = trained_toy_lgb
    explainer = build_tree_explainer(clf)
    
    # Test on a single customer vector
    x_cust = X[0:1]
    shap_vals = explainer(x_cust)
    
    # Additivity property: sum(phi_i) + phi_0 == f(x)
    raw_margin = clf.predict(x_cust, raw_score=True)[0]
    reconstructed_margin = explainer.expected_value + shap_vals.values[0].sum()
    
    assert np.isclose(raw_margin, reconstructed_margin, atol=1e-5), (
        f"Additivity failed: raw {raw_margin} vs reconstructed {reconstructed_margin}"
    )


def test_global_feature_importance(trained_toy_lgb):
    clf, X, feature_names = trained_toy_lgb
    explainer = build_tree_explainer(clf)
    
    df = compute_global_feature_importance(explainer, X, feature_names)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Feature", "Mean_Abs_SHAP"]
    assert len(df) == 5
    # Verify sorted descending
    assert df["Mean_Abs_SHAP"].is_monotonic_decreasing


def test_explain_single_customer(trained_toy_lgb):
    clf, X, feature_names = trained_toy_lgb
    explainer = build_tree_explainer(clf)
    
    x_cust = X[0]
    explanation = explain_single_customer(explainer, x_cust, feature_names, top_n=2)
    
    assert "base_value" in explanation
    assert "raw_margin" in explanation
    assert "risk_drivers" in explanation
    assert "protective_factors" in explanation
    
    for item in explanation["risk_drivers"]:
        assert item["SHAP_Value"] > 0
    for item in explanation["protective_factors"]:
        assert item["SHAP_Value"] < 0


def test_plot_generation(trained_toy_lgb, tmp_path):
    clf, X, feature_names = trained_toy_lgb
    explainer = build_tree_explainer(clf)
    
    summary_path = plot_shap_global_summary(
        explainer, X, feature_names, filename="test_summary.png"
    )
    assert summary_path.exists()
    
    waterfall_path = plot_customer_waterfall(
        explainer, X[0], feature_names, customer_id="UTEST_01", filename="test_waterfall.png"
    )
    assert waterfall_path.exists()
    
    # Clean up test figures
    if summary_path.exists():
        summary_path.unlink()
    if waterfall_path.exists():
        waterfall_path.unlink()

