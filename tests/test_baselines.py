"""
Unit tests for baseline models (DummyClassifier and Logistic Regression).
"""

import pytest
import numpy as np
import pandas as pd
from src.data.loader import load_prediction_eligible_data
from src.features.engineer import engineer_all_features
from src.features.pipeline import chronological_split, ChurnFeaturePipeline
from src.models.baselines import (
    train_dummy_baseline,
    train_logistic_baseline,
    get_logistic_coefficients,
    save_model,
    load_model
)


@pytest.fixture(scope="module")
def prepared_data():
    """Fixture that prepares train and validation data once for baseline tests."""
    df = load_prediction_eligible_data()
    df_feat = engineer_all_features(df)
    train_df, val_df, test_df = chronological_split(df_feat)
    
    pipeline = ChurnFeaturePipeline()
    X_train = pipeline.fit_transform(train_df)
    X_val = pipeline.transform(val_df)
    
    y_train = train_df["churned_next_month"].values
    y_val = val_df["churned_next_month"].values
    
    return X_train, y_train, X_val, y_val, pipeline.get_feature_names()


def test_dummy_baseline_fitting_and_predictions(prepared_data):
    """Verify dummy classifier fits and predicts majority class (all 0s)."""
    X_train, y_train, X_val, y_val, _ = prepared_data
    dummy = train_dummy_baseline(X_train, y_train)
    preds = dummy.predict(X_val)
    
    assert (preds == 0).all(), "DummyClassifier predicted non-zero class!"
    proba = dummy.predict_proba(X_val)
    assert proba.shape == (len(X_val), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_regression_probabilities(prepared_data):
    """Verify Logistic Regression outputs valid calibrated probabilities in [0, 1]."""
    X_train, y_train, X_val, y_val, feature_names = prepared_data
    model = train_logistic_baseline(X_train, y_train, class_weight="balanced")
    
    proba = model.predict_proba(X_val)[:, 1]
    assert (proba >= 0.0).all() and (proba <= 1.0).all(), "Probabilities outside [0, 1]!"
    assert len(proba) == len(X_val)
    
    # Check coefficients
    coef_df = get_logistic_coefficients(model, feature_names)
    assert len(coef_df) == len(feature_names)
    assert "Coefficient" in coef_df.columns
    assert "Odds_Ratio" in coef_df.columns


def test_model_serialization_and_idempotence(prepared_data, tmp_path):
    """Verify that saving and loading a model preserves exact prediction probabilities."""
    X_train, y_train, X_val, _, _ = prepared_data
    model = train_logistic_baseline(X_train, y_train, class_weight="balanced")
    
    # Save to artifacts
    filename = "test_logistic_baseline.joblib"
    save_path = save_model(model, filename)
    assert save_path.exists()
    
    # Load and compare
    loaded_model = load_model(filename)
    orig_proba = model.predict_proba(X_val)
    loaded_proba = loaded_model.predict_proba(X_val)
    
    assert np.allclose(orig_proba, loaded_proba), "Loaded model outputs do not match original!"
    
    # Clean up test artifact
    save_path.unlink()

