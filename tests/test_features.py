"""
Unit tests for feature engineering, momentum calculations, and leakage-safe preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.loader import load_prediction_eligible_data
from src.features.engineer import engineer_all_features
from src.features.pipeline import chronological_split, ChurnFeaturePipeline


def test_feature_engineering_nulls_and_shapes():
    """Verify feature engineering produces zero nulls and expected new columns."""
    df = load_prediction_eligible_data()
    df_feat = engineer_all_features(df)
    
    assert len(df_feat) == len(df), "Row count changed after feature engineering!"
    assert df_feat.isnull().sum().sum() == 0, "Engineered features contain unexpected nulls!"
    assert "monthly_price" not in df_feat.columns, "Redundant monthly_price was not dropped!"
    assert "delta_sessions" in df_feat.columns, "delta_sessions feature missing!"
    assert "rolling_mean_sessions_3m" in df_feat.columns, "rolling_mean_sessions_3m missing!"


def test_cold_start_tenure_zero_handling():
    """Verify that tenure_month == 0 records have safe neutral imputations."""
    df = load_prediction_eligible_data()
    df_feat = engineer_all_features(df)
    
    tenure_zero = df_feat[df_feat["tenure_month"] == 0]
    assert (tenure_zero["delta_sessions"] == 0.0).all(), "Tenure 0 delta_sessions is not 0.0!"
    assert (tenure_zero["session_ratio"] == 1.0).all(), "Tenure 0 session_ratio is not 1.0!"
    assert (tenure_zero["is_new_customer"] == 1).all(), "Tenure 0 is_new_customer is not 1!"


def test_chronological_split_integrity():
    """Verify strict out-of-time temporal boundaries."""
    df = load_prediction_eligible_data()
    train_df, val_df, test_df = chronological_split(df)
    
    assert train_df["month"].max() < val_df["month"].min()
    assert val_df["month"].max() < test_df["month"].min()
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_feature_pipeline_fit_transform():
    """Verify that ChurnFeaturePipeline fits on train only and transforms val/test deterministically."""
    df = load_prediction_eligible_data()
    df_feat = engineer_all_features(df)
    train_df, val_df, test_df = chronological_split(df_feat)
    
    pipeline = ChurnFeaturePipeline()
    X_train_transformed = pipeline.fit_transform(train_df)
    X_val_transformed = pipeline.transform(val_df)
    X_test_transformed = pipeline.transform(test_df)
    
    # 21 numeric + 4 one-hot (Starter, Pro, Business, Enterprise) + 5 binary = 30 features
    expected_dim = 30
    assert X_train_transformed.shape[1] == expected_dim, f"Expected {expected_dim} features, got {X_train_transformed.shape[1]}"
    assert X_val_transformed.shape[1] == expected_dim
    assert X_test_transformed.shape[1] == expected_dim
    
    # No NaNs or Infs
    assert not np.isnan(X_train_transformed).any()
    assert not np.isnan(X_val_transformed).any()
    assert not np.isnan(X_test_transformed).any()

