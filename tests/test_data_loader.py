"""
Unit tests for data ingestion, schema validation, and eligibility filtering.
"""

import pytest
import pandas as pd
from src.data.loader import load_raw_dataset, load_prediction_eligible_data


def test_raw_dataset_loading_and_shape():
    """Verify raw dataset loads with exact expected shape and columns."""
    df = load_raw_dataset()
    assert df.shape == (79817, 15), f"Expected (79817, 15), got {df.shape}"
    
    expected_cols = [
        "user_id", "month", "tenure_month", "plan_type", "monthly_price", "mrr",
        "sessions", "feature_usage_score", "support_tickets", "payment_failures",
        "churned", "churned_next_month", "nps_score", "product_incident", "active_seats"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"


def test_prediction_eligible_filtering():
    """Verify that only active customers before Dec 2025 are eligible for prediction."""
    eligible_df = load_prediction_eligible_data()
    
    # Check that no churned == 1 records exist in eligible pool
    assert (eligible_df["churned"] == 0).all(), "Found churned == 1 in prediction-eligible data!"
    
    # Check that Dec 2025 is excluded
    assert (eligible_df["month"] < "2025-12-01").all(), "Found month >= 2025-12 in prediction-eligible data!"
    
    # Verify expected count
    # Total raw: 79,817
    # Dec 2025 records: 3,931
    # Churned in months < Dec 2025: 1,131 - 62 = 1,069
    # Eligible = 79,817 - 3,931 - 1,069 = 74,817
    assert len(eligible_df) == 74817, f"Expected 74,817 eligible records, got {len(eligible_df)}"

