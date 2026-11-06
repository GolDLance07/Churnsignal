"""
Unit tests for ChurnSignal retention intelligence and simulation engine.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.explainability.retention import (
    calculate_customer_health_score,
    assign_risk_tier,
    assign_risk_category,
    get_retention_status,
    calculate_expected_mrr_at_risk,
    generate_retention_playbook,
    simulate_what_if_counterfactual,
    plot_retention_matrix
)


@pytest.fixture
def sample_customer_df():
    """Generates synthetic customer records with contrasting operational profiles."""
    return pd.DataFrame([
        {
            "user_id": "U_HEALTHY",
            "sessions": 60,
            "feature_usage_score": 95.0,
            "nps_score": 9,
            "support_tickets": 0,
            "payment_failures": 0,
            "product_incident": 0,
            "mrr": 2500.0,
            "plan_type": "Enterprise"
        },
        {
            "user_id": "U_DISTRESSED",
            "sessions": 2,
            "feature_usage_score": 8.0,
            "nps_score": 2,
            "support_tickets": 5,
            "payment_failures": 2,
            "product_incident": 1,
            "mrr": 50.0,
            "plan_type": "Starter"
        }
    ])


def test_customer_health_score(sample_customer_df):
    scores = calculate_customer_health_score(sample_customer_df)
    assert len(scores) == 2
    assert (scores >= 0).all() and (scores <= 100).all()
    # Healthy customer should score high (> 80)
    assert scores.iloc[0] > 80
    # Distressed customer should score low (< 30)
    assert scores.iloc[1] < 30


def test_assign_risk_tier():
    assert assign_risk_tier(0.85) == "Critical"
    assert assign_risk_tier(0.50) == "High"
    assert assign_risk_tier(0.25) == "Medium"
    assert assign_risk_tier(0.05) == "Low"


def test_assign_risk_category():
    assert assign_risk_category(0.85) == "High"
    assert assign_risk_category(0.35) == "High"
    assert assign_risk_category(0.25) == "Medium"
    assert assign_risk_category(0.15) == "Medium"
    assert assign_risk_category(0.05) == "Low"


def test_get_retention_status():
    status_urgent = get_retention_status(0.80, 25.0)
    assert "Urgent Action" in status_urgent
    
    status_stable = get_retention_status(0.05, 85.0)
    assert "Account Stable" in status_stable


def test_calculate_expected_mrr_at_risk():
    assert calculate_expected_mrr_at_risk(3000.0, 0.25) == 750.0
    assert calculate_expected_mrr_at_risk(25.0, 0.90) == 22.5
    assert calculate_expected_mrr_at_risk(0.0, 0.50) == 0.0


def test_generate_retention_playbook():
    drivers = [
        {"Feature": "payment_failures", "SHAP_Value": 0.45},
        {"Feature": "rolling_mean_usage_3m", "SHAP_Value": 0.35}
    ]
    playbook = generate_retention_playbook(drivers, churn_prob=0.65, mrr=1200.0, health_score=25.0)
    
    assert playbook["risk_tier"] == "Critical"
    assert playbook["expected_mrr_at_risk"] == 780.0
    
    playbook_names = [p["playbook"] for p in playbook["recommended_playbooks"]]
    assert "Billing & Dunning Safeguard" in playbook_names
    assert "Executive Adoption & Workflow Sprint" in playbook_names


def test_plot_retention_matrix(sample_customer_df, tmp_path):
    probs = np.array([0.05, 0.75])
    health_scores = np.array([92.0, 18.0])
    
    fig_path = plot_retention_matrix(
        sample_customer_df,
        probs,
        health_scores,
        filename="test_retention_matrix.png",
        sample_size=10
    )
    
    assert fig_path.exists()
    if fig_path.exists():
        fig_path.unlink()

