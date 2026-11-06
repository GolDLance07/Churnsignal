"""
Unit and integration tests for ChurnSignal production API service.
"""

import pytest
from pydantic import ValidationError
from src.service.app import (
    load_production_artifacts,
    health_check,
    get_overview_metrics,
    get_customers,
    get_customer_detail,
    predict_churn,
    simulate_counterfactual
)
from src.service.schemas import (
    CustomerInput,
    SimulationRequest
)


@pytest.fixture(scope="module", autouse=True)
def setup_artifacts():
    """Ensures models and cohort cache are loaded once for test suite."""
    load_production_artifacts()


def test_health_check():
    res = health_check()
    assert res["status"] == "healthy"
    assert res["model_loaded"] is True
    assert res["cohort_accounts_cached"] > 0


def test_overview_metrics():
    res = get_overview_metrics()
    assert res.total_active_accounts > 0
    assert res.total_mrr > 0
    assert res.total_expected_mrr_at_risk > 0
    assert 0.0 <= res.average_churn_probability <= 1.0
    assert 0.0 <= res.average_health_score <= 100.0
    assert len(res.top_revenue_at_risk) > 0


def test_get_customers_pagination_and_filter():
    # Test pagination
    res_page = get_customers(limit=10, offset=0)
    assert len(res_page) == 10
    
    # Test filtering by Enterprise plan
    res_plan = get_customers(plan_type="Enterprise", limit=5)
    assert len(res_plan) > 0
    for acct in res_plan:
        assert acct.plan_type == "Enterprise"


def test_get_customer_detail():
    # Get first customer from cohort
    customers = get_customers(limit=1)
    user_id = customers[0].user_id
    detail = get_customer_detail(user_id)
    assert detail["user_id"] == user_id
    assert "churn_probability" in detail
    assert "health_score" in detail


def test_predict_churn_valid():
    inp = CustomerInput(
        user_id="U_TEST_PRED",
        plan_type="Enterprise",
        mrr=3500.0,
        sessions=50,
        feature_usage_score=92.0,
        support_tickets=1,
        payment_failures=0,
        nps_score=9,
        product_incident=0,
        active_seats=10,
        tenure_month=15
    )
    pred = predict_churn(inp)
    assert pred.user_id == "U_TEST_PRED"
    assert 0.0 <= pred.churn_probability <= 1.0
    assert pred.risk_tier in ["Critical", "High", "Medium", "Low"]
    assert 0.0 <= pred.health_score <= 100.0
    assert len(pred.top_risk_drivers) > 0
    assert len(pred.recommended_playbooks) > 0


def test_predict_churn_validation_error():
    # Negative MRR or NPS > 10 should fail Pydantic validation
    with pytest.raises(ValidationError):
        CustomerInput(
            plan_type="Pro",
            mrr=-100.0,
            sessions=10,
            feature_usage_score=50.0,
            nps_score=15
        )


def test_simulation():
    req = SimulationRequest(
        customer=CustomerInput(
            user_id="U_SIM_TEST",
            plan_type="Pro",
            mrr=600.0,
            sessions=12,
            feature_usage_score=25.0,
            support_tickets=5,
            payment_failures=2,
            nps_score=3,
            product_incident=1,
            active_seats=3,
            tenure_month=6
        ),
        modifications={
            "payment_failures": 0,
            "support_tickets": 0,
            "feature_usage_score": 85.0
        }
    )
    res = simulate_counterfactual(req)
    assert res.user_id == "U_SIM_TEST"
    assert res.simulated_health_score > res.base_health_score
    assert "causal" in res.causal_disclaimer.lower()
