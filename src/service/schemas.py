"""
ChurnSignal — Pydantic Schemas and Data Contracts for Production Serving
=======================================================================
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Raw input payload for a customer account at month T."""
    user_id: Optional[str] = Field("U_PREDICT", description="Unique customer identifier")
    plan_type: str = Field("Pro", description="Subscription tier: Starter, Pro, or Enterprise")
    mrr: float = Field(..., ge=0.0, description="Monthly Recurring Revenue in USD")
    sessions: int = Field(..., ge=0, description="Platform login sessions in past month")
    feature_usage_score: float = Field(..., ge=0.0, le=100.0, description="Product feature usage index (0-100)")
    support_tickets: int = Field(0, ge=0, description="Number of customer support tickets submitted")
    payment_failures: int = Field(0, ge=0, description="Number of billing/card payment failures")
    nps_score: int = Field(..., ge=0, le=10, description="Net Promoter Score rating (0-10)")
    product_incident: int = Field(0, ge=0, le=1, description="Binary flag: 1 if affected by platform incident, else 0")
    active_seats: int = Field(1, ge=0, description="Number of active user seats")
    tenure_month: int = Field(1, ge=0, description="Customer account tenure in months")
    
    # Optional velocity and rolling inputs (if not provided, default heuristics apply)
    delta_sessions: Optional[float] = Field(0.0, description="Month-over-month session delta")
    delta_usage: Optional[float] = Field(0.0, description="Month-over-month usage delta")
    delta_nps: Optional[float] = Field(0.0, description="Month-over-month NPS delta")
    delta_support_tickets: Optional[float] = Field(0.0, description="Month-over-month support ticket delta")
    delta_payment_failures: Optional[float] = Field(0.0, description="Month-over-month payment failure delta")
    delta_active_seats: Optional[float] = Field(0.0, description="Month-over-month active seats delta")
    rolling_mean_sessions_3m: Optional[float] = Field(None, description="3-month rolling mean sessions")
    rolling_mean_usage_3m: Optional[float] = Field(None, description="3-month rolling mean usage")
    rolling_max_payment_failures_3m: Optional[float] = Field(None, description="3-month rolling max payment failures")
    rolling_sum_support_tickets_3m: Optional[float] = Field(None, description="3-month rolling sum support tickets")


class FeatureAttribution(BaseModel):
    feature: str
    shap_value: float
    feature_input: float


class RetentionPlaybookAction(BaseModel):
    playbook: str
    urgency: str
    action: str
    channel: str


class PredictionResponse(BaseModel):
    """Production response contract for customer churn scoring."""
    user_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_percentage: str
    risk_tier: str = Field(..., description="Critical, High, Medium, or Low")
    health_score: float = Field(..., ge=0.0, le=100.0)
    expected_mrr_at_risk: float
    top_risk_drivers: List[Dict[str, Any]]
    top_protective_factors: List[Dict[str, Any]]
    recommended_playbooks: List[RetentionPlaybookAction]


class SimulationRequest(BaseModel):
    """Payload for counterfactual what-if simulation."""
    customer: CustomerInput
    modifications: Dict[str, Any] = Field(..., description="Key-value modifications to simulate (e.g. usage: 80)")


class SimulationResponse(BaseModel):
    user_id: str
    base_churn_probability: float
    simulated_churn_probability: float
    probability_delta: float
    base_health_score: float
    simulated_health_score: float
    health_score_delta: float
    modifications_applied: Dict[str, Any]
    causal_disclaimer: str


class CustomerSummary(BaseModel):
    user_id: str
    plan_type: str
    mrr: float
    tenure_month: int
    churn_probability: float
    risk_tier: str
    health_score: float
    expected_mrr_at_risk: float


class OverviewMetricsResponse(BaseModel):
    total_active_accounts: int
    total_mrr: float
    total_expected_mrr_at_risk: float
    average_churn_probability: float
    average_health_score: float
    critical_risk_accounts: int
    high_risk_accounts: int
    medium_risk_accounts: int
    low_risk_accounts: int
    top_revenue_at_risk: List[CustomerSummary]

