"""
ChurnSignal — Production REST API Service (FastAPI)
===================================================
Provides low-latency scoring, SHAP explainability, retention playbooks,
and counterfactual simulation for the ChurnSignal SaaS Retention Platform.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure churnrate root in path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.service.schemas import (
    CustomerInput,
    PredictionResponse,
    SimulationRequest,
    SimulationResponse,
    CustomerSummary,
    OverviewMetricsResponse,
    RetentionPlaybookAction
)
from src.data.loader import load_prediction_eligible_data
from src.features.engineer import engineer_all_features
from src.explainability.explainer import build_tree_explainer, explain_single_customer
from src.explainability.retention import (
    calculate_customer_health_score,
    assign_risk_tier,
    calculate_expected_mrr_at_risk,
    generate_retention_playbook,
    simulate_what_if_counterfactual
)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_production_artifacts()
    yield

# Initialize FastAPI App
app = FastAPI(
    title="ChurnSignal Retention Intelligence API",
    description="Machine Learning Churn Prediction, SHAP Explainability & Customer Retention Decision Engine",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global artifacts storage
STATE: Dict[str, Any] = {}


def prepare_raw_feature_dict(cust: CustomerInput) -> Dict[str, Any]:
    """Expands CustomerInput into full feature schema expected by the pipeline."""
    row = cust.model_dump()
    
    # Defaults for derived metrics
    sessions = row["sessions"]
    active_seats = row["active_seats"]
    usage = row["feature_usage_score"]
    nps = row["nps_score"]
    failures = row["payment_failures"]
    tickets = row["support_tickets"]
    tenure = row["tenure_month"]
    
    delta_s = row.get("delta_sessions") or 0.0
    delta_u = row.get("delta_usage") or 0.0
    
    # Ratios and utilization
    row["session_ratio"] = float((sessions + 1.0) / (max(0, sessions - delta_s) + 1.0))
    row["usage_ratio"] = float((usage + 1.0) / (max(0, usage - delta_u) + 1.0))
    row["seat_utilization"] = float(sessions / (active_seats + 1.0))
    
    # Rolling aggregations fallback
    if row.get("rolling_mean_sessions_3m") is None:
        row["rolling_mean_sessions_3m"] = float(sessions)
    if row.get("rolling_mean_usage_3m") is None:
        row["rolling_mean_usage_3m"] = float(usage)
    if row.get("rolling_max_payment_failures_3m") is None:
        row["rolling_max_payment_failures_3m"] = float(failures)
    if row.get("rolling_sum_support_tickets_3m") is None:
        row["rolling_sum_support_tickets_3m"] = float(tickets * 2)
        
    # Domain binary flags
    row["is_zero_seats"] = int(active_seats == 0)
    row["is_nps_detractor"] = int(nps <= 6)
    row["has_payment_failure"] = int(failures > 0)
    row["is_new_customer"] = int(tenure == 0)
    
    return row


def load_production_artifacts():
    """Loads serialized pipeline, champion model, explainer, and pre-scores latest cohort."""
    print("Loading serialized artifacts from disk...")
    artifacts_dir = project_root / "artifacts"
    
    pipeline_path = artifacts_dir / "feature_pipeline.joblib"
    model_path = artifacts_dir / "champion_model.joblib"
    
    if not pipeline_path.exists() or not model_path.exists():
        raise RuntimeError("Model or pipeline artifacts not found in artifacts/ directory!")
        
    pipeline = joblib.load(pipeline_path)
    champion_model = joblib.load(model_path)
    explainer = build_tree_explainer(champion_model)
    
    STATE["pipeline"] = pipeline
    STATE["model"] = champion_model
    STATE["explainer"] = explainer
    STATE["feature_names"] = pipeline.get_feature_names()
    
    print("Pre-scoring active customer cohort for dashboard caching...")
    try:
        cached_cohort_path = artifacts_dir / "scored_active_cohort.parquet"
        if cached_cohort_path.exists():
            print(f"Loading pre-scored cohort from {cached_cohort_path}...")
            cohort_df = pd.read_parquet(cached_cohort_path)
            X_cohort = pipeline.transform(cohort_df)
        else:
            raw_eligible = load_prediction_eligible_data()
            featured_df = engineer_all_features(raw_eligible)
            
            # Filter to the latest active observation month (2025-11-01)
            latest_month = featured_df["month"].max()
            cohort_df = featured_df[featured_df["month"] == latest_month].copy().reset_index(drop=True)
            
            X_cohort = pipeline.transform(cohort_df)
            cohort_probs = champion_model.predict_proba(X_cohort)[:, 1]
            cohort_health = calculate_customer_health_score(cohort_df).to_numpy()
            
            cohort_df["churn_probability"] = cohort_probs
            cohort_df["health_score"] = cohort_health
            cohort_df["risk_tier"] = [assign_risk_tier(p) for p in cohort_probs]
            cohort_df["expected_mrr_at_risk"] = [
                calculate_expected_mrr_at_risk(mrr, p) for mrr, p in zip(cohort_df["mrr"], cohort_probs)
            ]
            cohort_df.to_parquet(cached_cohort_path, index=False)
        
        STATE["cohort_df"] = cohort_df
        STATE["X_cohort"] = X_cohort
        print(f"Cohort loaded: {len(cohort_df)} active accounts ready.")
    except Exception as e:
        print(f"Warning: Could not pre-cache cohort: {e}")
        STATE["cohort_df"] = pd.DataFrame()


@app.get("/health", tags=["Health"])
def health_check():
    """Health probe verifying service and model readiness."""
    is_ready = "model" in STATE and "pipeline" in STATE
    return {
        "status": "healthy" if is_ready else "uninitialized",
        "model_loaded": "model" in STATE,
        "cohort_accounts_cached": len(STATE.get("cohort_df", []))
    }


@app.get("/api/overview", response_model=OverviewMetricsResponse, tags=["Dashboard"])
def get_overview_metrics():
    """Returns top-level KPI metrics for the customer retention dashboard."""
    df = STATE.get("cohort_df")
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="Customer cohort not ready.")
        
    total_active = len(df)
    total_mrr = float(df["mrr"].sum())
    total_expected_risk = float(df["expected_mrr_at_risk"].sum())
    avg_churn_prob = float(df["churn_probability"].mean())
    avg_health = float(df["health_score"].mean())
    
    tier_counts = df["risk_tier"].value_counts().to_dict()
    
    top_risk_df = df.sort_values(by="expected_mrr_at_risk", ascending=False).head(5)
    top_summaries = [
        CustomerSummary(
            user_id=row["user_id"],
            plan_type=row["plan_type"],
            mrr=round(row["mrr"], 2),
            tenure_month=int(row["tenure_month"]),
            churn_probability=round(row["churn_probability"], 4),
            risk_tier=row["risk_tier"],
            health_score=round(row["health_score"], 1),
            expected_mrr_at_risk=round(row["expected_mrr_at_risk"], 2)
        )
        for _, row in top_risk_df.iterrows()
    ]
    
    return OverviewMetricsResponse(
        total_active_accounts=total_active,
        total_mrr=round(total_mrr, 2),
        total_expected_mrr_at_risk=round(total_expected_risk, 2),
        average_churn_probability=round(avg_churn_prob, 4),
        average_health_score=round(avg_health, 1),
        critical_risk_accounts=tier_counts.get("Critical", 0),
        high_risk_accounts=tier_counts.get("High", 0),
        medium_risk_accounts=tier_counts.get("Medium", 0),
        low_risk_accounts=tier_counts.get("Low", 0),
        top_revenue_at_risk=top_summaries
    )


@app.get("/api/customers", response_model=List[CustomerSummary], tags=["Customers"])
def get_customers(
    plan_type: Optional[str] = Query(None, description="Filter by plan: Starter, Pro, Enterprise"),
    risk_tier: Optional[str] = Query(None, description="Filter by tier: Critical, High, Medium, Low"),
    sort_by: str = Query("expected_mrr_at_risk", description="Sort by expected_mrr_at_risk, churn_probability, mrr, or health_score"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Retrieves paginated and filtered accounts ranked by risk or financial exposure."""
    df = STATE.get("cohort_df")
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="Customer cohort not ready.")
        
    filtered = df.copy()
    if isinstance(plan_type, str) and plan_type:
        filtered = filtered[filtered["plan_type"].str.lower() == plan_type.lower()]
    if isinstance(risk_tier, str) and risk_tier:
        filtered = filtered[filtered["risk_tier"].str.lower() == risk_tier.lower()]
        
    sort_col = sort_by if isinstance(sort_by, str) else "expected_mrr_at_risk"
    if sort_col in filtered.columns:
        filtered = filtered.sort_values(by=sort_col, ascending=False)
        
    n_limit = limit if isinstance(limit, int) else 50
    n_offset = offset if isinstance(offset, int) else 0
    paged = filtered.iloc[n_offset:n_offset+n_limit]
    
    return [
        CustomerSummary(
            user_id=row["user_id"],
            plan_type=row["plan_type"],
            mrr=round(row["mrr"], 2),
            tenure_month=int(row["tenure_month"]),
            churn_probability=round(row["churn_probability"], 4),
            risk_tier=row["risk_tier"],
            health_score=round(row["health_score"], 1),
            expected_mrr_at_risk=round(row["expected_mrr_at_risk"], 2)
        )
        for _, row in paged.iterrows()
    ]


@app.get("/api/customer/{user_id}", tags=["Customers"])
def get_customer_detail(user_id: str):
    """Retrieves detailed profile metrics and risk status for an account."""
    df = STATE.get("cohort_df")
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="Customer cohort not ready.")
        
    matches = df[df["user_id"] == user_id]
    if len(matches) == 0:
        raise HTTPException(status_code=404, detail=f"Customer {user_id} not found.")
        
    row = matches.iloc[0].to_dict()
    # Convert numpy types to python native types
    clean_dict = {}
    for k, v in row.items():
        if isinstance(v, (np.floating, float)):
            clean_dict[k] = round(float(v), 2)
        elif isinstance(v, (np.integer, int)):
            clean_dict[k] = int(v)
        elif isinstance(v, pd.Timestamp):
            clean_dict[k] = str(v.date())
        else:
            clean_dict[k] = v
            
    return clean_dict


@app.post("/api/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict_churn(customer: CustomerInput):
    """Generates real-time churn probability, SHAP factors, and retention playbooks."""
    model = STATE.get("model")
    pipeline = STATE.get("pipeline")
    explainer = STATE.get("explainer")
    feature_names = STATE.get("feature_names")
    
    if None in (model, pipeline, explainer, feature_names):
        raise HTTPException(status_code=503, detail="Service not ready.")
        
    # Prepare feature dictionary and dataframe
    feat_dict = prepare_raw_feature_dict(customer)
    df_single = pd.DataFrame([feat_dict])
    
    X_vec = pipeline.transform(df_single)
    prob = float(model.predict_proba(X_vec)[0, 1])
    health = float(calculate_customer_health_score(df_single).iloc[0])
    tier = assign_risk_tier(prob)
    expected_risk = calculate_expected_mrr_at_risk(customer.mrr, prob)
    
    # SHAP local attribution
    explanation = explain_single_customer(explainer, X_vec, feature_names, top_n=3)
    
    # Playbooks
    playbook_data = generate_retention_playbook(
        explanation["risk_drivers"],
        churn_prob=prob,
        mrr=customer.mrr,
        health_score=health
    )
    
    playbook_actions = [
        RetentionPlaybookAction(
            playbook=p["playbook"],
            urgency=p["urgency"],
            action=p["action"],
            channel=p["channel"]
        )
        for p in playbook_data["recommended_playbooks"]
    ]
    
    return PredictionResponse(
        user_id=customer.user_id or "UNKNOWN",
        churn_probability=round(prob, 4),
        churn_percentage=f"{prob * 100:.1f}%",
        risk_tier=tier,
        health_score=round(health, 1),
        expected_mrr_at_risk=round(expected_risk, 2),
        top_risk_drivers=explanation["risk_drivers"],
        top_protective_factors=explanation["protective_factors"],
        recommended_playbooks=playbook_actions
    )


@app.get("/api/explain/{user_id}", tags=["Explainability"])
def explain_customer(user_id: str):
    """Returns local SHAP attribution and factor diagnosis for an active account."""
    df = STATE.get("cohort_df")
    X_cohort = STATE.get("X_cohort")
    explainer = STATE.get("explainer")
    feature_names = STATE.get("feature_names")
    
    if None in (df, X_cohort, explainer, feature_names) or len(df) == 0:
        raise HTTPException(status_code=503, detail="Cohort data or explainer not initialized.")
        
    matches = df[df["user_id"] == user_id]
    if len(matches) == 0:
        raise HTTPException(status_code=404, detail=f"Customer {user_id} not found.")
        
    idx = matches.index[0]
    explanation = explain_single_customer(explainer, X_cohort[idx], feature_names, top_n=5)
    row = matches.iloc[0]
    
    return {
        "user_id": user_id,
        "churn_probability": round(float(row["churn_probability"]), 4),
        "health_score": round(float(row["health_score"]), 1),
        "risk_tier": row["risk_tier"],
        "base_value_log_odds": round(explanation["base_value"], 4),
        "raw_margin": round(explanation["raw_margin"], 4),
        "top_risk_drivers": explanation["risk_drivers"],
        "top_protective_factors": explanation["protective_factors"]
    }


@app.post("/api/simulate", response_model=SimulationResponse, tags=["Simulation"])
def simulate_counterfactual(payload: SimulationRequest):
    """Simulates the impact of hypothetical feature changes on churn probability."""
    model = STATE.get("model")
    pipeline = STATE.get("pipeline")
    
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model service uninitialized.")
        
    raw_dict = prepare_raw_feature_dict(payload.customer)
    raw_series = pd.Series(raw_dict)
    
    sim_result = simulate_what_if_counterfactual(
        model=model,
        pipeline=pipeline,
        customer_raw_row=raw_series,
        modifications=payload.modifications
    )
    
    return SimulationResponse(
        user_id=payload.customer.user_id or "SIMULATED",
        base_churn_probability=sim_result["base_churn_probability"],
        simulated_churn_probability=sim_result["simulated_churn_probability"],
        probability_delta=sim_result["probability_delta"],
        base_health_score=sim_result["base_health_score"],
        simulated_health_score=sim_result["simulated_health_score"],
        health_score_delta=sim_result["health_score_delta"],
        modifications_applied=payload.modifications,
        causal_disclaimer=sim_result["causal_disclaimer"]
    )
