"""
ChurnSignal — Retention Intelligence & Business Simulation Engine
================================================================
Translates statistical predictions and SHAP factor attributions into actionable
business decisions for Customer Success teams:
1. Customer Health Score (0–100 transparent operational vitality index)
2. Expected Revenue at Risk (MRR * P(churn))
3. Risk Tier assignment (Critical, High, Medium, Low)
4. Contextual Retention Playbooks mapped from local SHAP drivers
5. Counterfactual What-If Simulation with explicit causal safeguards
6. Strategic 2x2 Retention Matrix visualization (Health vs Risk)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def get_figures_dir() -> Path:
    """Returns the reports/figures directory."""
    fig_dir = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def calculate_customer_health_score(df: pd.DataFrame) -> pd.Series:
    """
    Computes a transparent, bounded 0–100 Customer Health Score based on 4 operational pillars:
    
    1. Engagement Pillar (35%):
       - Sessions and Feature Usage Score normalized to [0, 100].
    2. Sentiment & Satisfaction Pillar (25%):
       - NPS Score (0-10) scaled to [0, 100].
    3. Product & Service Friction Pillar (20%):
       - Deductions for open support tickets and product incidents.
    4. Billing Hygiene Pillar (20%):
       - Deductions for payment failures.
       
    Formula:
      Health = 0.35 * Engagement + 0.25 * NPS + 0.20 * Friction + 0.20 * Billing
      Clamped strictly to [0, 100].
    """
    # Pillar 1: Engagement (Benchmark: 50+ sessions is 100%, 100 usage score is 100%)
    session_comp = np.clip((df["sessions"] / 50.0) * 100.0, 0, 100.0)
    usage_comp = np.clip(df["feature_usage_score"], 0, 100.0)
    engagement_pillar = 0.5 * session_comp + 0.5 * usage_comp
    
    # Pillar 2: Sentiment (NPS is 0 to 10)
    sentiment_pillar = np.clip((df["nps_score"] / 10.0) * 100.0, 0, 100.0)
    
    # Pillar 3: Product Friction (Deductions: -15 per ticket, -30 per product incident)
    incidents = df["product_incident"] if "product_incident" in df.columns else 0
    friction_penalties = (df["support_tickets"] * 15.0) + (incidents * 30.0)
    friction_pillar = np.clip(100.0 - friction_penalties, 0, 100.0)
    
    # Pillar 4: Billing Hygiene (Deductions: -40 per payment failure)
    billing_penalties = df["payment_failures"] * 40.0
    billing_pillar = np.clip(100.0 - billing_penalties, 0, 100.0)
    
    # Weighted composite score
    composite = (
        0.35 * engagement_pillar +
        0.25 * sentiment_pillar +
        0.20 * friction_pillar +
        0.20 * billing_pillar
    )
    
    return composite.clip(0.0, 100.0).round(1)


def assign_risk_tier(churn_prob: float) -> str:
    """
    Categorizes churn probability into business-friendly operational tiers.
    """
    if churn_prob >= 0.60:
        return "Critical"
    elif churn_prob >= 0.35:
        return "High"
    elif churn_prob >= 0.15:
        return "Medium"
    else:
        return "Low"


def assign_risk_category(churn_prob: float) -> str:
    """
    Categorizes churn probability into 3 standard risk categories: Low, Medium, High.
    
    Decision Thresholds:
    - High Risk: >= 0.35 (35%+ estimated churn probability)
    - Medium Risk: 0.15 to 0.35 (15% to 35% estimated churn probability)
    - Low Risk: < 0.15 (under 15% estimated churn probability)
    """
    if churn_prob >= 0.35:
        return "High"
    elif churn_prob >= 0.15:
        return "Medium"
    else:
        return "Low"


def get_retention_status(churn_prob: float, health_score: float) -> str:
    """
    Provides a concise, business-oriented interpretation of customer churn risk and health.
    """
    if churn_prob >= 0.50 and health_score < 40:
        return "Urgent Action Required: High churn probability and severe operational disengagement."
    elif churn_prob >= 0.35 and health_score >= 60:
        return "Blindspot Risk: Active account experiencing sudden friction or billing distress."
    elif churn_prob >= 0.35:
        return "Elevated Risk: Proactive outreach recommended to address early churn indicators."
    elif churn_prob >= 0.15:
        return "Moderate Risk: Monitor engagement trends closely over the next billing cycle."
    elif health_score >= 70:
        return "Account Stable: Strong product engagement and healthy customer vitality."
    else:
        return "Low Risk: Passive or stable account requiring routine maintenance."


def calculate_expected_mrr_at_risk(mrr: float, churn_prob: float) -> float:
    """
    Computes financial risk exposure: Expected MRR Loss = MRR * P(churn).
    """
    return round(float(mrr * churn_prob), 2)


def generate_retention_playbook(
    risk_drivers: List[Dict[str, Any]],
    churn_prob: float,
    mrr: float,
    health_score: float
) -> Dict[str, Any]:
    """
    Synthesizes local SHAP risk drivers and financial tier into an actionable CS playbook.
    """
    tier = assign_risk_tier(churn_prob)
    expected_loss = calculate_expected_mrr_at_risk(mrr, churn_prob)
    
    actions = []
    driver_names = [d["Feature"] for d in risk_drivers]
    
    # Check for billing / dunning distress
    if any("payment" in f for f in driver_names):
        actions.append({
            "playbook": "Billing & Dunning Safeguard",
            "urgency": "Immediate (Within 24 Hours)",
            "action": "Trigger automated card update reminder; grant 7-day dunning grace period before service suspension.",
            "channel": "Finance / Automated Email"
        })
        
    # Check for technical friction or support ticket escalations
    if any("support" in f or "incident" in f for f in driver_names):
        actions.append({
            "playbook": "Engineering Priority Escalation",
            "urgency": "High (Within 48 Hours)",
            "action": "Assign dedicated Solutions Engineer to unblock open support tickets and conduct root-cause review.",
            "channel": "Technical CS / Engineering Call"
        })
        
    # Check for disengagement or usage drop
    if any("usage" in f or "session" in f for f in driver_names):
        actions.append({
            "playbook": "Executive Adoption & Workflow Sprint",
            "urgency": "Medium (Within 5 Business Days)",
            "action": "Schedule 1-on-1 CSM re-onboarding session; review feature adoption gaps with key account stakeholders.",
            "channel": "CSM Strategic Review Call"
        })
        
    # Check for sentiment detractor
    if any("nps" in f for f in driver_names) or health_score < 40:
        actions.append({
            "playbook": "Executive Sponsor Outreach",
            "urgency": "High",
            "action": "VP / Director of CS personal check-in to gather qualitative feedback and address strategic concerns.",
            "channel": "Leadership Video Call"
        })
        
    # Fallback default if no specific triggers
    if not actions:
        actions.append({
            "playbook": "Routine Relationship Health Check",
            "urgency": "Standard",
            "action": "Periodic value-delivery check-in and product roadmap preview.",
            "channel": "Email / Quarterly Review"
        })
        
    return {
        "risk_tier": tier,
        "expected_mrr_at_risk": expected_loss,
        "recommended_playbooks": actions
    }


def simulate_what_if_counterfactual(
    model: Any,
    pipeline: Any,
    customer_raw_row: pd.Series,
    modifications: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes a counterfactual simulation: assesses how modifying customer behavioral inputs
    alters the model's predicted churn probability.

    CAUTION: This simulation computes P(Y | X_modified). It demonstrates model responsiveness,
    NOT proven causal effect P(Y | do(X)). Artificially boosting usage metrics in software does
    not guarantee identical real-world churn prevention unless root value drivers are resolved.
    """
    # 1. Base prediction
    base_df = pd.DataFrame([customer_raw_row])
    X_base = pipeline.transform(base_df)
    base_prob = float(model.predict_proba(X_base)[0, 1])
    base_health = float(calculate_customer_health_score(base_df).iloc[0])
    
    # 2. Modified scenario
    modified_row = customer_raw_row.copy()
    for key, value in modifications.items():
        if key in modified_row:
            modified_row[key] = value
            
    sim_df = pd.DataFrame([modified_row])
    X_sim = pipeline.transform(sim_df)
    sim_prob = float(model.predict_proba(X_sim)[0, 1])
    sim_health = float(calculate_customer_health_score(sim_df).iloc[0])
    
    prob_delta = sim_prob - base_prob
    health_delta = sim_health - base_health
    
    return {
        "base_churn_probability": round(base_prob, 4),
        "simulated_churn_probability": round(sim_prob, 4),
        "probability_delta": round(prob_delta, 4),
        "base_health_score": base_health,
        "simulated_health_score": sim_health,
        "health_score_delta": round(health_delta, 1),
        "modifications_applied": modifications,
        "causal_disclaimer": (
            "Model simulation measures associational sensitivity P(Y|X), not proven causal intervention P(Y|do(X)). "
            "Real-world retention requires addressing the underlying customer satisfaction problem."
        )
    }


def plot_retention_matrix(
    df: pd.DataFrame,
    probs: np.ndarray,
    health_scores: np.ndarray,
    filename: str = "15_retention_matrix.png",
    sample_size: int = 1500
) -> Path:
    """
    Plots the strategic 2x2 Retention Matrix:
    - X-Axis: Customer Health Score (0-100)
    - Y-Axis: Predicted Churn Probability (0-100%)
    - Quadrants:
      1. Top-Left: Crisis Accounts (Low Health, High Churn Risk)
      2. Top-Right: Blindspot Churn (High Health, High Churn Risk)
      3. Bottom-Left: Passive / Zombie Accounts (Low Health, Low Churn Risk)
      4. Bottom-Right: Champion Accounts (High Health, Low Churn Risk)
    """
    plot_df = pd.DataFrame({
        "Health_Score": health_scores,
        "Churn_Probability": probs * 100.0,
        "MRR": df["mrr"].to_numpy(),
        "Plan_Type": df["plan_type"].to_numpy()
    })
    
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)
        
    fig, ax = plt.subplots(figsize=(11, 7.5))
    
    scatter = sns.scatterplot(
        data=plot_df,
        x="Health_Score",
        y="Churn_Probability",
        hue="Plan_Type",
        size="MRR",
        sizes=(30, 250),
        alpha=0.65,
        palette="tab10",
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )
    
    # Quadrant dividing lines
    ax.axvline(50, color="gray", linestyle="--", lw=1.5, alpha=0.7)
    ax.axhline(35, color="gray", linestyle="--", lw=1.5, alpha=0.7)
    
    # Quadrant annotations
    ax.text(10, 85, "CRISIS ZONE\n(Low Health, High Risk)\nImmediate Intervention",
            fontsize=10, fontweight="bold", color="#C44E52",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEB", ec="#C44E52", alpha=0.9))
            
    ax.text(60, 85, "BLINDSPOT CHURN\n(High Health, High Risk)\nBilling / Incident Blockers",
            fontsize=10, fontweight="bold", color="#CC79A7",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF0F5", ec="#CC79A7", alpha=0.9))
            
    ax.text(10, 8, "PASSIVE / ZOMBIE\n(Low Health, Low Risk)\nNeeds Value Re-Activation",
            fontsize=10, fontweight="bold", color="#D55E00",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3E6", ec="#D55E00", alpha=0.9))
            
    ax.text(60, 8, "CHAMPION ACCOUNTS\n(High Health, Low Risk)\nExpansion & Upsell Targets",
            fontsize=10, fontweight="bold", color="#009E73",
            bbox=dict(boxstyle="round,pad=0.4", fc="#E8F8F0", ec="#009E73", alpha=0.9))
            
    ax.set_title("ChurnSignal — Strategic 2x2 Retention & Health Matrix", fontweight="bold", fontsize=13, pad=12)
    ax.set_xlabel("Customer Health Score (0 = Severely Distressed, 100 = Flourishing)", fontweight="bold")
    ax.set_ylabel("Predicted 30-Day Churn Probability (%)", fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    
    plt.tight_layout()
    save_path = get_figures_dir() / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path

