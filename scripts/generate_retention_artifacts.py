"""
ChurnSignal — Generate Retention Intelligence Artifacts & Figures (Milestone 8)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from src.data.loader import load_prediction_eligible_data
from src.features.engineer import engineer_all_features
from src.features.pipeline import chronological_split, TARGET_COLUMN
from src.explainability.explainer import build_tree_explainer, explain_single_customer
from src.explainability.retention import (
    calculate_customer_health_score,
    assign_risk_tier,
    calculate_expected_mrr_at_risk,
    generate_retention_playbook,
    simulate_what_if_counterfactual,
    plot_retention_matrix
)


def main():
    print("Loading data and pipeline artifacts...")
    eligible_df = load_prediction_eligible_data()
    featured_df = engineer_all_features(eligible_df)
    train_df, val_df, test_df = chronological_split(featured_df)
    
    pipeline = joblib.load("artifacts/feature_pipeline.joblib")
    champion_model = joblib.load("artifacts/champion_model.joblib")
    feature_names = pipeline.get_feature_names()
    
    X_val = pipeline.transform(val_df)
    y_val = val_df[TARGET_COLUMN].to_numpy()
    val_probs = champion_model.predict_proba(X_val)[:, 1]
    
    # 1. Health Scores & Financial Risk
    print("\nComputing Customer Health Scores and Financial Exposure...")
    health_scores = calculate_customer_health_score(val_df).to_numpy()
    
    val_df_analysis = val_df.copy()
    val_df_analysis["churn_probability"] = val_probs
    val_df_analysis["health_score"] = health_scores
    val_df_analysis["risk_tier"] = [assign_risk_tier(p) for p in val_probs]
    val_df_analysis["expected_mrr_at_risk"] = [
        calculate_expected_mrr_at_risk(mrr, p) for mrr, p in zip(val_df["mrr"], val_probs)
    ]
    
    print("\nRisk Tier Distribution (Validation Cohort):")
    print(val_df_analysis["risk_tier"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")
    
    print(f"\nTotal Validation MRR: ${val_df_analysis['mrr'].sum():,.2f}")
    print(f"Total Expected MRR at Risk: ${val_df_analysis['expected_mrr_at_risk'].sum():,.2f}")
    
    # Top 5 Accounts by Expected Revenue Exposure vs Raw Probability
    top_by_revenue = val_df_analysis.sort_values(by="expected_mrr_at_risk", ascending=False).head(5)
    print("\nTop 5 Accounts by Financial Risk (Expected MRR at Risk):")
    cols_to_show = ["user_id", "plan_type", "mrr", "churn_probability", "health_score", "expected_mrr_at_risk"]
    print(top_by_revenue[cols_to_show].to_string(index=False))
    
    # 2. Generate Strategic Retention Matrix Figure
    print("\nGenerating Figure 15: Strategic 2x2 Retention & Health Matrix...")
    fig15 = plot_retention_matrix(val_df, val_probs, health_scores, filename="15_retention_matrix.png")
    print(f"Saved: {fig15}")
    
    # 3. Retention Playbook for Top Financial Risk Account
    top_acct = top_by_revenue.iloc[0]
    print(f"\nGenerating Retention Playbook for Top Risk Account: {top_acct['user_id']} ({top_acct['plan_type']})...")
    explainer = build_tree_explainer(champion_model)
    cust_idx = val_df.index.get_loc(top_acct.name)
    explanation = explain_single_customer(explainer, X_val[cust_idx], feature_names, top_n=3)
    
    playbook = generate_retention_playbook(
        explanation["risk_drivers"],
        churn_prob=top_acct["churn_probability"],
        mrr=top_acct["mrr"],
        health_score=top_acct["health_score"]
    )
    print(f"Plan: {top_acct['plan_type']} | MRR: ${top_acct['mrr']:.2f}")
    print(f"Churn Probability: {top_acct['churn_probability']:.2%} | Risk Tier: {playbook['risk_tier']}")
    print(f"Expected MRR Loss: ${playbook['expected_mrr_at_risk']:.2f} | Health Score: {top_acct['health_score']:.1f}")
    print("Recommended Action Playbooks:")
    for a in playbook["recommended_playbooks"]:
        print(f"  [{a['urgency']}] {a['playbook']}: {a['action']}")
        
    # 4. What-If Counterfactual Simulation
    print(f"\nRunning What-If Simulation on Customer {top_acct['user_id']}...")
    target_row = val_df.loc[top_acct.name]
    
    # Simulate resolving support tickets and boosting usage
    modifications = {
        "support_tickets": 0,
        "payment_failures": 0,
        "feature_usage_score": min(100.0, float(target_row["feature_usage_score"]) + 30.0)
    }
    
    sim_result = simulate_what_if_counterfactual(
        champion_model,
        pipeline,
        target_row,
        modifications
    )
    
    print(f"Base Churn Probability: {sim_result['base_churn_probability']:.2%}")
    print(f"Simulated Churn Probability: {sim_result['simulated_churn_probability']:.2%}")
    print(f"Probability Delta: {sim_result['probability_delta']:+.2%}")
    print(f"Base Health Score: {sim_result['base_health_score']:.1f}")
    print(f"Simulated Health Score: {sim_result['simulated_health_score']:.1f} ({sim_result['health_score_delta']:+.1f})")
    print(f"Safeguard: {sim_result['causal_disclaimer']}")
    
    print("\nMilestone 8 Retention Intelligence artifacts successfully generated!")


if __name__ == "__main__":
    main()

