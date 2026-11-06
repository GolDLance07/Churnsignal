"""
ChurnSignal — Generate SHAP Artifacts & Figures (Milestone 7)
"""

import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from src.data.loader import load_prediction_eligible_data
from src.features.engineer import engineer_all_features
from src.features.pipeline import chronological_split, TARGET_COLUMN
from src.explainability.explainer import (
    build_tree_explainer,
    compute_global_feature_importance,
    explain_single_customer,
    plot_shap_global_summary,
    plot_customer_waterfall
)

def main():
    print("Loading dataset and preprocessing...")
    eligible_df = load_prediction_eligible_data()
    featured_df = engineer_all_features(eligible_df)
    
    train_df, val_df, test_df = chronological_split(featured_df)
    
    print("Loading pipeline and champion LightGBM model...")
    pipeline = joblib.load("artifacts/feature_pipeline.joblib")
    champion_model = joblib.load("artifacts/champion_model.joblib")
    
    feature_names = pipeline.get_feature_names()
    X_val = pipeline.transform(val_df)
    y_val = val_df[TARGET_COLUMN].to_numpy()
    val_probs = champion_model.predict_proba(X_val)[:, 1]
    
    print(f"Building TreeExplainer on Champion LightGBM ({len(X_val)} validation rows)...")
    explainer = build_tree_explainer(champion_model)
    
    # 1. Global Feature Importance
    print("Computing global SHAP feature importance...")
    global_importance = compute_global_feature_importance(explainer, X_val, feature_names)
    print("\nTop 10 Global Features by Mean |SHAP Value|:")
    print(global_importance.head(10).to_string(index=False))
    
    fig13 = plot_shap_global_summary(explainer, X_val, feature_names)
    print(f"\nGenerated: {fig13}")
    
    # 2. Local Explanation for Top Risk Customer
    # Find a customer with high predicted churn risk in validation set
    top_risk_idx = np.argsort(val_probs)[::-1][0]
    top_customer_id = val_df.iloc[top_risk_idx]["user_id"]
    top_cust_prob = val_probs[top_risk_idx]
    top_cust_true = y_val[top_risk_idx]
    
    print(f"\nExplaining Customer {top_customer_id} (Predicted P(churn)={top_cust_prob:.4f}, True={top_cust_true})...")
    customer_vector = X_val[top_risk_idx]
    explanation = explain_single_customer(explainer, customer_vector, feature_names, top_n=5)
    
    print(f"Base Value (log-odds): {explanation['base_value']:.4f}")
    print(f"Raw Margin (sum + base): {explanation['raw_margin']:.4f}")
    print("Top Risk Drivers (pushing churn risk UP):")
    for r in explanation["risk_drivers"]:
        print(f"  + {r['Feature']}: SHAP={r['SHAP_Value']:+.4f} (Input={r['Feature_Input']:.3f})")
    print("Top Protective Factors (pulling churn risk DOWN):")
    for p in explanation["protective_factors"]:
        print(f"  - {p['Feature']}: SHAP={p['SHAP_Value']:+.4f} (Input={p['Feature_Input']:.3f})")
        
    fig14 = plot_customer_waterfall(
        explainer,
        customer_vector,
        feature_names,
        customer_id=str(top_customer_id),
        filename="14_customer_shap_waterfall.png"
    )
    print(f"Generated: {fig14}")
    print("\nMilestone 7 SHAP artifacts successfully created!")

if __name__ == "__main__":
    main()
