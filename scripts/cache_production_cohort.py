"""
Pre-scores and caches the active customer cohort for low-latency serving.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from src.data.loader import load_prediction_eligible_data
from src.features.engineer import engineer_all_features
from src.explainability.retention import (
    calculate_customer_health_score,
    assign_risk_tier,
    calculate_expected_mrr_at_risk
)

def main():
    print("Loading eligible data and engineering features...")
    raw = load_prediction_eligible_data()
    featured = engineer_all_features(raw)
    
    latest_month = featured["month"].max()
    cohort = featured[featured["month"] == latest_month].copy().reset_index(drop=True)
    
    print(f"Scoring {len(cohort)} accounts for latest month {latest_month}...")
    pipeline = joblib.load("artifacts/feature_pipeline.joblib")
    model = joblib.load("artifacts/champion_model.joblib")
    
    X_cohort = pipeline.transform(cohort)
    probs = model.predict_proba(X_cohort)[:, 1]
    health = calculate_customer_health_score(cohort).to_numpy()
    
    cohort["churn_probability"] = probs
    cohort["health_score"] = health
    cohort["risk_tier"] = [assign_risk_tier(p) for p in probs]
    cohort["expected_mrr_at_risk"] = [
        calculate_expected_mrr_at_risk(mrr, p) for mrr, p in zip(cohort["mrr"], probs)
    ]
    
    target_path = Path("artifacts/scored_active_cohort.parquet")
    cohort.to_parquet(target_path, index=False)
    print(f"Successfully cached {len(cohort)} scored accounts to {target_path}!")

if __name__ == "__main__":
    main()

