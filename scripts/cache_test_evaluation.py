"""
Precomputes and caches held-out test set predictions and metrics for Streamlit.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data.loader import load_prediction_eligible_data
from src.features.engineer import engineer_all_features
from src.features.pipeline import chronological_split, TARGET_COLUMN

def main():
    print("Loading data for held-out test evaluation...")
    raw = load_prediction_eligible_data()
    featured = engineer_all_features(raw)
    _, _, test_df = chronological_split(featured)
    
    pipeline = joblib.load("artifacts/feature_pipeline.joblib")
    model = joblib.load("artifacts/champion_model.joblib")
    
    print(f"Transforming test split ({len(test_df)} observations)...")
    X_test = pipeline.transform(test_df)
    y_test = test_df[TARGET_COLUMN].to_numpy().astype(int)
    y_prob = model.predict_proba(X_test)[:, 1].astype(float)
    
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    
    out_path = Path("artifacts/test_evaluation.npz")
    np.savez_compressed(
        out_path,
        y_test=y_test,
        y_prob=y_prob,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        test_size=len(y_test),
        actual_churn_count=int(y_test.sum())
    )
    print(f"Cached test evaluation to {out_path}!")
    print(f"Test size: {len(y_test)}, Churns: {int(y_test.sum())}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

if __name__ == "__main__":
    main()

