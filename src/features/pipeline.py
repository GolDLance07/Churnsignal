"""
ChurnSignal — Preprocessing & Temporal Partitioning Pipeline
============================================================
Provides scikit-learn compatible ColumnTransformer pipelines with:
- Strict Out-of-Time chronological train/validation/test splitting.
- Leakage-safe fitting (scalers and encoders fit exclusively on the training split).
- One-Hot Encoding for categorical tiers.
- StandardScaler for numeric and velocity features.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Feature Category Definitions
CATEGORICAL_FEATURES = ["plan_type"]

BINARY_FEATURES = [
    "product_incident",
    "is_zero_seats",
    "is_nps_detractor",
    "has_payment_failure",
    "is_new_customer"
]

NUMERICAL_FEATURES = [
    "tenure_month",
    "mrr",
    "sessions",
    "feature_usage_score",
    "support_tickets",
    "payment_failures",
    "nps_score",
    "active_seats",
    "delta_sessions",
    "delta_usage",
    "delta_nps",
    "delta_support_tickets",
    "delta_payment_failures",
    "delta_active_seats",
    "session_ratio",
    "usage_ratio",
    "seat_utilization",
    "rolling_mean_sessions_3m",
    "rolling_mean_usage_3m",
    "rolling_max_payment_failures_3m",
    "rolling_sum_support_tickets_3m"
]

TARGET_COLUMN = "churned_next_month"
METADATA_COLUMNS = ["user_id", "month", "churned"]


def chronological_split(
    df: pd.DataFrame,
    train_end: str = "2025-06-01",
    val_start: str = "2025-07-01",
    val_end: str = "2025-09-01",
    test_start: str = "2025-10-01",
    test_end: str = "2025-11-01"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs an Out-of-Time (OOT) chronological train/validation/test partition.

    Parameters
    ----------
    df : pd.DataFrame
        Eligible customer panel dataframe.
    train_end : str
        Upper bound date for training history (inclusive).
    val_start : str
        Lower bound date for validation set (inclusive).
    val_end : str
        Upper bound date for validation set (inclusive).
    test_start : str
        Lower bound date for holdout test set (inclusive).
    test_end : str
        Upper bound date for holdout test set (inclusive).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df) strictly partitioned by calendar month.
    """
    train_mask = (df["month"] <= train_end)
    val_mask = (df["month"] >= val_start) & (df["month"] <= val_end)
    test_mask = (df["month"] >= test_start) & (df["month"] <= test_end)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # Integrity verification
    assert len(train_df) > 0, "Train split is empty!"
    assert len(val_df) > 0, "Validation split is empty!"
    assert len(test_df) > 0, "Test split is empty!"
    assert train_df["month"].max() < val_df["month"].min(), "Temporal overlap between train and val!"
    assert val_df["month"].max() < test_df["month"].min(), "Temporal overlap between val and test!"

    return train_df, val_df, test_df


class ChurnFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible ColumnTransformer wrapper for ChurnSignal.
    
    Guarantees:
    - Scales numeric features using StandardScaler.
    - Encodes plan_type using OneHotEncoder.
    - Preserves binary risk flags without distortion.
    - Retains clean feature names for interpretability and SHAP attribution.
    """

    def __init__(self):
        self.numerical_cols = NUMERICAL_FEATURES
        self.categorical_cols = CATEGORICAL_FEATURES
        self.binary_cols = BINARY_FEATURES
        self.preprocessor = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fits scalers and encoders strictly on the training partition."""
        transformers = [
            ("num", StandardScaler(), self.numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols),
            ("bin", "passthrough", self.binary_cols)
        ]
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )
        self.preprocessor.fit(X)
        
        # Store clean feature names
        num_names = self.numerical_cols
        cat_names = self.preprocessor.named_transformers_["cat"].get_feature_names_out(self.categorical_cols).tolist()
        bin_names = self.binary_cols
        self.feature_names_ = num_names + cat_names + bin_names
        
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms feature dataframe using fitted parameters."""
        if self.preprocessor is None:
            raise RuntimeError("Pipeline must be fitted before calling transform!")
        return self.preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms and returns a labeled Pandas DataFrame with column names."""
        transformed_array = self.transform(X)
        return pd.DataFrame(transformed_array, columns=self.feature_names_, index=X.index)

    def get_feature_names(self) -> List[str]:
        """Returns the list of final transformed feature names."""
        return self.feature_names_

