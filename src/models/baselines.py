"""
ChurnSignal — Baseline Models Engine
====================================
Implements transparent, interpretable baseline classification models:
- DummyClassifier: Most frequent class (null performance floor)
- Logistic Regression (Unweighted): Standard empirical empirical risk minimization
- Logistic Regression (Balanced): Inverse frequency class weighting for rare churn events

Provides coefficient extraction, odds-ratio translation, and model artifact serialization.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def get_artifacts_dir() -> Path:
    """Returns the path to the artifacts directory."""
    art_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    return art_dir


def train_dummy_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "most_frequent"
) -> DummyClassifier:
    """
    Trains a heuristic DummyClassifier baseline.
    
    Demonstrates the baseline floor where a model guesses the majority class (retained).
    """
    model = DummyClassifier(strategy=strategy, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: Optional[str] = "balanced",
    C: float = 1.0,
    max_iter: int = 1000
) -> LogisticRegression:
    """
    Trains an L2-regularized Logistic Regression model.

    Parameters
    ----------
    X_train : np.ndarray
        Standardized training feature matrix.
    y_train : np.ndarray
        Binary churn labels (0 or 1).
    class_weight : Optional[str]
        If 'balanced', weights inversely proportional to class frequencies to combat rare event imbalance.
    C : float
        Inverse regularization strength (smaller values specify stronger regularization).
    max_iter : int
        Maximum solver iterations.
    """
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=42,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    return model


def get_logistic_coefficients(
    model: LogisticRegression,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extracts weights from a fitted Logistic Regression model.

    Returns
    -------
    pd.DataFrame
        Table with Feature, Weight (log-odds), and Odds Ratio (exp(weight)).
        Odds Ratio > 1.0 indicates increasing churn risk; < 1.0 indicates protective factor.
    """
    weights = model.coef_[0]
    odds_ratios = np.exp(weights)
    
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": weights,
        "Odds_Ratio": odds_ratios
    }).sort_values(by="Coefficient", ascending=False).reset_index(drop=True)
    
    return coef_df


def save_model(model: Any, filename: str) -> Path:
    """Serializes a fitted model to artifacts/."""
    save_path = get_artifacts_dir() / filename
    joblib.dump(model, save_path)
    return save_path


def load_model(filename: str) -> Any:
    """Loads a serialized model from artifacts/."""
    load_path = get_artifacts_dir() / filename
    if not load_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {load_path}")
    return joblib.load(load_path)

