"""
ChurnSignal — Data Ingestion & Preprocessing Loader
===================================================
Module for robust loading, validation, and filtering of the SaaS churn panel dataset.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np


def get_project_root() -> Path:
    """Returns the absolute root directory of the churnrate project."""
    return Path(__file__).resolve().parent.parent.parent


def load_raw_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the monthly user panel dataset from local parquet or Hugging Face.

    Parameters
    ----------
    data_path : Optional[str]
        Explicit path to the parquet file. If None, defaults to `data/user_monthly.parquet`.

    Returns
    -------
    pd.DataFrame
        Raw dataset containing 79,817 customer-month records.
    """
    if data_path is None:
        target_path = get_project_root() / "data" / "user_monthly.parquet"
    else:
        target_path = Path(data_path)

    if target_path.exists():
        df = pd.read_parquet(target_path)
    else:
        # Fallback to Hugging Face
        from datasets import load_dataset
        ds = load_dataset("arti199919/synthetic-saas-churn-sample", data_files="user_monthly.parquet")
        df = ds["train"].to_pandas()

    # Ensure datetime parsing for month column
    if not np.issubdtype(df["month"].dtype, np.datetime64):
        df["month"] = pd.to_datetime(df["month"])

    return df


def load_prediction_eligible_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and filters the dataset to include only records eligible for next-month churn prediction.

    Eligibility Rules:
    1. Active Customer at Month T: Customer has not already churned in the current month (churned == 0).
       (Because churn is an absorbing terminal state, churned == 1 records are exits, not prediction candidates).
    2. Non-truncated Observation: Month must be < 2025-12-01.
       (Month 2025-12 has artificial zero labels for churned_next_month because Jan 2026 was not recorded).

    Returns
    -------
    pd.DataFrame
        Eligible prediction dataset.
    """
    df = load_raw_dataset(data_path)
    
    # Rule 1: Customer must be active during month T
    active_mask = (df["churned"] == 0)
    
    # Rule 2: Month must have observed future next month
    observed_future_mask = (df["month"] < "2025-12-01")
    
    eligible_df = df[active_mask & observed_future_mask].copy()
    return eligible_df

