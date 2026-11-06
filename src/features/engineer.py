"""
ChurnSignal — Feature Engineering Engine
========================================
Transforms raw monthly snapshots into dynamic behavioral features:
- Velocity & delta signals (momentum)
- Rolling window health aggregations (recency & frequency)
- Customer health and friction risk flags
- Cold-start handling for new customer accounts

Guarantees:
- Zero temporal look-ahead (all transformations use strictly current or prior months).
- Deterministic, reproducible calculations.
"""

import pandas as pd
import numpy as np


def compute_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes month-over-month change (velocity) for key engagement and friction metrics.
    
    Formula: Delta = Metric_T - Metric_{T-1}
    For tenure 0 (cold start / month 1), Delta is imputed as 0.0 (neutral baseline).
    """
    df = df.sort_values(["user_id", "month"]).copy()
    
    delta_cols = [
        ("sessions", "delta_sessions"),
        ("feature_usage_score", "delta_usage"),
        ("nps_score", "delta_nps"),
        ("support_tickets", "delta_support_tickets"),
        ("payment_failures", "delta_payment_failures"),
        ("active_seats", "delta_active_seats")
    ]
    
    for raw_col, new_col in delta_cols:
        df[new_col] = df.groupby("user_id")[raw_col].diff().fillna(0.0).astype(np.float32)
        
    return df


def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes relative change ratios and seat utilization.
    
    Formula: Ratio = (Metric_T + 1) / (Metric_{T-1} + 1)
    Adding 1 prevents division-by-zero errors.
    For tenure 0, Ratio is imputed as 1.0 (neutral ratio).
    """
    df = df.sort_values(["user_id", "month"]).copy()
    
    prev_sessions = df.groupby("user_id")["sessions"].shift(1)
    df["session_ratio"] = ((df["sessions"] + 1.0) / (prev_sessions + 1.0)).fillna(1.0).astype(np.float32)
    
    prev_usage = df.groupby("user_id")["feature_usage_score"].shift(1)
    df["usage_ratio"] = ((df["feature_usage_score"] + 1.0) / (prev_usage + 1.0)).fillna(1.0).astype(np.float32)
    
    # Engagement density: average sessions per active seat
    df["seat_utilization"] = (df["sessions"] / (df["active_seats"] + 1.0)).astype(np.float32)
    
    return df


def compute_rolling_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Computes rolling past-window aggregations strictly on past history (months <= T).
    
    Uses min_periods=1 so newer accounts (tenure < window) compute over available history.
    """
    df = df.sort_values(["user_id", "month"]).copy()
    
    # Rolling 3-month average engagement
    df["rolling_mean_sessions_3m"] = (
        df.groupby("user_id")["sessions"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
        .astype(np.float32)
    )
    
    df["rolling_mean_usage_3m"] = (
        df.groupby("user_id")["feature_usage_score"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
        .astype(np.float32)
    )
    
    # Rolling friction: maximum payment failure in past 3 months
    df["rolling_max_payment_failures_3m"] = (
        df.groupby("user_id")["payment_failures"]
        .transform(lambda s: s.rolling(window, min_periods=1).max())
        .astype(np.float32)
    )
    
    # Rolling support burden: sum of support tickets over past 3 months
    df["rolling_sum_support_tickets_3m"] = (
        df.groupby("user_id")["support_tickets"]
        .transform(lambda s: s.rolling(window, min_periods=1).sum())
        .astype(np.float32)
    )
    
    return df


def compute_domain_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Computes binary operational risk indicators based on SaaS domain knowledge."""
    df = df.copy()
    
    # Inactive team flag: Paying account with zero active user seats
    df["is_zero_seats"] = (df["active_seats"] == 0).astype(np.int8)
    
    # Detractor flag: NPS score <= 6 (standard Net Promoter Score detractor band)
    df["is_nps_detractor"] = (df["nps_score"] <= 6).astype(np.int8)
    
    # Billing distress flag: Any payment failure in the current snapshot
    df["has_payment_failure"] = (df["payment_failures"] > 0).astype(np.int8)
    
    # Cold start flag: First month of customer tenure
    df["is_new_customer"] = (df["tenure_month"] == 0).astype(np.int8)
    
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline orchestrator.
    
    Applies deltas, ratios, rolling windows, and risk flags while dropping redundant features.
    """
    df = df.sort_values(["user_id", "month"]).copy()
    
    # 1. Delta features
    df = compute_delta_features(df)
    
    # 2. Ratio features
    df = compute_ratio_features(df)
    
    # 3. Rolling window features
    df = compute_rolling_features(df, window=3)
    
    # 4. Domain risk flags
    df = compute_domain_risk_flags(df)
    
    # 5. Drop exact redundant collinear feature
    if "monthly_price" in df.columns:
        df = df.drop(columns=["monthly_price"])
        
    return df

