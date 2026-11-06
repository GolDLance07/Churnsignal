"""
Unit tests for EDA figure generation and visual artifact integrity.
"""

from pathlib import Path
import pytest


def test_eda_figures_exist_and_non_empty():
    """Verify all 6 publication figures have been rendered to reports/figures/."""
    figures_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
    
    expected_figures = [
        "01_target_imbalance.png",
        "02_univariate_distributions.png",
        "03_bivariate_comparisons.png",
        "04_plan_type_analysis.png",
        "05_correlation_heatmap.png",
        "06_temporal_dynamics.png"
    ]
    
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Figure missing: {fig_name}"
        assert fig_path.stat().st_size > 5000, f"Figure appears corrupted/empty: {fig_name}"

