"""
ChurnSignal — Explainability Module
"""

from src.explainability.explainer import (
    build_tree_explainer,
    compute_global_feature_importance,
    explain_single_customer,
    plot_shap_global_summary,
    plot_customer_waterfall,
)
from src.explainability.retention import (
    calculate_customer_health_score,
    assign_risk_tier,
    assign_risk_category,
    get_retention_status,
    calculate_expected_mrr_at_risk,
    generate_retention_playbook,
    simulate_what_if_counterfactual,
    plot_retention_matrix,
)

__all__ = [
    "build_tree_explainer",
    "compute_global_feature_importance",
    "explain_single_customer",
    "plot_shap_global_summary",
    "plot_customer_waterfall",
    "calculate_customer_health_score",
    "assign_risk_tier",
    "assign_risk_category",
    "get_retention_status",
    "calculate_expected_mrr_at_risk",
    "generate_retention_playbook",
    "simulate_what_if_counterfactual",
    "plot_retention_matrix",
]

