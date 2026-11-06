"""Models package for ChurnSignal."""

from src.models.baselines import (
    train_dummy_baseline,
    train_logistic_baseline,
    get_logistic_coefficients,
    save_model,
    load_model
)
from src.models.comparison import (
    train_decision_tree,
    train_random_forest,
    train_lightgbm_champion,
    train_all_models,
    evaluate_model_suite,
    plot_model_comparison_curves,
    plot_model_comparison_bar
)

__all__ = [
    "train_dummy_baseline",
    "train_logistic_baseline",
    "get_logistic_coefficients",
    "save_model",
    "load_model",
    "train_decision_tree",
    "train_random_forest",
    "train_lightgbm_champion",
    "train_all_models",
    "evaluate_model_suite",
    "plot_model_comparison_curves",
    "plot_model_comparison_bar"
]
