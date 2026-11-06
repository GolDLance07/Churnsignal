"""
ChurnSignal — Exploratory Data Analysis & Visualization Suite
=============================================================
Generates comprehensive univariate, bivariate, correlation, and temporal
visualizations for SaaS customer retention intelligence.
"""

import os
from pathlib import Path
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")  # Headless backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def get_figures_dir() -> Path:
    """Returns the directory where EDA figures are saved."""
    fig_dir = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def plot_target_distribution(df: pd.DataFrame) -> Path:
    """Figure 1: Target distribution and severe class imbalance."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Raw Counts
    counts = df["churned_next_month"].value_counts().sort_index()
    labels = ["Retained (0)", "Churns Next Month (1)"]
    colors = ["#4C72B0", "#C44E52"]
    
    bars = axes[0].bar(labels, counts.values, color=colors, width=0.45, edgecolor="black", alpha=0.85)
    axes[0].set_title("Target Class Counts (`churned_next_month`)", fontweight="bold")
    axes[0].set_ylabel("Customer-Month Records")
    axes[0].set_yscale("log")  # Log scale to visually present both classes
    
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval * 1.15, f"{int(yval):,}", ha="center", va="bottom", fontweight="bold")

    # Right: Proportions with Donut Chart
    pcts = df["churned_next_month"].value_counts(normalize=True).sort_index() * 100
    wedges, texts, autotexts = axes[1].pie(
        pcts,
        labels=labels,
        autopct="%1.2f%%",
        startangle=140,
        colors=colors,
        explode=(0, 0.15),
        wedgeprops=dict(width=0.6, edgecolor="black")
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    axes[1].set_title("Target Class Proportions (Extreme Imbalance)", fontweight="bold")

    plt.suptitle("ChurnSignal — Target Variable Imbalance Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = get_figures_dir() / "01_target_imbalance.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_numerical_distributions(df: pd.DataFrame) -> Path:
    """Figure 2: Univariate distributions of key numeric features."""
    features = [
        ("sessions", "Monthly Sessions", "int"),
        ("feature_usage_score", "Feature Usage Score (0-100)", "float"),
        ("active_seats", "Active Seats Count", "int"),
        ("mrr", "Monthly Recurring Revenue ($)", "float"),
        ("nps_score", "NPS Score (0-10)", "int"),
        ("tenure_month", "Customer Tenure (Months)", "int")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (col, label, kind) in enumerate(features):
        ax = axes[idx]
        if kind == "int" and df[col].nunique() <= 12:
            sns.countplot(data=df, x=col, ax=ax, color="#4C72B0", edgecolor="black", alpha=0.8)
        else:
            sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0", bins=30, edgecolor="black", alpha=0.7)
            
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="green", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.1f}")
        
        ax.set_title(f"Distribution: {label}", fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("ChurnSignal — Univariate Distributions of Key Customer Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = get_figures_dir() / "02_univariate_distributions.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_bivariate_churn_comparisons(df: pd.DataFrame) -> Path:
    """Figure 3: Bivariate comparisons showing how features differ between retained and churning customers."""
    features = [
        ("feature_usage_score", "Feature Usage Score"),
        ("sessions", "Monthly Sessions"),
        ("nps_score", "NPS Score (0-10)"),
        ("support_tickets", "Support Tickets"),
        ("payment_failures", "Payment Failures"),
        ("active_seats", "Active Seats Count")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(features):
        ax = axes[idx]
        
        # Plot boxplot without extreme fliers to see interquartile shifts clearly
        sns.boxplot(
            data=df,
            x="churned_next_month",
            y=col,
            hue="churned_next_month",
            ax=ax,
            palette=["#4C72B0", "#C44E52"],
            legend=False,
            showfliers=False,
            width=0.4
        )
        
        retained_mean = df[df["churned_next_month"] == 0][col].mean()
        churn_mean = df[df["churned_next_month"] == 1][col].mean()
        pct_diff = ((churn_mean - retained_mean) / (retained_mean if retained_mean != 0 else 1)) * 100
        
        ax.set_title(f"{label}\n(Churn Mean: {churn_mean:.1f} vs Retained: {retained_mean:.1f} | Δ: {pct_diff:+.1f}%)", fontweight="bold", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Retained (0)", "Churns Next Month (1)"], fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(label)

    plt.suptitle("ChurnSignal — Bivariate Analysis: Retained vs. Churning Customers", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = get_figures_dir() / "03_bivariate_comparisons.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_categorical_plan_analysis(df: pd.DataFrame) -> Path:
    """Figure 4: Churn rates and customer counts across subscription plan tiers."""
    plan_order = ["Starter", "Pro", "Business", "Enterprise"]
    
    plan_stats = df.groupby("plan_type", observed=False).agg(
        total_records=("user_id", "count"),
        churn_events=("churned_next_month", "sum"),
        churn_rate_pct=("churned_next_month", lambda x: x.mean() * 100)
    ).reindex(plan_order)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Bar plot for customer volume
    color_bar = "#4C72B0"
    ax1.bar(plan_stats.index, plan_stats["total_records"], color=color_bar, alpha=0.7, width=0.45, edgecolor="black", label="Customer-Month Records")
    ax1.set_ylabel("Total Observations", color=color_bar, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_bar)
    ax1.set_xlabel("Subscription Tier (`plan_type`)", fontweight="bold")

    for i, v in enumerate(plan_stats["total_records"]):
        ax1.text(i, v * 0.5, f"{int(v):,}", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    # Line plot on twin axis for churn rate
    ax2 = ax1.twinx()
    color_line = "#C44E52"
    ax2.plot(plan_stats.index, plan_stats["churn_rate_pct"], color=color_line, marker="o", linewidth=2.5, markersize=8, label="Next-Month Churn Rate (%)")
    ax2.set_ylabel("Next-Month Churn Rate (%)", color=color_line, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.set_ylim(0, max(plan_stats["churn_rate_pct"]) * 1.4)
    ax2.grid(False)

    for i, v in enumerate(plan_stats["churn_rate_pct"]):
        ax2.text(i, v + 0.12, f"{v:.2f}%", ha="center", va="bottom", color=color_line, fontweight="bold")

    plt.title("ChurnSignal — Churn Dynamics Across Subscription Plan Types", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    
    save_path = get_figures_dir() / "04_plan_type_analysis.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_correlation_matrix(df: pd.DataFrame) -> Path:
    """Figure 5: Correlation matrix highlighting multicollinearity and leading indicators."""
    numeric_cols = [
        "tenure_month", "monthly_price", "mrr", "sessions", "feature_usage_score",
        "support_tickets", "payment_failures", "nps_score", "product_incident",
        "active_seats", "churned", "churned_next_month"
    ]
    
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    plt.title("ChurnSignal — Feature Correlation Heatmap (Pearson)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    
    save_path = get_figures_dir() / "05_correlation_heatmap.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_temporal_cohort_dynamics(df: pd.DataFrame) -> Path:
    """Figure 6: Temporal trajectory showing active base and monthly churn rate."""
    monthly_summary = df.groupby(df["month"].dt.strftime("%Y-%m")).agg(
        active_users=("user_id", "count"),
        churned_current=("churned", "sum"),
        churned_next=("churned_next_month", "sum"),
        churn_rate_next=("churned_next_month", lambda x: x.mean() * 100)
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: Active User Trajectory
    ax1.plot(monthly_summary.index, monthly_summary["active_users"], color="#2B5C8F", marker="s", linewidth=2.5)
    ax1.set_title("Active Customer Base Evolution (24 Calendar Months)", fontweight="bold")
    ax1.set_ylabel("Active Customers")
    ax1.axvline("2025-01", color="purple", linestyle="--", alpha=0.7, label="End of User Onboarding Phase")
    ax1.legend(loc="lower right")

    # Bottom: Monthly Next-Month Churn Rate
    ax2.plot(monthly_summary.index, monthly_summary["churn_rate_next"], color="#C44E52", marker="o", linewidth=2.5)
    ax2.set_title("Next-Month Churn Rate (%) Over Time", fontweight="bold")
    ax2.set_ylabel("Churn Rate (%)")
    ax2.set_xlabel("Calendar Month (Observation Timestamp)")
    ax2.set_xticks(range(len(monthly_summary.index)))
    ax2.set_xticklabels(monthly_summary.index, rotation=45, ha="right")
    
    # Annotate Dec 2025 anomaly
    ax2.annotate(
        "Truncation Anomaly\n(Artificially 0.00%)",
        xy=("2025-12", 0),
        xytext=("2025-09", 1.2),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
    )

    plt.suptitle("ChurnSignal — Macro Temporal Dynamics & Lifecycle Trajectory", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    save_path = get_figures_dir() / "06_temporal_dynamics.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def run_full_eda(data_path: str = None):
    """Executes the full EDA visualization pipeline."""
    from src.data.loader import load_raw_dataset
    
    print("================================================================================")
    print("  CHURNSIGNAL — MILESTONE 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("  Generating publication-quality charts and behavioral diagnostic tables")
    print("================================================================================")
    
    df = load_raw_dataset(data_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns.\n")
    
    print("1. Generating Target Imbalance Plot...")
    p1 = plot_target_distribution(df)
    print(f"   -> Saved: {p1}")

    print("2. Generating Univariate Numerical Distributions...")
    p2 = plot_numerical_distributions(df)
    print(f"   -> Saved: {p2}")

    print("3. Generating Bivariate Churn Comparison Plots...")
    p3 = plot_bivariate_churn_comparisons(df)
    print(f"   -> Saved: {p3}")

    print("4. Generating Plan Type Categorical Breakdown...")
    p4 = plot_categorical_plan_analysis(df)
    print(f"   -> Saved: {p4}")

    print("5. Generating Feature Correlation Heatmap...")
    p5 = plot_correlation_matrix(df)
    print(f"   -> Saved: {p5}")

    print("6. Generating Temporal Cohort Trajectory...")
    p6 = plot_temporal_cohort_dynamics(df)
    print(f"   -> Saved: {p6}")

    print("\n================================================================================")
    print("  EDA COMPLETE: All 6 figures exported to reports/figures/")
    print("================================================================================")


if __name__ == "__main__":
    run_full_eda()
