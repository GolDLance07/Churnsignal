"""
ChurnSignal — Customer Retention Intelligence Platform
======================================================
Interactive Streamlit Application featuring:
1. Executive Retention Overview & Batch Risk Distribution
2. Customer Risk Intelligence & Search Table
3. Individual Customer Prediction, Risk Categorization & SHAP Diagnosis
4. Counterfactual What-If Simulator with Causal Safeguards
5. Held-Out Model Evaluation & Interactive Threshold Explorer
6. Technical Architecture & Decision Records
"""

from pathlib import Path
import os
import sys
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Ensure churnrate root in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_prediction_eligible_data
from src.features.engineer import engineer_all_features
from src.explainability.explainer import (
    build_tree_explainer,
    explain_single_customer,
    compute_global_feature_importance
)
from src.explainability.retention import (
    calculate_customer_health_score,
    assign_risk_tier,
    assign_risk_category,
    get_retention_status,
    calculate_expected_mrr_at_risk,
    generate_retention_playbook,
    simulate_what_if_counterfactual
)

# Page configuration
st.set_page_config(
    page_title="ChurnSignal | Customer Retention Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-high {
        background-color: #ffebe9;
        color: #cf222e;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fff8c5;
        color: #9a6700;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-low {
        background-color: #dafbe1;
        color: #1a7f37;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .callout-info {
        background-color: #f0f7ff;
        border-left: 4px solid #0969da;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
    }
    .causal-box {
        background-color: #fff8c5;
        border-left: 4px solid #bf8700;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_artifacts():
    """Loads feature pipeline, champion model, and TreeExplainer."""
    pipeline = joblib.load(project_root / "artifacts" / "feature_pipeline.joblib")
    champion_model = joblib.load(project_root / "artifacts" / "champion_model.joblib")
    explainer = build_tree_explainer(champion_model)
    return pipeline, champion_model, explainer


@st.cache_data
def load_active_cohort():
    """Loads pre-scored active customer cohort for fast UI responsiveness."""
    pipeline, model, _ = load_ml_artifacts()
    cached_path = project_root / "artifacts" / "scored_active_cohort.parquet"
    if cached_path.exists():
        cohort = pd.read_parquet(cached_path)
    else:
        raw = load_prediction_eligible_data()
        featured = engineer_all_features(raw)
        latest_month = featured["month"].max()
        cohort = featured[featured["month"] == latest_month].copy().reset_index(drop=True)
        X_vec = pipeline.transform(cohort)
        probs = model.predict_proba(X_vec)[:, 1]
        cohort["churn_probability"] = probs
        cohort["health_score"] = calculate_customer_health_score(cohort).to_numpy()
        cohort.to_parquet(cached_path, index=False)
        
    # Standardize risk categories and retention status
    cohort["risk_category"] = [assign_risk_category(p) for p in cohort["churn_probability"]]
    cohort["retention_status"] = [
        get_retention_status(p, h) for p, h in zip(cohort["churn_probability"], cohort["health_score"])
    ]
    cohort["expected_mrr_at_risk"] = [
        calculate_expected_mrr_at_risk(mrr, p) for mrr, p in zip(cohort["mrr"], cohort["churn_probability"])
    ]
    
    X_vec = pipeline.transform(cohort)
    latest_month = cohort["month"].max()
    return cohort, X_vec, latest_month


@st.cache_data
def load_test_evaluation_cache():
    """Loads held-out test evaluation predictions for performance metrics."""
    test_cache_path = project_root / "artifacts" / "test_evaluation.npz"
    if test_cache_path.exists():
        data = np.load(test_cache_path)
        return {
            "y_test": data["y_test"],
            "y_prob": data["y_prob"],
            "roc_auc": float(data["roc_auc"]),
            "pr_auc": float(data["pr_auc"]),
            "test_size": int(data["test_size"]),
            "actual_churn_count": int(data["actual_churn_count"])
        }
    return None


# Initialize assets
try:
    pipeline, champion_model, explainer = load_ml_artifacts()
    cohort_df, X_cohort, latest_month = load_active_cohort()
    test_eval_data = load_test_evaluation_cache()
    feature_names = pipeline.get_feature_names()
except Exception as e:
    st.error(f"Error initializing ChurnSignal assets: {e}")
    st.stop()


# Sidebar Navigation
st.sidebar.title("⚡ ChurnSignal")
st.sidebar.caption("Customer Retention Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Portfolio Overview",
        "🎯 Customer Risk Explorer",
        "🔍 Individual Prediction & SHAP",
        "📈 Model Performance & Evaluation",
        "🧪 What-If Simulator",
        "📋 System Architecture & Docs"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Cohort Month**: `{str(latest_month)[:7]}`")
st.sidebar.markdown(f"**Analyzed Accounts**: `{len(cohort_df):,}`")
st.sidebar.markdown(f"**Model**: `LightGBM (OOT PR-AUC: 0.070)`")
st.sidebar.markdown("---")
st.sidebar.info(r"💡 **Decision Rule**: Accounts with predicted churn probability $\ge 35\%$ are classified as **High Risk**.")


# ==============================================================================
# PAGE 1: PORTFOLIO OVERVIEW & BATCH PREDICTION METRICS
# ==============================================================================
if page == "📊 Portfolio Overview":
    st.title("📊 Portfolio Retention Overview")
    st.markdown("High-level customer health, batch risk categorization, and revenue exposure across all analyzed accounts.")
    
    # Batch Prediction Metrics
    total_customers = len(cohort_df)
    high_risk_df = cohort_df[cohort_df["risk_category"] == "High"]
    high_risk_count = len(high_risk_df)
    high_risk_pct = (high_risk_count / total_customers) * 100
    avg_churn_prob = cohort_df["churn_probability"].mean() * 100
    avg_health_score = cohort_df["health_score"].mean()
    total_mrr = cohort_df["mrr"].sum()
    total_risk_mrr = cohort_df["expected_mrr_at_risk"].sum()
    
    # Primary KPI Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers Analyzed", f"{total_customers:,}")
    col2.metric("High-Risk Customers", f"{high_risk_count:,}", f"{high_risk_pct:.1f}% of cohort", delta_color="inverse")
    col3.metric("Avg Churn Probability", f"{avg_churn_prob:.1f}%")
    col4.metric("Avg Customer Health", f"{avg_health_score:.1f} / 100")
    col5.metric("Expected MRR at Risk", f"${total_risk_mrr:,.0f}", f"{total_risk_mrr/total_mrr*100:.1f}% Exposure", delta_color="inverse")
    
    st.markdown("---")
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("Strategic 2x2 Retention & Health Matrix")
        st.markdown("Cross-tabulating **Customer Health Score** against **Forward Churn Probability** to identify blindspots and crises.")
        matrix_path = project_root / "reports" / "figures" / "15_retention_matrix.png"
        if matrix_path.exists():
            st.image(str(matrix_path), width=750)
        else:
            st.info("Retention matrix image is being generated.")
            
    with col_right:
        st.subheader("Risk Category Distribution")
        st.caption(r"Categorization: High ($\ge 35\%$), Medium ($15\% - 35\%$), Low ($< 15\%$)")
        
        dist_df = cohort_df["risk_category"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        dist_df.columns = ["Risk Category", "Customer Count"]
        dist_df["Percentage"] = (dist_df["Customer Count"] / total_customers * 100).round(1).astype(str) + "%"
        
        st.dataframe(dist_df, hide_index=True, width=450)
        
        # Streamlit Bar Chart for distribution
        chart_data = pd.DataFrame({
            "Count": dist_df["Customer Count"].values
        }, index=dist_df["Risk Category"].values)
        st.bar_chart(chart_data)
        
        st.subheader("Top Revenue Exposure Accounts")
        st.caption("Prioritized by Expected MRR at Risk (MRR * P(churn))")
        top_rev = cohort_df.sort_values(by="expected_mrr_at_risk", ascending=False).head(5)
        display_top = top_rev[["user_id", "plan_type", "mrr", "churn_probability", "expected_mrr_at_risk"]].copy()
        display_top["mrr"] = display_top["mrr"].apply(lambda v: f"${v:,.2f}")
        display_top["churn_probability"] = display_top["churn_probability"].apply(lambda v: f"{v*100:.1f}%")
        display_top["expected_mrr_at_risk"] = display_top["expected_mrr_at_risk"].apply(lambda v: f"${v:,.2f}")
        st.dataframe(display_top, hide_index=True, width=450)


# ==============================================================================
# PAGE 2: CUSTOMER RISK EXPLORER TABLE
# ==============================================================================
elif page == "🎯 Customer Risk Explorer":
    st.title("🎯 Customer Risk Intelligence Explorer")
    st.markdown("Filter, sort, and inspect active customer accounts based on predictive churn probability and financial impact.")
    
    # Filter Controls
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        selected_plan = st.selectbox("Filter by Plan Tier", ["All"] + list(cohort_df["plan_type"].unique()))
    with fcol2:
        selected_cat = st.selectbox("Filter by Risk Category", ["All", "High", "Medium", "Low"])
    with fcol3:
        sort_choice = st.selectbox("Sort Accounts By", [
            "Expected MRR at Risk (Highest First)",
            "Churn Probability (Highest First)",
            "Customer Health Score (Lowest First)",
            "MRR (Highest First)"
        ])
        
    filtered = cohort_df.copy()
    if selected_plan != "All":
        filtered = filtered[filtered["plan_type"] == selected_plan]
    if selected_cat != "All":
        filtered = filtered[filtered["risk_category"] == selected_cat]
        
    if "Expected MRR at Risk" in sort_choice:
        filtered = filtered.sort_values(by="expected_mrr_at_risk", ascending=False)
    elif "Churn Probability" in sort_choice:
        filtered = filtered.sort_values(by="churn_probability", ascending=False)
    elif "Customer Health Score" in sort_choice:
        filtered = filtered.sort_values(by="health_score", ascending=True)
    elif "MRR" in sort_choice:
        filtered = filtered.sort_values(by="mrr", ascending=False)
        
    st.markdown(f"**Showing {len(filtered):,} matching accounts**")
    
    display_table = filtered[[
        "user_id", "plan_type", "mrr", "tenure_month", "active_seats",
        "churn_probability", "risk_category", "health_score", "retention_status", "expected_mrr_at_risk"
    ]].copy()
    
    display_table["mrr"] = display_table["mrr"].apply(lambda v: f"${v:,.2f}")
    display_table["churn_probability"] = display_table["churn_probability"].apply(lambda v: f"{v*100:.1f}%")
    display_table["health_score"] = display_table["health_score"].apply(lambda v: f"{v:.1f}/100")
    display_table["expected_mrr_at_risk"] = display_table["expected_mrr_at_risk"].apply(lambda v: f"${v:,.2f}")
    
    st.dataframe(display_table, width=1100, height=480)
    
    # Download action list
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Filtered Retention Action List (CSV)",
        csv_data,
        "churnsignal_action_list.csv",
        "text/csv"
    )


# ==============================================================================
# PAGE 3: INDIVIDUAL CUSTOMER PREDICTION & SHAP EXPLAINABILITY
# ==============================================================================
elif page == "🔍 Individual Prediction & SHAP":
    st.title("🔍 Individual Customer Prediction & Diagnosis")
    st.markdown("Inspect granular model predictions, risk categorization, operational health scores, and SHAP feature drivers.")
    
    # Account selector
    top_accounts = cohort_df.sort_values(by="expected_mrr_at_risk", ascending=False)["user_id"].tolist()
    selected_id = st.selectbox("Select Customer Account:", top_accounts[:150])
    
    cust_data = cohort_df[cohort_df["user_id"] == selected_id].iloc[0]
    cust_row_idx = cohort_df.index.get_loc(cust_data.name)
    
    prob_val = cust_data["churn_probability"]
    prob_pct = f"{prob_val * 100:.1f}%"
    risk_cat = cust_data["risk_category"]
    health_val = cust_data["health_score"]
    ret_status = cust_data["retention_status"]
    
    # Primary Prediction Score Cards
    st.markdown("### Prediction Scores")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated Churn Probability", prob_pct)
    c2.metric("Assigned Risk Category", risk_cat)
    c3.metric("Customer Health Score", f"{health_val:.1f} / 100")
    c4.metric("Expected MRR at Risk", f"${cust_data['expected_mrr_at_risk']:,.2f}")
    
    # Visual Progress Indicators
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.caption("Churn Likelihood Gauge (0% = Retained, 100% = Churn)")
        st.progress(float(prob_val))
    with pcol2:
        st.caption("Customer Health Gauge (0 = Critical, 100 = Flourishing)")
        st.progress(float(min(1.0, max(0.0, health_val / 100.0))))
        
    # Retention Status Alert Box
    badge_class = f"badge-{risk_cat.lower()}"
    st.markdown(f"""
    <div class="callout-info">
        <strong>Retention Status:</strong> {ret_status}<br>
        <strong>Risk Category:</strong> <span class="{badge_class}">{risk_cat} Risk</span> | 
        <strong>Subscription Tier:</strong> {cust_data['plan_type']} (${cust_data['mrr']:,.2f}/month) | 
        <strong>Tenure:</strong> {cust_data['tenure_month']} months
    </div>
    """, unsafe_allow_html=True)
    
    # Clear Explanation Box: Health Score vs Churn Probability
    with st.expander("ℹ️ How Customer Health Score Differs from Churn Probability", expanded=True):
        st.markdown("""
        It is important to distinguish between **Customer Health Score** and **Predicted Churn Probability**:
        
        - **Customer Health Score (0–100):** A deterministic, operational vitality index based on 4 visible business pillars: Engagement (35%), Sentiment (25%), Product Friction (20%), and Billing Hygiene (20%). It answers: *"How active and satisfied is the customer right now?"*
        - **Predicted Churn Probability (0–100%):** A forward-looking statistical forecast generated by our trained LightGBM model across 30 non-linear features. It answers: *"What is the probability this customer will leave during the upcoming month?"*
        - **Why Both are Critical:** An account might have good health (e.g., 70/100) but still have a high churn probability if an unresolved payment failure or contract renewal blocker arises. Conversely, a low-usage account might have low health (30/100) but remain retained due to long-term contract lock-in.
        """)
        
    st.markdown("---")
    
    # SHAP Feature Attribution
    st.subheader("Local SHAP Attribution: What Factors Drive This Prediction?")
    st.caption("Game-theoretic Shapley values decomposing the model's prediction into positive risk drivers and protective factors.")
    
    diag_left, diag_right = st.columns([3, 2])
    
    with diag_left:
        explanation = explain_single_customer(explainer, X_cohort[cust_row_idx], feature_names, top_n=5)
        drivers = explanation["risk_drivers"] + explanation["protective_factors"]
        df_bars = pd.DataFrame(drivers).sort_values(by="SHAP_Value", ascending=True)
        
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        bar_colors = ["#cf222e" if v > 0 else "#0969da" for v in df_bars["SHAP_Value"]]
        ax.barh(df_bars["Feature"], df_bars["SHAP_Value"], color=bar_colors, edgecolor="black", alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", lw=1.2)
        ax.set_xlabel("SHAP Impact on Log-Odds of Churn", fontweight="bold")
        ax.set_title(f"SHAP Diagnosis: Customer {selected_id} (Red = Pushes Risk UP | Blue = Pulls Risk DOWN)", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        
    with diag_right:
        st.markdown("#### Recommended Retention Playbook")
        playbook_res = generate_retention_playbook(
            explanation["risk_drivers"],
            churn_prob=prob_val,
            mrr=cust_data["mrr"],
            health_score=health_val
        )
        
        for idx, pb in enumerate(playbook_res["recommended_playbooks"], 1):
            st.markdown(f"**Playbook {idx}: {pb['playbook']}**")
            st.markdown(f"- **Urgency:** `{pb['urgency']}`")
            st.markdown(f"- **Channel:** `{pb['channel']}`")
            st.markdown(f"- **Action:** {pb['action']}")
            st.markdown("---")


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE & EVALUATION (HELD-OUT TEST SET)
# ==============================================================================
elif page == "📈 Model Performance & Evaluation":
    st.title("📈 Model Evaluation & Validation Performance")
    st.markdown("""
    This section evaluates the trained **Champion LightGBM** model strictly on the **held-out chronological test dataset** 
    (`2025-10-01` to `2025-11-01`). This partition was completely isolated during feature pipeline fitting, model training, 
    and hyperparameter selection to ensure unbiased generalization metrics.
    """)
    
    if test_eval_data is None:
        st.error("Test evaluation cache not found. Please run scripts/cache_test_evaluation.py.")
        st.stop()
        
    y_test = test_eval_data["y_test"]
    y_prob = test_eval_data["y_prob"]
    test_size = test_eval_data["test_size"]
    actual_churns = test_eval_data["actual_churn_count"]
    roc_auc_val = test_eval_data["roc_auc"]
    pr_auc_val = test_eval_data["pr_auc"]
    base_churn_rate = (actual_churns / test_size) * 100
    
    # Dataset Summary Metrics
    st.markdown("### 1. Test Dataset Characteristics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Held-Out Test Set Size", f"{test_size:,} records")
    m2.metric("Actual Churn Cases", f"{actual_churns:,} accounts")
    m3.metric("Observed Churn Rate", f"{base_churn_rate:.2f}%")
    m4.metric("Temporal Window", "Oct 2025 – Nov 2025")
    
    st.markdown("---")
    
    # Discrimination Summary
    st.markdown("### 2. Discrimination Performance (Threshold-Independent)")
    d1, d2 = st.columns(2)
    d1.metric("ROC-AUC Score", f"{roc_auc_val:.4f}", "Measures overall ranking separation")
    d2.metric("PR-AUC / Average Precision", f"{pr_auc_val:.4f}", f"{pr_auc_val / (base_churn_rate/100):.1f}x Lift over Random", delta_color="normal")
    
    st.markdown("""
    <div class="callout-info">
        <strong>Understanding Discrimination Metrics:</strong><br>
        • <strong>ROC-AUC (Receiver Operating Characteristic Area):</strong> Measures how well the model separates churners from non-churners across all possible thresholds. A score of 0.5 represents random guessing; 1.0 represents a perfect classifier.<br>
        • <strong>PR-AUC (Precision-Recall Area):</strong> The true gold standard for rare-event imbalance (~1.65% churn). At <strong>0.0698</strong>, our model delivers <strong>4.2x higher precision</strong> than random guessing.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive Threshold Tuning
    st.markdown("### 3. Threshold-Dependent Operational Metrics & Confusion Matrix")
    st.markdown("Adjust the classification threshold slider to see how decision boundaries affect Precision, Recall, and the Confusion Matrix:")
    
    threshold = st.slider(
        "Classification Decision Threshold (tau)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Customers with predicted probability >= threshold are classified as predicted churners."
    )
    
    preds_thresh = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds_thresh)
    tn, fp, fn, tp = cm.ravel()
    
    prec = precision_score(y_test, preds_thresh, zero_division=0)
    rec = recall_score(y_test, preds_thresh, zero_division=0)
    f1 = f1_score(y_test, preds_thresh, zero_division=0)
    acc = accuracy_score(y_test, preds_thresh)
    
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    tcol1.metric("Precision", f"{prec * 100:.2f}%", help="Of customers flagged by model, how many actually churned?")
    tcol2.metric("Recall", f"{rec * 100:.2f}%", help="Of customers who actually churned, how many were identified?")
    tcol3.metric("F1-Score", f"{f1:.4f}", help="Harmonic balance between precision and recall")
    tcol4.metric("Accuracy (Secondary)", f"{acc * 100:.2f}%", help="Displayed for completeness; not the primary metric under severe class imbalance")
    
    # Confusion Matrix Presentation
    col_cm_table, col_cm_desc = st.columns([3, 2])
    
    with col_cm_table:
        st.subheader("Confusion Matrix on Test Set")
        cm_display = pd.DataFrame(
            [
                [f"True Negatives (TN): {tn:,}", f"False Positives (FP): {fp:,}"],
                [f"False Negatives (FN): {fn:,}", f"True Positives (TP): {tp:,}"]
            ],
            index=["Actually Retained", "Actually Churned"],
            columns=["Predicted Retained", "Predicted Churn"]
        )
        st.dataframe(cm_display, width=500)
        
    with col_cm_desc:
        st.markdown(f"""
        **Operational Interpretation at Threshold `{threshold:.2f}`:**
        - **True Positives (TP = {tp:,}):** Correctly flagged accounts who churned. Successfully queued for CS intervention.
        - **False Positives (FP = {fp:,}):** False alarms—accounts that were flagged but stayed. Drives CS outreach cost.
        - **False Negatives (FN = {fn:,}):** Missed churners—accounts that cancelled without alert.
        - **True Negatives (TN = {tn:,}):** Retained accounts correctly left alone.
        """)
        
    st.markdown("---")
    
    # Visual Curves from Reports
    st.subheader("4. Pre-Computed Test Curves & Trade-Offs")
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        fig_roc = project_root / "reports" / "figures" / "08_roc_and_pr_curves.png"
        if fig_roc.exists():
            st.image(str(fig_roc), caption="Validation ROC and Precision-Recall Curves", width=500)
    with rcol2:
        fig_thresh = project_root / "reports" / "figures" / "09_threshold_tradeoffs.png"
        if fig_thresh.exists():
            st.image(str(fig_thresh), caption="Precision-Recall Threshold Trade-Offs", width=500)


# ==============================================================================
# PAGE 5: WHAT-IF SIMULATOR
# ==============================================================================
elif page == "🧪 What-If Simulator":
    st.title("🧪 Counterfactual What-If Simulator")
    st.markdown("Explore how prospective customer behavioral changes influence model predictions and health scores.")
    
    # Causal inference safeguard
    st.markdown("""
    <div class="causal-box">
        <strong>⚠️ Causal Inference Disclosure:</strong><br>
        This simulator evaluates <em>associational model responsiveness</em> P(Y | X_modified), not proven 
        <em>causal effect</em> P(Y | do(X)). For instance, while reducing payment failures lowers predicted churn probability, 
        artificially inflating login counts or feature scores without resolving root customer dissatisfaction does not guarantee real-world retention.
    </div>
    """, unsafe_allow_html=True)
    
    sim_account_list = cohort_df.sort_values(by="expected_mrr_at_risk", ascending=False)["user_id"].tolist()
    target_cust_id = st.selectbox("Select Customer to Simulate:", sim_account_list[:50])
    
    base_record = cohort_df[cohort_df["user_id"] == target_cust_id].iloc[0]
    
    st.markdown("### Adjust Customer Metrics")
    scol1, scol2, scol3 = st.columns(3)
    
    with scol1:
        new_feat_usage = st.slider("Feature Usage Score", 0.0, 100.0, float(base_record["feature_usage_score"]), step=1.0)
        new_sess = st.slider("Monthly Sessions", 0, 150, int(base_record["sessions"]))
        
    with scol2:
        new_supp_tickets = st.slider("Support Tickets", 0, 15, int(base_record["support_tickets"]))
        new_pay_failures = st.slider("Payment Failures", 0, 5, int(base_record["payment_failures"]))
        
    with scol3:
        new_nps_val = st.slider("NPS Score", 0, 10, int(base_record["nps_score"]))
        new_active_seats = st.slider("Active Seats", 0, 25, int(base_record["active_seats"]))
        
    if st.button("Run Simulation", type="primary"):
        mods = {
            "feature_usage_score": new_feat_usage,
            "sessions": new_sess,
            "support_tickets": new_supp_tickets,
            "payment_failures": new_pay_failures,
            "nps_score": new_nps_val,
            "active_seats": new_active_seats
        }
        
        sim_out = simulate_what_if_counterfactual(
            champion_model,
            pipeline,
            base_record,
            mods
        )
        
        st.markdown("---")
        st.subheader("Simulation Results")
        
        sr1, sr2, sr3, sr4 = st.columns(4)
        sr1.metric("Baseline Churn Probability", f"{sim_out['base_churn_probability']*100:.1f}%")
        sr2.metric("Simulated Churn Probability", f"{sim_out['simulated_churn_probability']*100:.1f}%", f"{sim_out['probability_delta']*100:+.1f}%", delta_color="inverse")
        sr3.metric("Baseline Health Score", f"{sim_out['base_health_score']:.1f}")
        sr4.metric("Simulated Health Score", f"{sim_out['simulated_health_score']:.1f}", f"{sim_out['health_score_delta']:+.1f}")


# ==============================================================================
# PAGE 6: SYSTEM ARCHITECTURE & DOCUMENTATION
# ==============================================================================
elif page == "📋 System Architecture & Docs":
    st.title("📋 Technical Architecture & Governance")
    st.markdown("Overview of the engineering pipeline, data leakage safeguards, and model tournament results.")
    
    doc_tabs = st.tabs([
        "Pipeline Architecture",
        "Data Hygiene & Safeguards",
        "Model Tournament Results",
        "Explainability & SHAP Axioms"
    ])
    
    with doc_tabs[0]:
        st.markdown("""
        ### End-to-End Pipeline
        1. **Ingestion & Validation:** Ingests monthly panel data, validates data types, and checks schema hygiene.
        2. **Eligibility Filtering:** Excludes absorbing churn states (`churned == 1`) and right-censored months (`2025-12-01`).
        3. **Dynamic Feature Engineering:** Computes month-over-month deltas, 3-month rolling aggregations, ratios, and domain risk flags.
        4. **Leakage-Safe Preprocessing:** Preprocessing transformers fit exclusively on training data and transform forward splits.
        5. **Inference & Serving:** Serves real-time predictions via FastAPI and Streamlit.
        """)
        
    with doc_tabs[1]:
        st.markdown("""
        ### Data Hygiene & Safeguards
        - **Strict Out-of-Time Splitting:** Preserves customer identity and avoids future-data leakage by splitting chronologically.
        - **Multicollinearity Elimination:** `monthly_price` and `mrr` have $r = 1.000$; `monthly_price` was purged to prevent singularity.
        - **Cold-Start Handling:** New accounts (`tenure_month == 0`) receive neutral baseline imputations for delta features.
        """)
        
    with doc_tabs[2]:
        st.markdown("""
        ### Validation Tournament Leaderboard
        | Model Family | PR-AUC | ROC-AUC | Precision@Top-50 | Lift@Top-50 |
        |---|---|---|---|---|
        | **Tuned LightGBM (Champion)** | **0.0783** | **0.8066** | **18.0%** | **10.7x** |
        | Balanced Logistic Regression | 0.0712 | 0.8070 | 16.0% | 9.5x |
        | Random Forest | 0.0576 | 0.7788 | 8.0% | 4.8x |
        | Decision Tree | 0.0477 | 0.7101 | 0.0% | 0.0x |
        """)
        
    with doc_tabs[3]:
        st.markdown(r"""
        ### Game-Theoretic Explainability (TreeSHAP)
        The model satisfies all 4 mathematical Shapley axioms:
        - **Efficiency (Additivity):** $\sum_{j=1}^{30} \phi_j(x) + \phi_0 = f(x)$ verified to $< 10^{-5}$ precision.
        - **Symmetry:** Identical contributors receive equal attribution.
        - **Dummy Player:** Non-contributing features receive 0 attribution.
        - **Additivity:** Ensemble attributions match the sum of individual tree attributions.
        """)
