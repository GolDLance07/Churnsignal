# ChurnSignal — Customer Retention Intelligence

## Introduction

In subscription businesses, **customer churn** occurs when a customer cancels their subscription or stops paying for a service. Retaining existing customers is typically much less expensive than acquiring new ones, and even small reductions in churn can significantly improve a company's revenue and long-term health.

**ChurnSignal helps businesses identify customers who may stop using their service. It uses patterns in customer activity and behavior to estimate churn risk, helping businesses decide which customers may need attention.**

Rather than waiting for a customer to cancel, ChurnSignal provides an early warning system 30 days in advance, highlights the main factors contributing to each customer's risk, and helps customer success teams prioritize which accounts to contact first.

---

## Features

- **Forward Churn Prediction:** Estimates whether an active customer is likely to cancel their subscription during the upcoming month (30-day forecast).
- **Churn Probability Scores:** Clear, calibrated percentages (0% to 100%) indicating how likely a customer is to churn.
- **Customer Risk Categorization:** Automatically groups accounts into **High Risk** ($\ge 35\%$), **Medium Risk** ($15\% - 35\%$), or **Low Risk** ($< 15\%$) categories.
- **Customer Health Scoring:** An intuitive, transparent operational score (0 to 100) based on product usage, session frequency, Net Promoter Score (NPS), support tickets, and billing status.
- **Transparent Factor Attribution (SHAP):** Visual waterfall explanations showing which specific factors are pushing a customer's churn risk up (such as payment failures or declining usage) and which factors are keeping them retained (such as long tenure).
- **Financial Prioritization:** Computes **Expected Revenue at Risk** ($\text{Monthly Price} \times \text{Churn Probability}$) so teams can focus limited outreach time on the most valuable accounts.
- **Interactive Retention Dashboard:** A modern Streamlit application providing portfolio-level overview metrics, filterable customer lists, and single-customer diagnostic views.
- **Counterfactual What-If Simulator:** Allows users to adjust customer metrics (such as resolving support tickets or boosting usage) and observe how the model's prediction changes.
- **Held-Out Model Evaluation:** A dedicated performance evaluation section featuring interactive decision threshold adjustments, confusion matrix breakdowns, and plain-English metric explanations.

---

## How It Works

ChurnSignal follows a straightforward five-step workflow:

1. **Customer information is provided:** Monthly customer activity data (such as login sessions, feature usage, submitted support tickets, billing failures, and NPS ratings) is collected.
2. **The application processes the information:** Data is checked for quality, and key trends are calculated (such as month-over-month usage drops and 3-month rolling averages).
3. **The trained ML model estimates churn probability:** A trained gradient-boosted decision tree model analyzes the customer's behavioral patterns against historical churn patterns.
4. **The application displays the prediction and risk category:** The platform presents the estimated churn probability, the assigned risk category (High, Medium, or Low), the operational health score, and the primary risk factors.
5. **The business can use the result to prioritize customer retention efforts:** Customer Success and account management teams use the predictions and recommended playbooks to reach out to high-exposure accounts before cancellations occur.

---

## Machine Learning

### Problem Type
This is a **supervised binary classification** problem. The model learns from historical customer records where the outcome (whether the customer churned in the following month) is known.

### What the Model Predicts
The model predicts the variable `churned_next_month`:
- `1`: The customer will cancel their subscription during the next calendar month.
- `0`: The customer will remain active during the next calendar month.

### Model Architecture
The production model is a regularized **LightGBM (Light Gradient Boosted Machine)** classifier. It uses an ensemble of 100 shallow decision trees (`max_depth=3`, `num_leaves=7`) with balanced class weighting and a minimum of 100 observations per leaf split (`min_child_samples=100`) to prevent overfitting on rare churn events.

### What the Prediction Percentage Means
The predicted percentage represents the model's estimated statistical probability that a customer will cancel within the next 30 days. For example, a churn probability of **78%** means that customers with similar behavioral trends historically cancelled approximately 78% of the time.

### Evaluation Metrics
Because customer churn is a rare event (only about 1.65% of customer-months end in churn in the test dataset), standard evaluation requires metrics that account for class imbalance:

- **ROC-AUC (Receiver Operating Characteristic Area):** Measures how effectively the model ranks churners higher than non-churners across all possible decision thresholds.
- **PR-AUC (Precision-Recall Area / Average Precision):** The primary metric for rare-event problems. It measures the balance between catching true churners and avoiding false alarms.
- **Precision:** Of the customers flagged as likely to churn, what percentage actually churned?
- **Recall:** Of all customers who actually churned, what percentage did the model correctly identify?
- **F1-Score:** The harmonic average of precision and recall.

### Why Accuracy Alone Is Not Sufficient
In an imbalanced dataset where only 1.65% of customers churn, a naive model that predicts "Nobody Churns" achieves **98.35% accuracy**. However, this naive model identifies zero churners, provides zero lead time, and saves zero revenue. For this reason, ChurnSignal prioritizes PR-AUC, Recall, Precision, and financial revenue at risk over raw accuracy.

---

## Dataset

- **Source:** The synthetic SaaS customer churn dataset `arti199919/synthetic-saas-churn-sample` (hosted on Hugging Face).
- **Scale:** 79,817 monthly records across 5,000 unique customer accounts over 24 consecutive calendar months (January 2024 to December 2025).
- **Features:** Subscription tier (`plan_type`), monthly revenue (`mrr`), account tenure (`tenure_month`), active seats (`active_seats`), login sessions (`sessions`), product feature usage index (`feature_usage_score`), support tickets filed (`support_tickets`), billing failures (`payment_failures`), Net Promoter Score (`nps_score`), and platform incident involvement (`product_incident`).

### Synthetic Data Disclosure
**This dataset is synthetic.** It was generated using mathematical distributions to simulate SaaS customer behavior. Its findings should be understood as an educational and technical demonstration, and should not be interpreted as empirical evidence about real-world software businesses.

---

## Tech Stack

- **Programming Language:** Python 3.14+
- **Data Manipulation & Storage:** pandas, numpy, pyarrow
- **Machine Learning:** scikit-learn, LightGBM
- **Explainability:** SHAP (TreeExplainer)
- **Interactive UI:** Streamlit
- **REST API:** FastAPI, Uvicorn, Pydantic
- **Plotting & Visualization:** matplotlib, seaborn
- **Automated Testing:** pytest

---

## Project Structure

```
churnrate/
├── artifacts/                          # Serialized models and precomputed data
│   ├── champion_model.joblib           # Trained LightGBM champion classifier
│   ├── feature_pipeline.joblib         # Fitted scikit-learn ColumnTransformer
│   ├── scored_active_cohort.parquet    # Cached active customer scores
│   └── test_evaluation.npz             # Cached held-out test predictions
├── data/                               # Local dataset storage
│   ├── user_monthly.parquet            # Panel dataset (79,817 rows)
│   └── users.parquet                   # Customer profile metadata
├── docs/                               # Developer and technical documentation
│   ├── ARCHITECTURE.md                 # Detailed ML design, split logic & pipeline
│   ├── DECISIONS.md                    # Architecture Decision Records (ADRs 001-010)
│   ├── SPECIFICATION.md                # System specification & prediction bounds
│   └── DATASET_REPORT.md               # Data reconnaissance & hygiene findings
├── notebooks/                          # Step-by-step reproducible notebooks (01 to 08)
├── reports/figures/                    # 15 precomputed publication-grade figures
├── scripts/                            # Operational utility scripts
│   ├── cache_production_cohort.py     # Pre-scores active customer cohort
│   └── cache_test_evaluation.py       # Computes held-out test metrics
├── src/                                # Modular production Python package
│   ├── data/loader.py                  # Ingestion & eligibility filtering
│   ├── features/engineer.py            # Velocity deltas, rolling windows & flags
│   ├── features/pipeline.py            # Chronological splitting & preprocessing
│   ├── models/baselines.py             # Logistic regression baseline
│   ├── models/comparison.py            # Head-to-head tournament engine
│   ├── explainability/explainer.py     # TreeSHAP factor attribution engine
│   ├── explainability/retention.py     # Health scoring & What-If simulation
│   └── service/app.py                  # Production FastAPI REST application
├── tests/                              # Automated unit & integration tests (35 tests)
├── dashboard.py                        # Streamlit web application
├── requirements.txt                    # Project dependencies
├── pytest.ini                          # Test configuration
├── LICENSE                             # MIT License
└── README.md                           # Project documentation
```

---

## Installation and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/churnsignal.git
cd churnsignal
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to explore the interactive application.

### 5. Run the FastAPI REST Server
```bash
uvicorn src.service.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 6. Run the Automated Test Suite
```bash
pytest tests/
```

---

## Model Performance

The model was evaluated strictly on the **held-out chronological test dataset** (`2025-10-01` to `2025-11-01`). This partition was completely isolated during training, preprocessing fitting, and model selection.

### Held-Out Test Set Summary
- **Test Set Size:** 7,931 customer-month records
- **Actual Churn Cases:** 131 accounts (1.65% base rate)
- **ROC-AUC:** **0.7978** (demonstrates strong ranking ability)
- **PR-AUC / Average Precision:** **0.0698** (delivers a **4.2x lift** over random guessing)

### Performance Across Decision Thresholds

| Decision Threshold ($\tau$) | Precision | Recall | F1-Score | Accuracy | Operational Use Case |
|---|---|---|---|---|---|
| **0.35** | 3.11% | **92.37%** | 0.0602 | 52.34% | **Broad Safety Net:** Flags almost all churners (catches 121 of 131), but includes many false alarms. |
| **0.50** | 3.79% | **77.10%** | 0.0722 | 67.26% | **Balanced Default:** Identifies 101 of 131 churners while reducing false alarms. |
| **0.60** | **4.62%** | 65.65% | **0.0863** | 77.04% | **High-Efficiency Outreach:** Focuses only on high-confidence churn signals. |

### Explanation of Key Metrics in Simple Terms
- **Precision (3.79% at 0.50 threshold):** Of every 100 accounts flagged as likely to churn, approximately 4 accounts were actual churners. In rare-event problems, this is 2.3x higher than random selection.
- **Recall (77.10% at 0.50 threshold):** The model successfully detected more than 3 out of every 4 customers who actually cancelled.
- **ROC-AUC (0.7978):** If you randomly pick one churner and one non-churner, the model assigns a higher churn probability to the churner approximately 80% of the time.
- **PR-AUC (0.0698):** Measures overall precision across all sensitivity levels, demonstrating consistent superiority over the random 1.65% baseline.

---

## Limitations

1. **Synthetic Data:** The dataset is artificially simulated. Real SaaS churn dynamics may involve external market shifts, competitive poaching, macroeconomic downturns, and complex organizational changes that are not present in synthetic data.
2. **Correlation vs. Causation:** The model predicts churn using observational correlations ($P(Y \mid X)$). It identifies features associated with churn, but it does **not** prove that a feature caused the churn.
3. **What-If Simulation Boundary:** The What-If Simulator measures how the mathematical model responds to hypothetical inputs. Artificially increasing a usage score does not prove that a real-world customer will be retained unless the root product value is addressed.
4. **Data Stationarity:** The model assumes that customer behavior patterns observed in 2024 and 2025 will continue into future periods without major unmodeled shifts in pricing or platform functionality.

---

## Future Improvements

- **Survival Analysis Modeling:** Implement Cox Proportional Hazards or Kaplan-Meier estimation to predict *when* a customer is expected to churn (time-to-event).
- **Natural Language Processing (NLP):** Incorporate sentiment analysis on customer support ticket descriptions and cancellation feedback text.
- **CRM & Dunning Webhook Integrations:** Connect the FastAPI endpoint to tools such as Stripe, HubSpot, or Salesforce to trigger automated retention workflows upon elevated risk detection.
- **Dynamic Threshold Optimization:** Automatically adjust decision thresholds per subscription tier to maximize net retained Annual Recurring Revenue (ARR).

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
