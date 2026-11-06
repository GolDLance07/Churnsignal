# ChurnSignal — Technical Architecture & Machine Learning Design

This document provides a concise, developer-level overview of the ChurnSignal retention intelligence architecture, modeling methodology, and evaluation safeguards.

---

## 1. Problem Formulation & Data Integrity

### 1.1 Objective
Predict whether an active customer $i$ at month $T$ will cancel their subscription during month $T+1$:
$$y_i^{(T)} = \mathbb{I}(\text{customer } i \text{ churns during month } T+1)$$
- **Target Variable**: `churned_next_month` (binary classification).
- **Unit of Analysis**: Customer-month snapshot.

### 1.2 Data Mechanics & Safeguards
- **Absorbing Terminal State**: In SaaS subscription datasets, churn is absorbing. Once a customer has churned (`churned == 1`), they exit the active customer pool and do not generate future records. Records where `churned == 1` at month $T$ are excluded from prediction candidate pools.
- **Right-Censoring**: The final month in the dataset (`2025-12-01`) has artificially zeroed labels because the subsequent month ($T+1$) was not recorded. This month is permanently excluded from all evaluation splits.
- **Multicollinearity Resolution**: `monthly_price` and `mrr` exhibit an exact Pearson correlation ($r = 1.000$). `monthly_price` was purged during preprocessing to prevent matrix collinearity.

---

## 2. Chronological Validation Architecture

### 2.1 The Panel Data Leakage Problem
Customer data consists of longitudinal panels (repeated measurements on 5,000 unique customers over 24 calendar months). Standard random shuffling (e.g. `train_test_split(shuffle=True)`) leaks customer identity across splits and trains models on future behaviors to predict past churn.

### 2.2 Strict Out-of-Time (OOT) Partitioning
Data is split strictly across calendar time:
- **Training Set** (`2024-01-01` to `2025-06-01`): 54,139 records (18 months)
- **Validation Set** (`2025-07-01` to `2025-09-01`): 12,610 records (3 months)
- **Held-Out Test Set** (`2025-10-01` to `2025-11-01`): 7,931 records (2 months)

All transformers (scalers, encoders) fit exclusively on the training partition and transform forward splits without target or temporal look-ahead.

---

## 3. Feature Engineering Pipeline

The raw schema is expanded into 30 model input features via `src/features/engineer.py` and `src/features/pipeline.py`:

1. **Velocity / Momentum Features**:
   $$\Delta_{\text{metric}} = \text{Metric}_T - \text{Metric}_{T-1}$$
   Computed for sessions, feature usage, NPS, support tickets, payment failures, and active seats.
2. **Engagement & Utilization Ratios**:
   - `session_ratio` and `usage_ratio`: relative change with $+1$ Laplace smoothing to avoid zero-division.
   - `seat_utilization`: average monthly sessions per active seat.
3. **Rolling Historical Windows (Past 3 Months)**:
   Aggregations strictly bounded by $t \le T$ using `rolling(window=3, min_periods=1)`:
   - `rolling_mean_sessions_3m`
   - `rolling_mean_usage_3m`
   - `rolling_max_payment_failures_3m`
   - `rolling_sum_support_tickets_3m`
4. **Domain Risk Flags**:
   - `is_zero_seats`: paying account with 0 active seats.
   - `is_nps_detractor`: NPS rating $\le 6$.
   - `has_payment_failure`: billing failure flag.
   - `is_new_customer`: cold-start accounts with tenure 0 (imputed with neutral 0.0 delta baselines).

---

## 4. Modeling & Tournament Selection

Four model families were trained under identical conditions and evaluated on the untouched validation set:

| Architecture | PR-AUC | ROC-AUC | Precision@Top-50 | Lift@Top-50 | Brier Score | Status |
|---|---|---|---|---|---|---|
| **Tuned LightGBM** | **0.0783** | **0.8066** | **18.0%** | **10.7x** | **0.0163** | 🏆 **Champion** |
| Balanced Logistic Regression | 0.0712 | 0.8070 | 16.0% | 9.5x | 0.0168 | Linear Baseline |
| Subsampled Random Forest | 0.0576 | 0.7788 | 8.0% | 4.8x | 0.0175 | Bagged Ensemble |
| Decision Tree Classifier | 0.0477 | 0.7101 | 0.0% | 0.0x | 0.0210 | Single Tree |

### Champion Hyperparameters
- `n_estimators`: 100
- `learning_rate`: 0.03 (conservative shrinkage)
- `max_depth`: 3, `num_leaves`: 7 (guards against rare-class leaf fragmentation)
- `min_child_samples`: 100 (requires substantial evidence per leaf split)
- `class_weight`: balanced

---

## 5. Explainability Engine (TreeSHAP)

To convert predictions into actionable Customer Success workflows, we implement `shap.TreeExplainer`:
$$f(x) = \phi_0 + \sum_{j=1}^{30} \phi_j(x)$$
- **Efficiency Axiom**: $\sum \phi_j(x) + \phi_0 = f(x)$ verified to $< 10^{-5}$ floating-point precision against raw model margins.
- **Local Attribution**: Separates risk factors ($\phi_j > 0$, red bars) from protective retention anchors ($\phi_j < 0$, blue bars).

---

## 6. Retention Intelligence & Decision Architecture

1. **Customer Health Score (0–100)**:
   Bounded operational vitality metric:
   $$\text{Health} = 0.35 \cdot \text{Engagement} + 0.25 \cdot \text{Sentiment} + 0.20 \cdot \text{Friction} + 0.20 \cdot \text{Billing}$$
2. **Expected Revenue at Risk**:
   $$\text{Expected MRR at Risk} = \text{MRR} \times P(\text{churn})$$
   Ensures finite CS outreach bandwidth targets maximum financial exposure rather than low-value accounts with high churn likelihood.
3. **Counterfactual What-If Simulation**:
   Calculates model responsiveness under modified inputs $P(Y \mid X_{\text{mod}})$. Explicitly distinguished from proven causal intervention $P(Y \mid \text{do}(X))$.

