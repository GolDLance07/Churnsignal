# ChurnSignal — Technical Specification

## 1. System Overview & Problem Formulation
**ChurnSignal** is an enterprise-grade SaaS Customer Retention Intelligence platform designed to predict customer churn, explain underlying behavioral drivers, calculate customer health scores, simulate "what-if" interventions, and provide actionable operational playbooks for Customer Success teams.

### Formal Prediction Problem
Given all historical behavioral, financial, and support data observed for an active customer account during calendar month $T$ (observation window), predict whether the customer will experience a churn event during calendar month $T+1$ (prediction window):

$$\hat{y}_{i, T+1} = P(\text{churned\_next\_month}_{i, T} = 1 \mid \mathbf{X}_{i, \le T})$$

- **Prediction Unit**: One customer account snapshot at month $T$.
- **Target Variable**: `churned_next_month` $\in \{0, 1\}$.
- **Eligibility Filter**: Only active customers in month $T$ (`churned == 0`) are eligible for prediction. Accounts that already reached the absorbing churn state (`churned == 1`) in month $T$ are excluded from active scoring.

---

## 2. Dataset Specification
- **Source**: `arti199919/synthetic-saas-churn-sample` (`user_monthly.parquet`).
- **Scale**: 79,817 customer-month records, 5,000 unique customers, 24 calendar months (`2024-01-01` to `2025-12-01`).
- **Feature Categories**:
  - *Account Scale & Commercials*: `plan_type` (Starter, Pro, Business, Enterprise), `mrr` (replaces collinear `monthly_price`), `active_seats`.
  - *Usage & Engagement*: `tenure_month`, `sessions`, `feature_usage_score`.
  - *Health & Friction*: `support_tickets`, `payment_failures`, `nps_score`, `product_incident`.
  - *Labels*: `churned` (current month exit), `churned_next_month` (next month exit).

---

## 3. Full-Stack Platform Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      ChurnSignal Frontend (Streamlit)                           │
├─────────────────┬───────────────────┬───────────────────┬───────────────────────┤
│ Customer        │ Churn Prediction  │ Explainability    │ What-If Simulator     │
│ Overview & KPIs │ & Risk Tiers      │ & SHAP Driver UI  │ & Model Evaluation    │
└─────────────────┴───────────────────┴───────────────────┴───────────────────────┘
                                         ▲
                                         │ REST API / Direct Python Service
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Backend Service (FastAPI)                             │
├─────────────────┬───────────────────┬───────────────────┬───────────────────────┤
│ Data Endpoints  │ Batch Inference   │ Explainability    │ Health Score Engine   │
│ & Filtering     │ & Risk Scoring    │ (SHAP Attrib.)    │ & Retention Playbook  │
└─────────────────┴───────────────────┴───────────────────┴───────────────────────┘
                                         ▲
                                         │ Internal Python Interface
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Machine Learning Engine (Python)                         │
├─────────────────┬───────────────────┬───────────────────┬───────────────────────┤
│ Feature Eng.    │ OOT Chronological │ LightGBM Classifier│ Probability          │
│ & Transformers  │ Splitter          │ (Tuned Champion)  │ Calibration Engine    │
└─────────────────┴───────────────────┴───────────────────┴───────────────────────┘
```

---

## 4. Key Constraints & Data Integrity Guardrails
1. **No Target Leakage**: Current month `churned` is barred as an input feature for `churned_next_month`.
2. **Right-Truncation Exclusion**: Month `2025-12-01` has artificially zeroed forward targets due to dataset termination and is permanently excluded from all evaluation splits.
3. **Strict Out-of-Time Partitioning**: Chronological split isolates train (`2024-01` to `2025-06`), validation (`2025-07` to `2025-09`), and test (`2025-10` to `2025-11`).
4. **Collinearity Elimination**: `monthly_price` is dropped to prevent exact collinearity with `mrr`.

