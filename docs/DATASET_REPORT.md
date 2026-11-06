# ChurnSignal — Dataset Reconnaissance & Quality Report

## 1. Executive Summary
Before model development, we conducted exploratory reconnaissance of the synthetic SaaS churn dataset (`arti199919/synthetic-saas-churn-sample`). Our primary objective was to understand the data-generating process, temporal structure, target semantics, and potential data leakage risks.

```
       79,817 Customer-Month Snapshots
┌────────────────────────────────────────────────────────┐
│ 5,000 Unique Customers | 24 Calendar Months (2024-2025)│
└────────────────────────────────────────────────────────┘
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
  Active Retention Cohort              Terminal Churn Cohort
 3,869 Customers Active             1,131 Customers Churned
 (Active through Dec 2025)          (Absorbing Terminal State)
```

---

## 2. Reconnaissance Checklist

| # | Task | Finding | Implication |
|---|---|---|---|
| **1** | **Shape** | 79,817 rows × 15 columns | Longitudinal panel structure (multiple snapshots per entity). |
| **2** | **Data Types** | 1 string (`user_id`), 1 datetime (`month`), 1 categorical (`plan_type`), 12 numeric | Clean formatting; categorical encoding required for `plan_type`. |
| **3** | **Missing Values** | 0 missing values across all columns; 0 duplicate `(user_id, month)` pairs | No missing imputation needed; compound business key is unique. |
| **4** | **Entities** | 5,000 unique users, 24 continuous calendar months (Jan 2024 to Dec 2025) | Exactly 2 full calendar years of SaaS operational history. |
| **5** | **Multiplicity** | Mean = 15.96 records/user (Min: 1, Max: 24). 98.5% of users have >1 month. | Repeated measures require strict temporal (OOT) evaluation. |
| **6** | **Target Imbalance** | `churned`: 1.42% (1,131 rows); `churned_next_month`: 1.32% (1,056 rows) | Severe class imbalance (~1.3% positive class). Accuracy is deceptive. |
| **7** | **Terminal State** | When `churned == 1`, `churned_next_month` is always 0. Churn is terminal. | Once churned, the customer exits permanently. |
| **8** | **Active Seats** | `active_seats`: Integer count (0 to 197). 2,546 rows (3.19%) have 0 seats. | Inactive teams with 0 seats represent a primary churn risk flag. |
| **9** | **Collinearity** | `monthly_price` and `mrr` have $r = 1.000$ (100% identical duplicates). | Perfect multicollinearity; `monthly_price` dropped in preprocessing. |
| **10** | **Leakage Hazards** | Contemporaneous `churned` feature, Dec 2025 truncated zeros, MRR duplication. | Critical traps prevented by eligibility filters and OOT splitting. |
| **11** | **Chronological Split** | 24 continuous months, zero skipped months. | Enables strict train/validation/test chronological holdouts. |

---

## 3. Potential Target Leakage Risks & Safeguards

1. **Contemporaneous `churned` Feature**: Using current-month `churned` to predict `churned_next_month` is circular because churn is terminal. The feature is barred from inputs.
2. **Right-Censored Dec 2025 Labels**: Because January 2026 was unobserved, all Dec 2025 rows were defaulted to 0. Dec 2025 is permanently excluded from all evaluation splits.
3. **Random Train/Test Splitting**: Random shuffling mixes past and future months of the same customer, causing identity memorization. All splitting is performed chronologically.

---

## 4. Chronological Validation Strategy

```
2024-01                                     2025-06 2025-07     2025-09 2025-10     2025-11 2025-12
┌──────────────────────────────────────────────────┐┌──────────────────┐┌──────────────────┐┌────────┐
│                   TRAIN SPLIT                    ││  VALIDATION SET  ││     TEST SET     ││EXCLUDE │
│            18 Months (55,208 records)            ││ 3 Mos (12,610 r) ││ 2 Mos (8,068 r)  ││Truncated│
│           716 Positive Targets (1.30%)           ││ 209 Pos (1.66%)  ││ 131 Pos (1.62%)  ││Zeros   │
└──────────────────────────────────────────────────┘└──────────────────┘└──────────────────┘└────────┘
```

