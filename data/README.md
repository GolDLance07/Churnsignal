---
language: en
tags:
  - synthetic-data
  - churn-prediction
  - tabular
  - time-series
license: other
pretty_name: Synthetic B2B SaaS Churn & Revenue (Sample)
---

## Synthetic B2B SaaS Churn & Revenue (Sample)

This repository contains a **free sample** of a **100% synthetic** dataset for:
- churn prediction
- revenue / LTV modeling
- segmentation
- cohort analysis

### What’s inside
- `users.*` — 1 row per customer
- `user_monthly.*` — 1 row per customer-month (with churn labels)
- `schema.json` — data dictionary
- `meta.json` — stats + validation + legal fields (`data_origin=100% synthetic`)

### Labels
- **`churned_next_month`**: 1 if churn happens next month (recommended for ML)
- `churned`: churn event at end of the current month

### Load with pandas

```python
import pandas as pd

users = pd.read_parquet("users.parquet")
monthly = pd.read_parquet("user_monthly.parquet")
```

### Paid full dataset (100k users, 24 months)

Get the full release (+ baseline code) on Gumroad:
- **Gumroad link**: `gorchakov.gumroad.com/l/zhawn`


