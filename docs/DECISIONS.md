# ChurnSignal — Machine Learning Architecture Decision Records (ADR)

This document records the foundational machine learning and architectural decisions made throughout the ChurnSignal platform, including the rationale, alternatives considered, and trade-offs.

---

### Decision Log Summary

| ADR # | Decision Title | Status | Date |
|---|---|---|---|
| **ADR-001** | Ingestion of `user_monthly.parquet` via Targeted File Selection | **Accepted** | 2026-11-06 |
| **ADR-002** | Selection of `churned_next_month` as the Primary Prediction Target | **Accepted** | 2026-11-06 |
| **ADR-003** | Exclusion of Right-Censored Month `2025-12-01` from Target Evaluation | **Accepted** | 2026-11-06 |
| **ADR-004** | Deduplication of Identical Collinear Features (`monthly_price` vs `mrr`) | **Accepted** | 2026-11-06 |
| **ADR-005** | Strict Out-Of-Time (OOT) Chronological Validation Splitting | **Accepted** | 2026-11-06 |
| **ADR-006** | Adoption of PR-AUC, Brier Score, and Precision@Top-K over Accuracy | **Accepted** | 2026-11-06 |
| **ADR-007** | Selection of Class-Weighted Logistic Regression as Baseline Benchmark | **Accepted** | 2026-11-06 |
| **ADR-008** | Selection of Regularized LightGBM as Production Champion Model | **Accepted** | 2026-11-06 |
| **ADR-009** | TreeSHAP for Game-Theoretic Feature Attribution & Local Explainability | **Accepted** | 2026-11-06 |
| **ADR-010** | Retention Intelligence: Expected Revenue at Risk, Health Score & Playbooks | **Accepted** | 2026-11-06 |

---

### ADR-010: Retention Intelligence: Expected Revenue at Risk, Health Score & Playbooks

- **Context**: Statistical models generate probabilities $P(\text{churn}) \in [0, 1]$, but business teams operate under finite capacity and disparate account contract values. Ranking accounts solely by raw churn probability leads to inefficient outreach (e.g., spending hours attempting to save a \$20/month account while neglecting a \$3,000/month Enterprise account with moderate risk). Furthermore, non-technical stakeholders require a bounded, intuitive measure of current account health.
- **Decision**: 
  1. Define **Expected Revenue at Risk**: $\text{Expected MRR at Risk} = \text{MRR} \times P(\text{churn})$ as the primary sorting and prioritization metric for Customer Success workflows.
  2. Implement a 4-pillar, bounded $[0, 100]$ **Customer Health Score** (Engagement 35%, Sentiment 25%, Service Friction 20%, Billing Hygiene 20%) to provide a transparent operational vitality benchmark.
  3. Map individual local SHAP risk drivers directly into automated, contextual **Retention Action Playbooks** (Billing & Dunning Safeguards, Engineering Escalations, Adoption Sprints).
  4. Implement a **Counterfactual What-If Simulator** with strict causal safeguards highlighting $P(Y \mid X_{\text{mod}}) \neq P(Y \mid \text{do}(X))$.
- **Rationale**:
  - Maximizes net retained ARR per unit of Customer Success operational capacity.
  - Cross-tabulates operational vitality (Health Score) against statistical exit likelihood (Churn Probability) via the Strategic 2x2 Retention Matrix.
- **Consequences**:
  - Directly feeds the FastAPI `/retention/prioritize`, `/retention/playbook/{id}`, and `/retention/simulate` endpoints, as well as the Streamlit retention dashboard.

---

### ADR-009: TreeSHAP for Game-Theoretic Feature Attribution & Local Explainability

- **Context**: Black-box tree models output raw probabilities that lack interpretability for Customer Success teams. Traditional feature importance (Gini / Mean Decrease in Impurity) suffers from cardinal bias, inconsistency across tree variations, and zero individual directionality (it cannot tell if a feature pushed churn risk UP or DOWN for a specific customer).
- **Decision**: Implement `shap.TreeExplainer` on the Champion LightGBM model to provide game-theoretic local and global feature attribution satisfying the 4 Shapley axioms (Efficiency, Symmetry, Dummy Player, Additivity).
- **Rationale**:
  - *Additivity Guarantee*: Local Shapley values sum exactly to the difference between the model's raw score and the baseline expected value: $\sum \phi_j(x) = f(x) - \mathbb{E}[f(X)]$ (verified to $< 10^{-5}$ precision).
  - *Actionable Playbooks*: Enables separating positive risk drivers (e.g. usage drop, payment failure) from protective factors (e.g. high tenure), mapping directly to proactive CS interventions.
- **Consequences**:
  - Powers both individual account diagnostic cards (`/explain/{customer_id}`) and global monitoring charts in the user interface.

---

### ADR-008: Selection of Regularized LightGBM as Production Champion Model

- **Context**: We benchmarked four distinct model architectures on the untouched Out-of-Time validation set: Balanced Logistic Regression, Decision Tree, Random Forest, and LightGBM.
- **Decision**: Crown **LightGBM** as the production Champion Model:
  - Architecture: 100 trees, learning rate 0.03, max depth 3, 7 leaves, `min_child_samples=100`, `class_weight='balanced'`.
- **Validation Tournament Results**:
  1. **LightGBM (Champion)**: **PR-AUC = 0.0783**, **Precision@50 = 18.0%**, **Lift@50 = 10.7x**
  2. **Logistic Regression (Baseline)**: PR-AUC = 0.0712, Precision@50 = 16.0%, Lift@50 = 9.5x
  3. **Random Forest**: PR-AUC = 0.0576, Precision@50 = 8.0%, Lift@50 = 4.8x
  4. **Decision Tree**: PR-AUC = 0.0477, Precision@50 = 0.0%, Lift@50 = 0.0x
- **Rationale**:
  - LightGBM achieves the highest discrimination (PR-AUC 0.0783) and delivers the highest business value for CS outreach (10.7x Lift at Top-50).
  - Heavy leaf regularization (`min_child_samples=100`) successfully prevented leaf fragmentation and overfitting in rare classes.
- **Consequences**:
  - `champion_model.joblib` serves predictions for the API and retention dashboard.

---

### ADR-007: Selection of Class-Weighted Logistic Regression as Baseline Benchmark

- **Context**: Before testing complex non-linear models (Decision Trees, Random Forests, LightGBM/XGBoost), an interpretable, fast benchmark is required to test the end-to-end pipeline and anchor performance.
- **Decision**: Train an L2-regularized `LogisticRegression(class_weight='balanced', solver='lbfgs')` on the standardized 30-feature matrix.
- **Rationale**:
  1. *Sanity Checking*: Linear weights allow immediate auditing of feature directionality (e.g. usage drop must have a positive weight for churn risk).
  2. *Performance Anchor*: Yields a strong baseline benchmark on the validation set: **ROC-AUC: 0.8070**, **PR-AUC: 0.0712** (4.3x higher than random chance).
  3. *Zero Overfitting*: L2 regularization prevents extreme weights and guarantees rapid convergence.
- **Consequences**:
  - Sets the minimum bar that all subsequent non-linear models must surpass to justify additional architectural complexity.

---

### ADR-001: Ingestion of `user_monthly.parquet` via Targeted File Selection

- **Context**: The Hugging Face dataset `arti199919/synthetic-saas-churn-sample` contains two parquet files in its root directory: `user_monthly.parquet` (time-series panel) and `users.parquet` (static metadata). Calling `load_dataset("arti199919/synthetic-saas-churn-sample")` without parameters causes Hugging Face to attempt combining both tables into a single schema, triggering a `CastError`.
- **Decision**: Explicitly target `user_monthly.parquet` via `data_files="user_monthly.parquet"` in `load_dataset()`, with a fallback to local cached parquet loading.
- **Consequences**:
  - *Positive*: Guarantees deterministic, reproducible schema loading without crashes.

---

### ADR-002: Selection of `churned_next_month` as the Primary Prediction Target

- **Context**: The dataset provides two labels: `churned` (churn occurring at the end of the current month $T$) and `churned_next_month` (churn occurring in month $T+1$).
- **Decision**: Formulate the ML problem to predict `churned_next_month` using features aggregated through month $T$.
- **Rationale**:
  1. *Actionability*: Predicting $T+1$ churn provides Customer Success teams with a 30-day intervention window.
  2. *Leakage Elimination*: Contemporaneous features (like support tickets or billing failures) might occur *after* the customer makes the cancellation decision during month $T$. Using month $T$ features to predict month $T+1$ churn guarantees strict temporal separation.
- **Consequences**:
  - The model functions strictly as an early warning alert system. Current-month `churned` is barred from feature matrices.

---

### ADR-003: Exclusion of Right-Censored Month `2025-12-01` from Target Evaluation

- **Context**: The dataset ends on `2025-12-01`. Because January 2026 was not recorded, all 3,931 accounts in December 2025 have `churned_next_month = 0`.
- **Decision**: Permanently exclude the month `2025-12-01` from training, validation, and test datasets when modeling `churned_next_month`.
- **Rationale**:
  - Setting unobserved future events to zero creates artificial false negatives (right-censoring bias). Training on these false negative labels would penalize the model for correctly recognizing high-risk churn patterns in December 2025.
- **Consequences**:
  - Usable chronological dataset spans 23 full months (`2024-01-01` to `2025-11-01`), yielding 75,886 cleanly labeled customer-month snapshots.

---

### ADR-004: Deduplication of Identical Collinear Features (`monthly_price` vs `mrr`)

- **Context**: Statistical inspection revealed that $\text{monthly\_price} - \text{mrr} = 0.0$ for all 79,817 rows ($r = 1.0000$).
- **Decision**: Keep `mrr` as the standard SaaS financial metric and drop `monthly_price` during the preprocessing stage.
- **Rationale**:
  - Perfect collinearity produces singular or ill-conditioned covariance matrices in linear and logistic regression models, causing extreme coefficient variance and uninterpretable weights.
- **Consequences**:
  - Eliminates redundancy while preserving 100% of the underlying pricing information.

---

### ADR-005: Strict Out-Of-Time (OOT) Chronological Validation Splitting

- **Context**: The dataset is panel data (5,000 customers with up to 24 monthly snapshots each).
- **Decision**: Reject random K-Fold cross-validation and random `train_test_split`. Implement an Out-Of-Time (OOT) chronological split:
  - **Train**: `2024-01-01` to `2025-06-01` (18 months, 55,208 records, 716 positive targets)
  - **Validation**: `2025-07-01` to `2025-09-01` (3 months, 12,610 records, 209 positive targets)
  - **Test (Holdout)**: `2025-10-01` to `2025-11-01` (2 months, 8,068 records, 131 positive targets)
- **Rationale**:
  - Random splitting introduces **customer identity leakage** (memorizing customer-specific characteristics across randomly distributed months) and **temporal look-ahead leakage** (using future behavior to predict past churn).
  - Out-of-time evaluation simulates real-world production deployment: training on past history and deploying on future unseen calendar months.
- **Consequences**:
  - Model metrics reflect true generalization performance rather than overfitted cross-validation optimism.

---

### ADR-006: Adoption of PR-AUC, Brier Score, and Precision@Top-K over Accuracy

- **Context**: Churn in this SaaS business is a rare event (~1.32% positive class).
- **Decision**: Discard overall classification accuracy as a decision metric. Adopt:
  1. **PR-AUC (Average Precision)**: Primary metric evaluating the trade-off between precision and recall across all thresholds.
  2. **ROC-AUC**: Secondary metric measuring ranking quality.
  3. **Precision@Top-$K$**: Business operational metric evaluating the proportion of actual churners within the top $K$ accounts flagged for intervention.
  4. **Brier Score / Calibration Error**: Reliability metric ensuring predicted probabilities accurately reflect actual churn frequencies.
- **Rationale**:
  - In a 98.7% negative dataset, a dummy model that predicts 0 everywhere achieves 98.7% accuracy while finding zero churners. PR-AUC and Precision@Top-$K$ align directly with business retention ROI.

