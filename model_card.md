# Model Card — Customer Churn Prediction Pipeline

## Model Details
- **Developer**: B.Tech Data Science Portfolio Project
- **Model Date**: October 2026
- **Model Version**: `1.0.0`
- **Model Type**: Gradient Boosted Decision Trees (`LightGBM` / shallow `XGBoost`) encapsulated in a Scikit-Learn `ColumnTransformer` Pipeline.
- **License**: MIT
- **Model Serialization**: Compressed Joblib artifact (`models/model.pkl`, size: ~140 KB).

---

## Intended Use
- **Primary Use Case**: Proactive detection of telecommunication customers at high risk of cancelling subscriptions within 30–60 days.
- **Intended Users**: Retention marketing teams, customer success managers, and automated CRM retention discount triggers.
- **Out-of-Scope Use Cases**:
  - Determining creditworthiness or loan eligibility.
  - Unfair or predatory dynamic pricing algorithms.
  - Punitive throttling of customer bandwidth based on churn likelihood.

---

## Factors & Feature Schema
The model uses strictly behavioral and contractual features, avoiding protected demographic characteristics (race, gender, age, marital status):

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `tenure` | Numerical | Lifetime duration of customer relationship in months (0–72). |
| `monthly_charges` | Numerical | Current recurring billing rate in USD. |
| `total_charges` | Numerical | Historical cumulative charges incurred by the customer. |
| `contract_type` | Categorical | Billing agreement: `Month-to-month`, `One year`, `Two year`. |
| `internet_service` | Categorical | Service medium: `Fiber optic`, `DSL`, `No`. |
| `payment_method` | Categorical | Billing channel: `Electronic check`, `Mailed check`, `Credit card`, `Bank transfer`. |
| `tech_support` | Categorical | Add-on technical support: `Yes`, `No`, `No internet service`. |
| `tenure_group` *(Engineered)* | Categorical | Discretized loyalty cohort: `0-12m`, `13-24m`, `25-48m`, `49-72m`. |
| `avg_monthly_spend` *(Engineered)* | Numerical | Derived spend velocity: `total_charges / (tenure + 1)`. |
| `charge_ratio` *(Engineered)* | Numerical | Relative current price burden: `monthly_charges / (total_charges + 1)`. |

---

## Performance & Evaluation Metrics
Evaluated on a 20% stratified held-out test split:

| Metric | Target Goal | Model Score | Business Justification |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | > 0.80 | **0.8350** | Strong discrimination across varied classification probability thresholds. |
| **Recall (Class 1)** | > 0.70 | **0.7420** | High priority: failing to detect a churner (False Negative) loses customer lifetime value. |
| **Precision (Class 1)** | > 0.65 | **0.6880** | Controls retention marketing budget waste on false alarms. |
| **F1-Score** | > 0.70 | **0.7140** | Harmonious balance between precision and recall. |
| **Inference Latency** | < 100ms | **< 3ms** | Meets strict real-time API SLA requirements on low-spec hardware. |

---

## Ethical Considerations & Bias Mitigation
1. **Proxy Discrimination Prevention**: Personal demographic indicators (e.g., zip codes, marital status, gender) were omitted to prevent disparate impact.
2. **Economic Fairness**: Customers with high monthly charges are flagged with higher churn probability due to billing sensitivity; retention teams must offer value-add discounts rather than service degradation.

---

## Caveats and Continuous Monitoring
- **Data Drift**: In real telecommunication markets, competitor pricing actions or regional fiber rollouts can cause concept drift. Continuous distribution monitoring via **Evidently AI** is integrated (`src/drift_detection.py`) to trigger automated alerts when feature drift share exceeds 30%.
- **Model Refresh Cadence**: Recommended retraining cycle is monthly or upon detection of statistically significant covariate shift.
