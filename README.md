
# B5W5: Credit Risk Probability Model for Alternative Data

## Overview

This project implements an end-to-end Credit Scoring and Risk Probability Model for Bati Bank’s new Buy-Now-Pay-Later service. The goal is to predict the probability of default using alternative eCommerce behavioral data (RFM patterns), assign a risk-based credit score, and recommend optimal loan amounts and durations.

---

## Business Need

Bati Bank is partnering with a fast-growing eCommerce platform to provide flexible credit for online shoppers. To enable responsible lending, we must estimate each customer’s credit risk accurately and automatically, using behavioral transaction data when traditional credit histories are unavailable.

---

## Project Objectives

This solution will:
- ✅ Define a **proxy variable** for default using available transaction and fraud data.
- ✅ Select meaningful predictive features.
- ✅ Train a model to output a **risk probability score**.
- ✅ Transform probability estimates into **credit scores**.
- ✅ Predict the **optimal amount and duration** for the loan.
- ✅ Deploy the entire pipeline as an automated service.

---
## 📂 Project Structure

```plaintext
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile 
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```
## Task 3 - Feature Engineering

In this task, we developed an automated, reproducible feature engineering pipeline to transform raw eCommerce transaction data into model-ready inputs for credit risk modeling.

### Objectives
- Aggregate customer transaction data to create summary features such as total transaction amount, average transaction amount, transaction count, and transaction amount variability.
- Extract temporal features (hour, day, month, year) from transaction timestamps.
- Encode categorical variables using One-Hot Encoding with proper handling of missing values.
- Impute missing values in numerical and categorical features to ensure data quality.
- Normalize and standardize numerical features to bring them onto a comparable scale.
- Implement the entire transformation as a reusable `sklearn.pipeline.Pipeline` to guarantee consistent preprocessing during training and inference.

### Implementation Details
- The feature engineering logic is encapsulated in `src/data_processing.py`.
- Custom transformers for aggregate and time-based features extend `sklearn.base.BaseEstimator` and `TransformerMixin`.
- Categorical and numerical features are processed using `ColumnTransformer` with `SimpleImputer`, `OneHotEncoder`, and `StandardScaler`.
- Unit tests in `tests/test_data_processing.py` ensure pipeline correctness and robustness.

This pipeline forms a critical step towards building reliable credit risk models by ensuring consistent and meaningful feature extraction from raw transaction data.


