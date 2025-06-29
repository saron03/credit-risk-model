
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

## Task 2 — Exploratory Data Analysis (EDA)

The goal of this task is to understand the dataset, find patterns, check data quality, and develop early ideas that will guide feature engineering for the credit risk model.

### Steps Completed

- Loaded the dataset with 95,662 rows and 16 columns.

- Checked data types: Most columns are categorical (object), and a few are numeric (Amount, Value, CountryCode, PricingStrategy, FraudResult).

- Used describe() to check min, max, mean, and standard deviation for numeric features.

- Found Amount and Value have wide ranges, from large negatives to large positives — indicating refunds/credits and possible outliers.

- Checked for missing values — none found. The dataset is complete.

- Noted extreme values in Amount and Value from the summary stats. Will confirm these using box plots in the next EDA step.

### Key Insights So Far

- Complete Data: No missing values.

- Single Country: All transactions use CountryCode = 256.

- Large Value Spread: Transaction amounts show a big range with possible outliers.

- Fraud Data: FraudResult is very rare (only ~0.2%) but might help define risky users.

- Well-Structured Data: All columns are clearly named and understandable.

