
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


## Taks-1 

## Credit Scoring Business Understanding

### 1. Basel II Accord and the Need for Interpretability

The Basel II Capital Accord requires financial institutions to measure and manage credit risk in a transparent, auditable, and data-driven way. This means that our credit scoring model must not only produce accurate risk predictions but also be understandable and explainable to internal stakeholders, regulators, and auditors. An interpretable model helps ensure that risk measurement aligns with regulatory capital requirements and enables Bati Bank to justify lending decisions and risk-weighted assets calculations under Basel II.

### 2. Why Use a Proxy Variable for Default?

Since our eCommerce dataset does not contain a direct default label, we must engineer a proxy variable — for example, using customer behavior patterns such as transaction chargebacks, fraud flags, or abnormal RFM (Recency, Frequency, Monetary) trends to approximate default risk. While necessary, this approach introduces business risk because the proxy may not perfectly represent true default behavior. If the proxy is poorly chosen, the model might misclassify good customers as risky (or vice versa), leading to lost revenue opportunities or higher loan losses.

### 3. Trade-offs: Simple vs. Complex Models

In regulated environments, there is always a balance between model performance and interpretability. A simple model like Logistic Regression with Weight of Evidence (WoE) encoding is easy to explain and audit, which aligns well with Basel II compliance. However, it may underperform on complex, non-linear relationships in the data. On the other hand, advanced models like Gradient Boosting can capture hidden patterns and deliver higher predictive power but can be harder to interpret. This trade-off requires careful documentation, robust model validation, and possibly using explainability tools (e.g., SHAP) if a complex model is chosen.
