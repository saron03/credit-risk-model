
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
# Task 6 - Model Deployment and Continuous Integration

This project packages a trained credit risk model into a containerized FastAPI service and sets up a CI/CD pipeline to automate testing and ensure code quality.

## Features

- **FastAPI REST API** that loads the best model from the MLflow registry.
- `/predict` endpoint accepts customer data and returns risk probabilities.
- Request and response data validated using Pydantic models.
- **Dockerized** service with a `Dockerfile` and `docker-compose.yml` for easy build and deployment.
- **CI/CD pipeline** configured via GitHub Actions to:
  - Run code linting with `flake8`.
  - Execute unit tests using `pytest`.
- Build fails if linting or tests do not pass, ensuring code quality.