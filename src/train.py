# train.py

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

import os
import sys
print(sys.path)
from src.data_processing import build_pipeline


def load_data(path="data/processed/processed_data_with_target.csv"):
    data = pd.read_csv(path)
    print("Columns in dataset:", data.columns.tolist())
    drop_columns = ["TransactionId", "BatchId", "AccountId", "SubscriptionId", "ProductId", "FraudResult"]
    drop_columns = [col for col in drop_columns if col in data.columns]
    data = data.drop(columns=drop_columns, errors='ignore')
    return data

def evaluate_model(y_true, y_pred, y_proba):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }
    return metrics

def train_and_evaluate():
    data = load_data()
    target_column = "is_high_risk"
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {data.columns.tolist()}")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    customer_id_train = X_train["CustomerId"]
    customer_id_test = X_test["CustomerId"]
    
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    y_train_agg = pd.DataFrame({"CustomerId": customer_id_train, target_column: y_train})
    y_train_agg = y_train_agg.groupby("CustomerId")[target_column].max().reset_index()
    y_test_agg = pd.DataFrame({"CustomerId": customer_id_test, target_column: y_test})
    y_test_agg = y_test_agg.groupby("CustomerId")[target_column].max().reset_index()
    
    X_train_transformed = X_train_transformed.merge(y_train_agg, on="CustomerId", how="inner")
    y_train_transformed = X_train_transformed[target_column]
    X_train_transformed = X_train_transformed.drop(columns=[target_column, "CustomerId"])
    
    X_test_transformed = X_test_transformed.merge(y_test_agg, on="CustomerId", how="inner")
    y_test_transformed = X_test_transformed[target_column]
    X_test_transformed = X_test_transformed.drop(columns=[target_column, "CustomerId"])

    # Debug: Check transformed data
    print("X_train_transformed columns:", X_train_transformed.columns.tolist())
    print("X_train_transformed NaN counts:\n", X_train_transformed.isna().sum())
    print("X_train_transformed head:\n", X_train_transformed.head())

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear'),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    param_grids = {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"]
        },
        "RandomForest": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        }
    }

    best_model_name = None
    best_model = None
    best_score = 0
    best_params = None

    mlflow.set_experiment("Credit Risk Model Training")

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        grid = GridSearchCV(model, param_grids[model_name], cv=3, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train_transformed, y_train_transformed)

        y_pred = grid.predict(X_test_transformed)
        y_proba = grid.predict_proba(X_test_transformed)[:, 1]

        metrics = evaluate_model(y_test_transformed, y_pred, y_proba)
        print(f"{model_name} metrics:", metrics)

        # Log model with input example
        input_example = X_train_transformed.iloc[:5]
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(grid.best_estimator_, name=model_name, input_example=input_example)

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = grid.best_estimator_
            best_model_name = model_name
            best_params = grid.best_params_

    print(f"Best model: {best_model_name} with ROC-AUC: {best_score}")
    mlflow.sklearn.log_model(best_model, name="best_model", input_example=input_example, registered_model_name="CreditRiskModel")
    joblib.dump(best_model, "models/best_model.joblib")

if __name__ == "__main__":
    train_and_evaluate()