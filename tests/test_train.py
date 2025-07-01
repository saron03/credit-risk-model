import pytest
import pandas as pd
from src import train

def test_load_data():
    data = train.load_data()
    assert isinstance(data, pd.DataFrame)
    assert "is_high_risk" in data.columns

def test_evaluate_model():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    y_proba = [0.2, 0.8, 0.6, 0.9]
    metrics = train.evaluate_model(y_true, y_pred, y_proba)
    assert all(k in metrics for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"])
    assert 0 <= metrics["accuracy"] <= 1

def test_train_and_evaluate_runs():
    # Just check that train_and_evaluate runs without exceptions (slow tests can be excluded)
    train.train_and_evaluate()
