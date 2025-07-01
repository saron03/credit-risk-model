# tests/test_data_processing.py

import pandas as pd
from src.data_processing import build_pipeline
import pytest

def test_pipeline_output():
    data = pd.DataFrame({
        'TransactionId': [1, 2, 3],
        'CustomerId': [100, 100, 200],
        'Amount': [50, 60, 70],
        'Value': [50, 60, 70],
        'CurrencyCode': ['USD', 'USD', 'EUR'],
        'CountryCode': [1, 1, 2],
        'ProviderId': ['P1', 'P1', 'P2'],
        'ProductCategory': ['A', 'A', 'B'],
        'ChannelId': ['web', 'ios', 'android'],
        'PricingStrategy': ['standard', 'standard', 'discount'],
        'TransactionStartTime': ['2025-06-01 12:00:00',
                                '2025-06-01 13:00:00',
                                '2025-06-02 14:00:00']
    })

    pipeline = build_pipeline()
    output = pipeline.fit_transform(data)
    
    # Verify output
    assert isinstance(output, pd.DataFrame), "Output should be a pandas DataFrame"
    assert 'CustomerId' in output.columns, "CustomerId should be in output columns"
    assert 'TotalTransactionAmount' in output.columns, "TotalTransactionAmount should be in output columns"
    assert len(output) == 2, "Output should have 2 rows (one for each CustomerId)"
    assert output.loc[output['CustomerId'] == 100, 'TransactionCount'].iloc[0] == 2, "Customer 100 should have 2 transactions"
    assert output.loc[output['CustomerId'] == 200, 'TransactionCount'].iloc[0] == 1, "Customer 200 should have 1 transaction"
    
    # Note: Amounts are scaled, so we check the structure rather than exact values
    assert output['TotalTransactionAmount'].notnull().all(), "TotalTransactionAmount should not contain nulls"
    assert output['AverageTransactionAmount'].notnull().all(), "AverageTransactionAmount should not contain nulls"

if __name__ == "__main__":
    data = pd.DataFrame({
        'TransactionId': [1, 2, 3],
        'CustomerId': [100, 100, 200],
        'Amount': [50, 60, 70],
        'Value': [50, 60, 70],
        'CurrencyCode': ['USD', 'USD', 'EUR'],
        'CountryCode': [1, 1, 2],
        'ProviderId': ['P1', 'P1', 'P2'],
        'ProductCategory': ['A', 'A', 'B'],
        'ChannelId': ['web', 'ios', 'android'],
        'PricingStrategy': ['standard', 'standard', 'discount'],
        'TransactionStartTime': ['2025-06-01 12:00:00',
                                '2025-06-01 13:00:00',
                                '2025-06-02 14:00:00']
    })
    pipeline = build_pipeline()
    output = pipeline.fit_transform(data)
    print(output)