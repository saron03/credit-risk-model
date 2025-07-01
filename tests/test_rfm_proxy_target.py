import pandas as pd
from src.rfm_proxy_target import compute_rfm, cluster_rfm, assign_high_risk

def test_rfm_pipeline():
    data = {
        'CustomerId': [1, 1, 2, 2, 3],
        'TransactionStartTime': [
            '2025-06-01', '2025-06-15',
            '2025-05-01', '2025-05-20',
            '2025-01-01'
        ],
        'TransactionId': [101, 102, 103, 104, 105],
        'Amount': [100, 200, 50, 75, 20]
    }
    df = pd.DataFrame(data)
    snapshot_date = pd.Timestamp('2025-07-01')
    rfm = compute_rfm(df, snapshot_date)
    assert 'Recency' in rfm.columns
    clustered = cluster_rfm(rfm)
    labeled = assign_high_risk(clustered)
    assert 'is_high_risk' in labeled.columns
    assert labeled['is_high_risk'].isin([0, 1]).all()

def test_sample():
    assert 1 + 1 == 2
    print("Test ran successfully")

if __name__ == "__main__":
    test_sample()
    test_rfm_pipeline()
    # call other test functions here
