# rfm_proxy_target.py

from ast import main
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def compute_rfm(df, snapshot_date):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime']).dt.tz_localize(None)
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm

def cluster_rfm(rfm, n_clusters=3, random_state=42):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    return rfm

def assign_high_risk(rfm):
    # Assumption: the cluster with **highest Recency, lowest Frequency & Monetary** is high risk
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    high_risk_cluster = cluster_summary.sort_values(
        by=['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]
    ).iloc[0]['Cluster']
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    return rfm[['CustomerId', 'is_high_risk']]

def merge_target(df, rfm_labels):
    return df.merge(rfm_labels, on='CustomerId', how='left')

def main(input_path, output_path):
    snapshot_date = pd.Timestamp("2023-12-31")  # You can adjust this date
    df = pd.read_csv(input_path)
    rfm = compute_rfm(df, snapshot_date)
    clustered_rfm = cluster_rfm(rfm)
    labels = assign_high_risk(clustered_rfm)
    df_with_target = merge_target(df, labels)
    df_with_target.to_csv(output_path, index=False)
    print(f"✅ Saved processed data with target to: {output_path}")

if __name__ == "__main__":
    input_path = "./data/raw/data.csv"
    output_path = "./data/processed/processed_data_with_target.csv"
    main(input_path, output_path)