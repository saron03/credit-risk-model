# debug_data.py
import pandas as pd

data = pd.read_csv("../data/processed/processed_data_with_target.csv")
print("Columns in dataset:", data.columns.tolist())
print("\nData types:\n", data.dtypes)
print("\nSample data:\n", data.head())