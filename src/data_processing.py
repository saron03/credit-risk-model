# data_processing.py

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

class ExtractTimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        return X

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.feature_names_ is None:
            raise ValueError("Feature names not set. Ensure the transformer is used within a fitted pipeline.")
        
        # Convert NumPy array to DataFrame
        X_df = pd.DataFrame(X, columns=self.feature_names_)
        
        # Identify the correct column names for CustomerId and Amount
        customer_id_col = [col for col in X_df.columns if col.endswith('CustomerId')][0]
        amount_col = [col for col in X_df.columns if col.endswith('Amount')][0]
        
        # Check if required columns are present
        if customer_id_col not in X_df.columns or amount_col not in X_df.columns:
            raise ValueError("Required columns for CustomerId and Amount are missing from the input data")
        
        # Perform aggregation
        agg_df = X_df.groupby(customer_id_col).agg(
            TotalTransactionAmount=(amount_col, 'sum'),
            AverageTransactionAmount=(amount_col, 'mean'),
            TransactionCount=(customer_id_col, 'count'),
            StdTransactionAmount=(amount_col, 'std')
        ).reset_index()
        
        # Rename the grouped column back to CustomerId for clarity
        agg_df.columns = ['CustomerId'] + [col for col in agg_df.columns if col != customer_id_col]
        return agg_df

def build_pipeline():
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId',
                           'ProductCategory', 'ChannelId', 'PricingStrategy']
    numerical_features = ['Amount', 'Value']
    passthrough_features = ['CustomerId']

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('passthrough', 'passthrough', passthrough_features)
    ])

    pipeline = Pipeline([
        ('time_features', ExtractTimeFeatures()),
        ('preprocessor', preprocessor),
        ('aggregate', AggregateFeatures())
    ])

    # Override fit_transform to set feature names for AggregateFeatures
    original_fit_transform = pipeline.fit_transform
    def fit_transform_with_feature_names(X, y=None):
        # Fit the pipeline up to the preprocessor
        X_transformed = pipeline.named_steps['time_features'].fit_transform(X, y)
        X_transformed = pipeline.named_steps['preprocessor'].fit_transform(X_transformed, y)
        
        # Get feature names from preprocessor
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        # Set feature names in AggregateFeatures
        pipeline.named_steps['aggregate'].feature_names_ = feature_names
        
        # Complete the transformation
        return pipeline.named_steps['aggregate'].fit_transform(X_transformed, y)
    
    pipeline.fit_transform = fit_transform_with_feature_names
    return pipeline