from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn # type: ignore
import os
import logging
from src.data_processing import build_pipeline
from src.api.pydantic_models import Transaction, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk Prediction API")

# Load the model from MLflow
MODEL_NAME = "CreditRiskModel"
MODEL_VERSION = "3"
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        logger.info(f"Connecting to MLflow at {mlflow_tracking_uri}")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Successfully loaded model {MODEL_NAME} version {MODEL_VERSION}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model from MLflow: {str(e)}")

# Build the preprocessing pipeline
pipeline = build_pipeline()

@app.post("/predict", response_model=PredictionResponse)
async def predict(transactions: list[Transaction]):
    try:
        # Convert input transactions to DataFrame
        data = pd.DataFrame([t.dict() for t in transactions])
        logger.info(f"Received {len(data)} transactions for prediction")
        
        # Validate required columns
        required_columns = ['CustomerId', 'CurrencyCode', 'CountryCode', 'ProviderId',
                           'ProductCategory', 'ChannelId', 'Amount', 'Value',
                           'TransactionStartTime', 'PricingStrategy']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")
        
        # Process data through the pipeline
        customer_ids = data['CustomerId']
        transformed_data = pipeline.transform(data)
        
        # Merge with customer_ids to ensure alignment
        transformed_data = transformed_data.drop(columns=['CustomerId'], errors='ignore')
        transformed_data = transformed_data.astype('float64')  # Ensure float64 for model input
        logger.info(f"Transformed data shape: {transformed_data.shape}")
        
        # Predict probabilities
        probs = model.predict_proba(transformed_data)[:, 1]
        
        # Create response
        predictions = [
            {"CustomerId": cid, "RiskProbability": float(prob)}
            for cid, prob in zip(customer_ids.unique(), probs)
        ]
        
        logger.info(f"Generated {len(predictions)} predictions")
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}