from pydantic import BaseModel
from typing import List
from datetime import datetime

class Transaction(BaseModel):
    CustomerId: str
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str  # ISO format, e.g., "2023-01-01T12:00:00"
    PricingStrategy: str

class Prediction(BaseModel):
    CustomerId: str
    RiskProbability: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]