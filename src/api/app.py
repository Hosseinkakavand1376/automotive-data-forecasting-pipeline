"""
FastAPI Application
====================
RESTful API for the Automotive Data Pipeline.

Endpoints:
- GET  /             → Project name and status
- GET  /health       → API health check
- POST /predict-price → Predict car price from features
- GET  /data-summary  → Basic statistics from cleaned dataset
"""

import os
import logging
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

from src.config import (
    BEST_MODEL_PATH, CLEANED_CSV_PATH,
    VALID_FUEL_TYPES, VALID_TRANSMISSIONS, VALID_SELLER_TYPES, VALID_OWNER_TYPES,
    CURRENT_YEAR,
)

logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Automotive Data Pipeline API",
    description="API for predicting used-car prices and accessing pipeline data.",
    version="1.0.0",
)


# ============================================================================
# Request/Response Models
# ============================================================================

class CarFeatures(BaseModel):
    """Input features for price prediction."""
    year: int = Field(..., ge=1990, le=2030, description="Production year")
    km_driven: int = Field(..., ge=0, description="Total kilometers driven")
    fuel: str = Field(..., description="Fuel type (Petrol, Diesel, CNG, LPG, Electric)")
    transmission: str = Field(..., description="Transmission (Manual, Automatic)")
    seller_type: str = Field(..., description="Seller type (Individual, Dealer, Trustmark Dealer)")
    owner: str = Field(..., description="Owner type (First Owner, Second Owner, etc.)")
    brand: Optional[str] = Field(None, description="Car brand (optional)")
    model: Optional[str] = Field(None, description="Car model (optional)")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "year": 2018,
            "km_driven": 45000,
            "fuel": "Petrol",
            "transmission": "Manual",
            "seller_type": "Individual",
            "owner": "First Owner",
            "brand": "Maruti",
            "model": "Swift",
        }
    })


class PredictionResponse(BaseModel):
    """Response for price prediction."""
    predicted_price: float
    predicted_price_formatted: str
    model_used: str
    input_features: dict


# ============================================================================
# Helper Functions
# ============================================================================

def _load_model():
    """Load the trained model from disk."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run the pipeline first to train the model."
        )
    return joblib.load(BEST_MODEL_PATH)


def _encode_categorical(value: str, valid_values: list) -> int:
    """Simple label encoding for a categorical value."""
    sorted_values = sorted(valid_values)
    if value in sorted_values:
        return sorted_values.index(value)
    return 0


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
def root():
    """Project name and status."""
    return {
        "project": "Automotive Data Preparation & Forecasting Pipeline",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/", "/health", "/predict-price", "/data-summary"],
    }


@app.get("/health")
def health_check():
    """API health status."""
    model_available = os.path.exists(BEST_MODEL_PATH)
    data_available = os.path.exists(CLEANED_CSV_PATH)

    return {
        "status": "healthy",
        "model_loaded": model_available,
        "data_available": data_available,
    }


@app.post("/predict-price", response_model=PredictionResponse)
def predict_price(car: CarFeatures):
    """
    Predict the selling price of a used car based on its features.
    
    Accepts car details as JSON and returns the predicted price.
    """
    # Load model
    model_data = _load_model()
    model = model_data["model"]
    feature_names = model_data["feature_names"]

    # Build feature vector
    car_age = CURRENT_YEAR - car.year
    mileage_per_year = car.km_driven / max(car_age, 1)

    feature_dict = {
        "year": car.year,
        "km_driven": car.km_driven,
        "car_age": car_age,
        "mileage_per_year": mileage_per_year,
        "log_mileage": float(__import__("numpy").log1p(car.km_driven)),
        "brand_average_price": 5000,  # Default estimate in EUR
        "model_average_price": 5000,  # Default estimate in EUR
        "fuel_type_encoded": _encode_categorical(car.fuel, VALID_FUEL_TYPES),
        "transmission_encoded": _encode_categorical(car.transmission, VALID_TRANSMISSIONS),
        "seller_type_encoded": _encode_categorical(car.seller_type, VALID_SELLER_TYPES),
        "owner_type_encoded": _encode_categorical(car.owner, VALID_OWNER_TYPES),
    }

    # Build feature array in the correct order
    features = []
    for fname in feature_names:
        if fname in feature_dict:
            features.append(feature_dict[fname])
        else:
            features.append(0)

    # Make prediction
    import numpy as np
    prediction = model.predict(np.array([features]))[0]
    prediction = max(0, prediction)  # Price can't be negative

    return PredictionResponse(
        predicted_price=round(float(prediction), 2),
        predicted_price_formatted=f"EUR {prediction:,.0f}",
        model_used=model_data["model_name"],
        input_features=car.model_dump(),
    )


@app.get("/data-summary")
def data_summary():
    """Return basic statistics from the cleaned dataset."""
    if not os.path.exists(CLEANED_CSV_PATH):
        raise HTTPException(
            status_code=503,
            detail="Cleaned data not found. Run the pipeline first."
        )

    df = pd.read_csv(CLEANED_CSV_PATH)

    summary = {
        "total_records": len(df),
        "columns": list(df.columns),
        "numeric_stats": {},
        "categorical_stats": {},
    }

    # Numeric statistics
    numeric_cols = ["selling_price", "km_driven", "year"]
    for col in numeric_cols:
        if col in df.columns:
            summary["numeric_stats"][col] = {
                "mean": round(float(df[col].mean()), 2),
                "median": round(float(df[col].median()), 2),
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
                "std": round(float(df[col].std()), 2),
            }

    # Categorical statistics
    cat_cols = ["brand", "fuel", "transmission", "seller_type", "owner"]
    for col in cat_cols:
        if col in df.columns:
            summary["categorical_stats"][col] = {
                "unique_values": int(df[col].nunique()),
                "top_values": df[col].value_counts().head(5).to_dict(),
            }

    return summary
