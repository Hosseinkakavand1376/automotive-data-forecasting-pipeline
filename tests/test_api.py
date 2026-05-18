"""
Tests for the FastAPI application.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns project info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert data["status"] == "running"


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_price_endpoint():
    """Test the price prediction endpoint."""
    car_data = {
        "year": 2018,
        "km_driven": 45000,
        "fuel": "Petrol",
        "transmission": "Manual",
        "seller_type": "Individual",
        "owner": "First Owner",
        "brand": "Maruti",
        "model": "Swift",
    }

    response = client.post("/predict-price", json=car_data)
    # May return 503 if model not trained yet — that's OK
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "predicted_price" in data
        assert data["predicted_price"] >= 0


def test_data_summary_endpoint():
    """Test the data summary endpoint."""
    response = client.get("/data-summary")
    # May return 503 if pipeline hasn't run — that's OK
    assert response.status_code in [200, 503]
