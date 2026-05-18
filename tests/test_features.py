"""
Tests for the feature engineering module.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.engineer_features import (
    create_car_age,
    create_mileage_per_year,
    create_log_features,
    create_price_segment,
    create_mileage_bucket,
    create_brand_model_stats,
    encode_categorical_features,
    create_time_features,
)
from src.config import CURRENT_YEAR


@pytest.fixture
def cleaned_df():
    """Create a cleaned DataFrame for feature engineering tests."""
    return pd.DataFrame({
        "year": [2018, 2020, 2015, 2022, 2010],
        "selling_price": [500000, 700000, 300000, 1500000, 200000],
        "km_driven": [30000, 15000, 80000, 5000, 120000],
        "brand": ["Maruti", "Hyundai", "Honda", "Maruti", "Honda"],
        "model": ["Swift", "i20", "City", "Baleno", "City"],
        "fuel": ["Petrol", "Diesel", "Petrol", "Petrol", "Diesel"],
        "transmission": ["Manual", "Automatic", "Manual", "Automatic", "Manual"],
        "seller_type": ["Individual", "Dealer", "Individual", "Dealer", "Individual"],
        "owner": ["First Owner", "First Owner", "Second Owner", "First Owner", "Third Owner"],
        "listing_date": pd.to_datetime(["2023-01-15", "2023-06-20", "2022-03-10", "2024-01-05", "2021-11-30"]),
    })


def test_create_car_age(cleaned_df):
    result = create_car_age(cleaned_df)
    assert "car_age" in result.columns
    assert result["car_age"].iloc[0] == CURRENT_YEAR - 2018
    assert (result["car_age"] >= 0).all()


def test_create_mileage_per_year(cleaned_df):
    cleaned_df = create_car_age(cleaned_df)
    result = create_mileage_per_year(cleaned_df)
    assert "mileage_per_year" in result.columns
    assert (result["mileage_per_year"] > 0).all()


def test_create_log_features(cleaned_df):
    result = create_log_features(cleaned_df)
    assert "log_price" in result.columns
    assert "log_mileage" in result.columns
    assert np.isfinite(result["log_price"]).all()
    assert np.isfinite(result["log_mileage"]).all()


def test_create_price_segment(cleaned_df):
    result = create_price_segment(cleaned_df)
    assert "price_segment" in result.columns
    assert result["price_segment"].nunique() <= 4


def test_create_mileage_bucket(cleaned_df):
    result = create_mileage_bucket(cleaned_df)
    assert "mileage_bucket" in result.columns


def test_create_brand_model_stats(cleaned_df):
    result = create_brand_model_stats(cleaned_df)
    assert "brand_average_price" in result.columns
    assert "model_average_price" in result.columns


def test_encode_categorical_features(cleaned_df):
    result = encode_categorical_features(cleaned_df)
    assert "fuel_type_encoded" in result.columns
    assert "transmission_encoded" in result.columns
    assert "seller_type_encoded" in result.columns
    assert "owner_type_encoded" in result.columns


def test_create_time_features(cleaned_df):
    result = create_time_features(cleaned_df)
    assert "listing_month" in result.columns
    assert "listing_year" in result.columns
    assert "listing_season" in result.columns
