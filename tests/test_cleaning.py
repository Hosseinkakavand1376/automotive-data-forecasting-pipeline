"""
Tests for the data cleaning module.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cleaning.clean import (
    remove_duplicates,
    standardize_column_names,
    split_car_name,
    normalize_text_columns,
    handle_missing_values,
    remove_invalid_years,
    remove_invalid_mileage,
    handle_price_outliers,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "name": ["Maruti Swift", "Hyundai i20", "Honda City", "Maruti Swift", "Toyota Innova"],
        "year": [2018, 2020, 2015, 2018, 2022],
        "selling_price": [500000, 700000, 450000, 500000, 1500000],
        "km_driven": [30000, 15000, 80000, 30000, 5000],
        "fuel": ["Petrol", "Diesel", "Petrol", "Petrol", "Diesel"],
        "seller_type": ["Individual", "Dealer", "Individual", "Individual", "Dealer"],
        "transmission": ["Manual", "Automatic", "Manual", "Manual", "Automatic"],
        "owner": ["First Owner", "First Owner", "Second Owner", "First Owner", "First Owner"],
    })


def test_remove_duplicates(sample_df):
    """Test that exact duplicates are removed."""
    result = remove_duplicates(sample_df)
    assert len(result) == 4  # One duplicate pair removed
    assert not result.duplicated().any()


def test_standardize_column_names():
    """Test column name standardization."""
    df = pd.DataFrame({"Car Name": [1], "Selling Price": [2], "Km Driven": [3]})
    result = standardize_column_names(df)
    assert list(result.columns) == ["car_name", "selling_price", "km_driven"]


def test_split_car_name(sample_df):
    """Test splitting car name into brand and model."""
    result = split_car_name(sample_df)
    assert "brand" in result.columns
    assert "model" in result.columns
    assert result["brand"].iloc[0] == "Maruti"
    assert result["model"].iloc[0] == "Swift"


def test_handle_missing_values():
    """Test missing value handling."""
    df = pd.DataFrame({
        "selling_price": [100, np.nan, 300],
        "fuel": ["Petrol", None, "Diesel"],
        "km_driven": [1000, 2000, np.nan],
    })
    result = handle_missing_values(df)
    assert result.isnull().sum().sum() == 0


def test_remove_invalid_years():
    """Test removal of invalid production years."""
    df = pd.DataFrame({"year": [2018, 1950, 2020, 2035, 2010]})
    result = remove_invalid_years(df)
    assert 1950 not in result["year"].values
    assert 2035 not in result["year"].values
    assert len(result) == 3


def test_remove_invalid_mileage():
    """Test removal of invalid mileage values."""
    df = pd.DataFrame({"km_driven": [10000, -500, 50000, 2000000, 30000]})
    result = remove_invalid_mileage(df)
    assert (result["km_driven"] >= 0).all()
    assert (result["km_driven"] <= 1000000).all()
    assert len(result) == 3


def test_handle_price_outliers():
    """Test price outlier handling."""
    prices = [100000] * 98 + [50000000, 60000000]  # Two extreme outliers
    df = pd.DataFrame({"selling_price": prices})
    result = handle_price_outliers(df)
    assert result["selling_price"].max() < 60000000  # Outliers should be capped
