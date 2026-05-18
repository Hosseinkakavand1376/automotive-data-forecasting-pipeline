"""
Tests for the data validation module.
"""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.validate import (
    check_positive_price,
    check_valid_year,
    check_non_negative_mileage,
    check_brand_not_empty,
    check_valid_transmission,
    check_valid_fuel_type,
    check_no_duplicate_ids,
    check_no_critical_nulls,
)


@pytest.fixture
def valid_df():
    """Create a DataFrame that passes all validation checks."""
    return pd.DataFrame({
        "listing_id": [1, 2, 3],
        "selling_price": [500000, 700000, 300000],
        "year": [2018, 2020, 2015],
        "km_driven": [30000, 15000, 80000],
        "brand": ["Maruti", "Hyundai", "Honda"],
        "fuel": ["Petrol", "Diesel", "Petrol"],
        "transmission": ["Manual", "Automatic", "Manual"],
    })


@pytest.fixture
def invalid_df():
    """Create a DataFrame that fails some validation checks."""
    return pd.DataFrame({
        "listing_id": [1, 1, 3],  # Duplicate ID
        "selling_price": [500000, -100, 300000],  # Negative price
        "year": [2018, 1950, 2015],  # Invalid year
        "km_driven": [30000, -500, 80000],  # Negative mileage
        "brand": ["Maruti", "", "Honda"],  # Empty brand
        "fuel": ["Petrol", "Nuclear", "Petrol"],  # Invalid fuel
        "transmission": ["Manual", "Teleport", "Manual"],  # Invalid transmission
    })


def test_positive_price_pass(valid_df):
    result = check_positive_price(valid_df)
    assert result.passed


def test_positive_price_fail(invalid_df):
    result = check_positive_price(invalid_df)
    assert not result.passed
    assert result.violations == 1


def test_valid_year_pass(valid_df):
    result = check_valid_year(valid_df)
    assert result.passed


def test_valid_year_fail(invalid_df):
    result = check_valid_year(invalid_df)
    assert not result.passed


def test_non_negative_mileage_pass(valid_df):
    result = check_non_negative_mileage(valid_df)
    assert result.passed


def test_non_negative_mileage_fail(invalid_df):
    result = check_non_negative_mileage(invalid_df)
    assert not result.passed


def test_brand_not_empty_pass(valid_df):
    result = check_brand_not_empty(valid_df)
    assert result.passed


def test_valid_fuel_type_pass(valid_df):
    result = check_valid_fuel_type(valid_df)
    assert result.passed


def test_valid_fuel_type_fail(invalid_df):
    result = check_valid_fuel_type(invalid_df)
    assert not result.passed


def test_valid_transmission_pass(valid_df):
    result = check_valid_transmission(valid_df)
    assert result.passed


def test_valid_transmission_fail(invalid_df):
    result = check_valid_transmission(invalid_df)
    assert not result.passed


def test_no_duplicate_ids_pass(valid_df):
    result = check_no_duplicate_ids(valid_df)
    assert result.passed


def test_no_duplicate_ids_fail(invalid_df):
    result = check_no_duplicate_ids(invalid_df)
    assert not result.passed


def test_no_critical_nulls_pass(valid_df):
    result = check_no_critical_nulls(valid_df)
    assert result.passed
