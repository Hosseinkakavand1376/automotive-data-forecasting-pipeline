"""
Forecasting Dataset Preparation
=================================
Prepares time-series datasets for demand and price forecasting
by aggregating car listings at monthly and brand-monthly levels.
"""

import os
import logging
import sqlite3
import pandas as pd

from src.config import FORECASTING_CSV_PATH, DATABASE_PATH, FINAL_DATA_DIR

logger = logging.getLogger(__name__)


def prepare_monthly_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate car listings by month for time-series forecasting.

    Creates:
    - Average selling price per month
    - Number of listings per month
    - Average mileage per month
    - Average car age per month

    Args:
        df: DataFrame with engineered features (must have listing_date).

    Returns:
        Monthly aggregated DataFrame.
    """
    if "listing_date" not in df.columns:
        logger.error("listing_date column required for forecasting preparation")
        raise ValueError("listing_date column is required")

    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")

    # Create year-month period
    df["year_month"] = df["listing_date"].dt.to_period("M")

    # Overall monthly aggregation
    monthly = df.groupby("year_month").agg(
        avg_price=("selling_price", "mean"),
        listing_count=("selling_price", "count"),
        avg_mileage=("km_driven", "mean"),
        avg_car_age=("car_age", "mean") if "car_age" in df.columns else ("year", "count"),
        median_price=("selling_price", "median"),
        min_price=("selling_price", "min"),
        max_price=("selling_price", "max"),
        std_price=("selling_price", "std"),
    ).round(2).reset_index()

    monthly["year_month"] = monthly["year_month"].astype(str)

    logger.info(f"Monthly aggregation: {len(monthly)} months of data")
    return monthly


def prepare_brand_monthly_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate car listings by brand and month for brand-level forecasting.

    Args:
        df: DataFrame with engineered features.

    Returns:
        Brand-monthly aggregated DataFrame.
    """
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    df["year_month"] = df["listing_date"].dt.to_period("M")

    # Brand-monthly aggregation
    brand_monthly = df.groupby(["brand", "year_month"]).agg(
        avg_price=("selling_price", "mean"),
        listing_count=("selling_price", "count"),
        avg_mileage=("km_driven", "mean"),
        median_price=("selling_price", "median"),
    ).round(2).reset_index()

    brand_monthly["year_month"] = brand_monthly["year_month"].astype(str)

    logger.info(f"Brand-monthly aggregation: {len(brand_monthly)} brand-month combinations")
    return brand_monthly


def run_forecasting_preparation(df: pd.DataFrame) -> dict:
    """
    Execute forecasting dataset preparation and save outputs.

    Args:
        df: DataFrame with engineered features.

    Returns:
        Dictionary with monthly and brand_monthly DataFrames.
    """
    logger.info("=" * 60)
    logger.info("STARTING FORECASTING PREPARATION")
    logger.info("=" * 60)

    # Create monthly aggregations
    monthly_df = prepare_monthly_aggregations(df)

    # Create brand-monthly aggregations
    brand_monthly_df = prepare_brand_monthly_aggregations(df)

    # Combine into a single forecasting dataset
    # The monthly dataset is the primary one; brand-monthly is supplementary
    os.makedirs(FINAL_DATA_DIR, exist_ok=True)

    # Save overall monthly data
    monthly_df.to_csv(FORECASTING_CSV_PATH, index=False)
    logger.info(f"Saved monthly forecasting data to: {FORECASTING_CSV_PATH}")

    # Save brand-monthly data separately
    brand_monthly_path = os.path.join(FINAL_DATA_DIR, "brand_monthly_forecasting_dataset.csv")
    brand_monthly_df.to_csv(brand_monthly_path, index=False)
    logger.info(f"Saved brand-monthly forecasting data to: {brand_monthly_path}")

    # Save to database
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        monthly_df.to_sql("monthly_forecasting_data", conn, if_exists="replace", index=False)
        brand_monthly_df.to_sql("brand_monthly_forecasting_data", conn, if_exists="replace", index=False)
        logger.info("Saved forecasting data to database")
    finally:
        conn.close()

    logger.info("=" * 60)
    logger.info("FORECASTING PREPARATION COMPLETE")
    logger.info("=" * 60)

    return {
        "monthly": monthly_df,
        "brand_monthly": brand_monthly_df,
    }
