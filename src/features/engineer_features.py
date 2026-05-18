"""
Feature Engineering Module
============================
Creates ML-ready features from cleaned car listing data.
Includes numeric transformations, categorical encoding, brand/model
statistics, and time-based aggregated features.
"""

import os
import logging
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    ML_FEATURES_CSV_PATH, DATABASE_PATH, PROCESSED_DATA_DIR,
    CURRENT_YEAR, PRICE_SEGMENTS, MILEAGE_BINS, MILEAGE_LABELS, SEASON_MAP,
)

logger = logging.getLogger(__name__)


def create_car_age(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate car age from production year."""
    df["car_age"] = CURRENT_YEAR - df["year"]
    df["car_age"] = df["car_age"].clip(lower=0)
    logger.info(f"Created 'car_age' (range: {df['car_age'].min()}-{df['car_age'].max()})")
    return df


def create_mileage_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average mileage per year of ownership."""
    df["mileage_per_year"] = df["km_driven"] / df["car_age"].replace(0, 1)
    df["mileage_per_year"] = df["mileage_per_year"].round(0).astype(int)
    logger.info(f"Created 'mileage_per_year' (mean: {df['mileage_per_year'].mean():,.0f})")
    return df


def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create log-transformed features for price and mileage."""
    df["log_price"] = np.log1p(df["selling_price"])
    df["log_mileage"] = np.log1p(df["km_driven"])
    logger.info("Created 'log_price' and 'log_mileage'")
    return df


def create_price_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Segment cars into price quartiles."""
    df["price_segment"] = pd.qcut(
        df["selling_price"], q=4, labels=PRICE_SEGMENTS, duplicates="drop"
    )
    logger.info(f"Created 'price_segment': {df['price_segment'].value_counts().to_dict()}")
    return df


def create_mileage_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket mileage into categories."""
    df["mileage_bucket"] = pd.cut(
        df["km_driven"], bins=MILEAGE_BINS, labels=MILEAGE_LABELS, right=False
    )
    logger.info(f"Created 'mileage_bucket': {df['mileage_bucket'].value_counts().to_dict()}")
    return df


def create_brand_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average price statistics at brand and model level."""
    # Brand average price
    brand_avg = df.groupby("brand")["selling_price"].transform("mean").round(0)
    df["brand_average_price"] = brand_avg

    # Model average price
    model_avg = df.groupby("model")["selling_price"].transform("mean").round(0)
    df["model_average_price"] = model_avg

    logger.info(f"Created 'brand_average_price' (unique: {df['brand_average_price'].nunique()})")
    logger.info(f"Created 'model_average_price' (unique: {df['model_average_price'].nunique()})")
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features using LabelEncoder."""
    encodings = {
        "fuel": "fuel_type_encoded",
        "transmission": "transmission_encoded",
        "seller_type": "seller_type_encoded",
        "owner": "owner_type_encoded",
    }

    for source_col, target_col in encodings.items():
        if source_col in df.columns:
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[source_col].astype(str))
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            logger.info(f"Encoded '{source_col}' -> '{target_col}': {mapping}")

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from listing_date."""
    if "listing_date" not in df.columns:
        logger.warning("No listing_date column — skipping time features")
        return df

    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")

    # Basic time features
    df["listing_month"] = df["listing_date"].dt.month
    df["listing_year"] = df["listing_date"].dt.year
    df["listing_season"] = df["listing_month"].map(SEASON_MAP)

    logger.info("Created 'listing_month', 'listing_year', 'listing_season'")
    return df


def create_time_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based aggregated features at monthly and brand-monthly levels."""
    if "listing_date" not in df.columns:
        logger.warning("No listing_date — skipping time aggregations")
        return df

    # Create year-month key for grouping
    df["year_month"] = df["listing_date"].dt.to_period("M").astype(str)

    # Monthly aggregations
    monthly_avg_price = df.groupby("year_month")["selling_price"].transform("mean").round(0)
    df["monthly_average_price"] = monthly_avg_price

    monthly_count = df.groupby("year_month")["selling_price"].transform("count")
    df["monthly_listing_count"] = monthly_count

    # Brand-monthly aggregations
    brand_monthly_avg = df.groupby(["brand", "year_month"])["selling_price"].transform("mean").round(0)
    df["brand_monthly_average_price"] = brand_monthly_avg

    brand_monthly_count = df.groupby(["brand", "year_month"])["selling_price"].transform("count")
    df["brand_monthly_listing_count"] = brand_monthly_count

    # Drop the helper column
    df = df.drop(columns=["year_month"])

    logger.info("Created time-aggregated features: monthly_average_price, monthly_listing_count, "
                "brand_monthly_average_price, brand_monthly_listing_count")
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the full feature engineering pipeline.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with all engineered features.
    """
    logger.info("=" * 60)
    logger.info("STARTING FEATURE ENGINEERING")
    logger.info("=" * 60)
    logger.info(f"Input: {len(df)} rows, {len(df.columns)} columns")

    # Numeric features
    df = create_car_age(df)
    df = create_mileage_per_year(df)
    df = create_log_features(df)

    # Segmentation features
    df = create_price_segment(df)
    df = create_mileage_bucket(df)

    # Statistical features
    df = create_brand_model_stats(df)

    # Encoded features
    df = encode_categorical_features(df)

    # Time features
    df = create_time_features(df)
    df = create_time_aggregated_features(df)

    logger.info("=" * 60)
    logger.info(f"FEATURE ENGINEERING COMPLETE: {len(df)} rows, {len(df.columns)} columns")
    logger.info("=" * 60)

    # Save features CSV
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(ML_FEATURES_CSV_PATH, index=False)
    logger.info(f"Saved ML features to: {ML_FEATURES_CSV_PATH}")

    # Save to database
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        # Convert non-serializable columns for SQLite
        df_db = df.copy()
        for col in df_db.columns:
            if df_db[col].dtype.name == "category":
                df_db[col] = df_db[col].astype(str)
            elif df_db[col].dtype.name.startswith("datetime"):
                df_db[col] = df_db[col].astype(str)
        df_db.to_sql("ml_features", conn, if_exists="replace", index=False)
        logger.info("Saved ML features to database table 'ml_features'")
    finally:
        conn.close()

    return df
