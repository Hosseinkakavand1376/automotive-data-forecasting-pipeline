"""
Data Cleaning Module
=====================
Comprehensive cleaning pipeline for used-car listing data.
Handles duplicates, missing values, text normalization, outlier detection,
and standardization of categorical values.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from src.config import (
    CLEANED_CSV_PATH, DATABASE_PATH, PROCESSED_DATA_DIR,
    VALID_FUEL_TYPES, VALID_TRANSMISSIONS, VALID_SELLER_TYPES, VALID_OWNER_TYPES,
    MIN_VALID_YEAR, MAX_VALID_YEAR, MAX_VALID_MILEAGE, IQR_MULTIPLIER,
    CURRENT_YEAR,
)

logger = logging.getLogger(__name__)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    logger.info(f"Removed {n_removed} duplicate rows ({n_before} -> {len(df)})")
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    logger.info(f"Standardized column names: {list(df.columns)}")
    return df


def split_car_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the 'name' column into separate 'brand' and 'model' columns.
    The first word is treated as the brand, the rest as the model.
    """
    if "name" not in df.columns:
        logger.warning("Column 'name' not found — skipping name split")
        return df

    # Extract brand (first word) and model (remaining words)
    name_parts = df["name"].astype(str).str.strip().str.split(n=1)
    df["brand"] = name_parts.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown")
    df["model"] = name_parts.apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else "Unknown")

    logger.info(f"Split 'name' into 'brand' ({df['brand'].nunique()} unique) and 'model' ({df['model'].nunique()} unique)")
    return df


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text columns: strip whitespace and apply title case."""
    text_cols = ["brand", "model", "fuel", "seller_type", "transmission", "owner"]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            # Fix "Nan" strings from NaN conversion
            df[col] = df[col].replace("Nan", np.nan)

    logger.info("Normalized text columns (strip + title case)")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - Numeric columns: fill with median
    - Categorical columns: fill with mode
    """
    missing_before = df.isnull().sum().sum()

    # Numeric columns - fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            n_filled = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  Filled {n_filled} missing values in '{col}' with median ({median_val})")

    # Categorical columns - fill with mode
    cat_cols = ["fuel", "seller_type", "transmission", "owner", "brand", "model"]
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            n_filled = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  Filled {n_filled} missing values in '{col}' with mode ({mode_val})")

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    return df


def remove_invalid_years(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records with production years outside valid range."""
    if "year" not in df.columns:
        return df

    n_before = len(df)
    mask = (df["year"] >= MIN_VALID_YEAR) & (df["year"] <= MAX_VALID_YEAR)
    df = df[mask].reset_index(drop=True)
    n_removed = n_before - len(df)
    logger.info(f"Removed {n_removed} rows with invalid year (valid: {MIN_VALID_YEAR}-{MAX_VALID_YEAR})")
    return df


def remove_invalid_mileage(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records with negative or unrealistic mileage values."""
    if "km_driven" not in df.columns:
        return df

    n_before = len(df)
    mask = (df["km_driven"] >= 0) & (df["km_driven"] <= MAX_VALID_MILEAGE)
    df = df[mask].reset_index(drop=True)
    n_removed = n_before - len(df)
    logger.info(f"Removed {n_removed} rows with invalid mileage (valid: 0-{MAX_VALID_MILEAGE:,} km)")
    return df


def handle_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and handle price outliers using the IQR method.
    Prices beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR are capped.
    """
    if "selling_price" not in df.columns:
        return df

    # Remove non-positive prices first
    n_before = len(df)
    df = df[df["selling_price"] > 0].reset_index(drop=True)
    n_removed_zero = n_before - len(df)
    if n_removed_zero > 0:
        logger.info(f"Removed {n_removed_zero} rows with non-positive price")

    # IQR-based outlier capping
    Q1 = df["selling_price"].quantile(0.25)
    Q3 = df["selling_price"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR_MULTIPLIER * IQR
    upper_bound = Q3 + IQR_MULTIPLIER * IQR

    n_outliers = ((df["selling_price"] < lower_bound) | (df["selling_price"] > upper_bound)).sum()
    df["selling_price"] = df["selling_price"].clip(lower=max(0, lower_bound), upper=upper_bound)
    logger.info(f"Price outlier handling: {n_outliers} values capped (bounds: {lower_bound:,.0f} - {upper_bound:,.0f})")

    return df


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize categorical columns to valid category sets."""

    category_maps = {
        "fuel": VALID_FUEL_TYPES,
        "transmission": VALID_TRANSMISSIONS,
        "seller_type": VALID_SELLER_TYPES,
        "owner": VALID_OWNER_TYPES,
    }

    for col, valid_values in category_maps.items():
        if col not in df.columns:
            continue

        # Map close matches using title case
        valid_set = {v.lower(): v for v in valid_values}
        df[col] = df[col].apply(
            lambda x: valid_set.get(str(x).strip().lower(), x) if pd.notna(x) else x
        )

        # Log any values that don't match valid categories
        invalid_mask = ~df[col].isin(valid_values) & df[col].notna()
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            invalid_vals = df.loc[invalid_mask, col].unique()[:5]
            logger.warning(f"  {col}: {n_invalid} invalid values (examples: {invalid_vals})")
            # Replace invalid values with the most common valid one
            most_common = df.loc[df[col].isin(valid_values), col].mode()
            if not most_common.empty:
                df.loc[invalid_mask, col] = most_common.iloc[0]

    logger.info("Standardized categorical columns")
    return df


def generate_listing_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure listing_date column exists. If not present, generate
    realistic synthetic listing dates for demonstration purposes.
    """
    if "listing_date" in df.columns:
        df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
        n_invalid = df["listing_date"].isnull().sum()
        if n_invalid > 0:
            # Fill invalid dates with random dates
            rng = np.random.default_rng(42)
            start = pd.Timestamp("2020-01-01")
            end = pd.Timestamp("2024-12-31")
            random_dates = pd.to_datetime(
                rng.integers(start.value, end.value, size=n_invalid)
            )
            df.loc[df["listing_date"].isnull(), "listing_date"] = random_dates
        logger.info("Parsed existing listing_date column")
    else:
        # Generate synthetic listing dates
        rng = np.random.default_rng(42)
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2024-12-31")
        random_dates = pd.to_datetime(
            rng.integers(start.value, end.value, size=len(df))
        )
        df["listing_date"] = random_dates
        logger.info("Generated synthetic listing_date column")

    return df


def add_listing_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a unique listing_id column."""
    df.insert(0, "listing_id", range(1, len(df) + 1))
    logger.info(f"Added listing_id column (1 to {len(df)})")
    return df


def run_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the full data cleaning pipeline.

    Args:
        df: Raw DataFrame from ingestion.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA CLEANING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Input: {len(df)} rows, {len(df.columns)} columns")

    # Step 1: Remove duplicates
    df = remove_duplicates(df)

    # Step 2: Standardize column names
    df = standardize_column_names(df)

    # Step 3: Split car name into brand and model
    df = split_car_name(df)

    # Step 4: Normalize text columns
    df = normalize_text_columns(df)

    # Step 5: Handle missing values
    df = handle_missing_values(df)

    # Step 6: Remove invalid years
    df = remove_invalid_years(df)

    # Step 7: Remove invalid mileage
    df = remove_invalid_mileage(df)

    # Step 8: Handle price outliers
    df = handle_price_outliers(df)

    # Step 9: Standardize categorical values
    df = standardize_categories(df)

    # Step 10: Ensure listing_date exists
    df = generate_listing_date(df)

    # Step 11: Add unique listing ID
    df = add_listing_id(df)

    logger.info("=" * 60)
    logger.info(f"CLEANING COMPLETE: {len(df)} rows, {len(df.columns)} columns")
    logger.info("=" * 60)

    # Save cleaned data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(CLEANED_CSV_PATH, index=False)
    logger.info(f"Saved cleaned data to: {CLEANED_CSV_PATH}")

    # Save to database
    import sqlite3
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        df.to_sql("cleaned_car_listings", conn, if_exists="replace", index=False)
        logger.info(f"Saved cleaned data to database table 'cleaned_car_listings'")
    finally:
        conn.close()

    return df
