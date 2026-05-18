"""
Data Ingestion Module
======================
Handles loading raw CSV data, performing initial schema inspection,
and saving the raw data into a SQLite database for structured storage.
"""

import os
import logging
import sqlite3
import pandas as pd

from src.config import (
    RAW_CSV_PATH, DATABASE_PATH, LOG_FORMAT, LOG_DATE_FORMAT
)

logger = logging.getLogger(__name__)


def load_raw_csv(filepath: str = RAW_CSV_PATH) -> pd.DataFrame:
    """
    Load the raw CSV dataset from disk.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        DataFrame containing the raw data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw CSV not found at: {filepath}")

    logger.info(f"Loading raw CSV from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    return df


def inspect_schema(df: pd.DataFrame) -> dict:
    """
    Perform initial schema inspection and log key statistics.

    Args:
        df: The raw DataFrame to inspect.

    Returns:
        Dictionary containing schema inspection results.
    """
    inspection = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }

    logger.info("=" * 60)
    logger.info("SCHEMA INSPECTION REPORT")
    logger.info("=" * 60)
    logger.info(f"  Shape: {inspection['shape'][0]} rows x {inspection['shape'][1]} columns")
    logger.info(f"  Memory usage: {inspection['memory_usage_mb']} MB")
    logger.info(f"  Total missing values: {inspection['total_missing']}")
    logger.info(f"  Duplicate rows: {inspection['duplicate_rows']}")
    logger.info("-" * 60)
    logger.info("  Column types:")
    for col, dtype in inspection["dtypes"].items():
        missing = inspection["missing_values"].get(col, 0)
        logger.info(f"    {col:25s} | {dtype:10s} | missing: {missing}")
    logger.info("=" * 60)

    return inspection


def save_to_database(df: pd.DataFrame, table_name: str = "raw_car_listings",
                     db_path: str = DATABASE_PATH) -> None:
    """
    Save a DataFrame to a SQLite database table.

    Args:
        df: DataFrame to save.
        table_name: Name of the database table.
        db_path: Path to the SQLite database file.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    logger.info(f"Saving {len(df)} rows to database table '{table_name}'...")
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        logger.info(f"Successfully saved to '{table_name}' in {db_path}")
    finally:
        conn.close()


def run_ingestion(filepath: str = RAW_CSV_PATH) -> pd.DataFrame:
    """
    Execute the full ingestion pipeline:
    1. Load raw CSV
    2. Inspect schema
    3. Save to database

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        The raw DataFrame.
    """
    logger.info("Starting data ingestion pipeline...")

    # Step 1: Load raw data
    df = load_raw_csv(filepath)

    # Step 2: Schema inspection
    inspection = inspect_schema(df)

    # Step 3: Save to database
    save_to_database(df, table_name="raw_car_listings")

    logger.info("Data ingestion complete.")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    df = run_ingestion()
