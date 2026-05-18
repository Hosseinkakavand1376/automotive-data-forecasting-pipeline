"""
Central configuration for the Automotive Data Pipeline.
All paths, constants, valid categories, and model parameters are defined here.
No hardcoded paths should exist outside this file.
"""

import os
from datetime import datetime

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Project root directory (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FINAL_DATA_DIR = os.path.join(DATA_DIR, "final")

# Raw dataset path
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, "used_cars_raw.csv")

# Processed output paths
CLEANED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_cars.csv")
ML_FEATURES_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "ml_features.csv")

# Final output paths
FORECASTING_CSV_PATH = os.path.join(FINAL_DATA_DIR, "automotive_monthly_forecasting_dataset.csv")

# Database
DATABASE_PATH = os.path.join(DATA_DIR, "automotive_pipeline.db")

# Models
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")

# Reports
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
VALIDATION_REPORT_PATH = os.path.join(REPORTS_DIR, "validation_report.json")
MODEL_METRICS_PATH = os.path.join(REPORTS_DIR, "model_metrics.json")

# SQL
SQL_DIR = os.path.join(BASE_DIR, "sql")

# ============================================================================
# DATASET GENERATION PARAMETERS
# ============================================================================

SYNTHETIC_NUM_ROWS = 8000

# Current year for calculations
CURRENT_YEAR = datetime.now().year

# ============================================================================
# VALID CATEGORIES
# ============================================================================

VALID_FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]

VALID_TRANSMISSIONS = ["Manual", "Automatic"]

VALID_SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]

VALID_OWNER_TYPES = [
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
    "Test Drive Car",
]

# ============================================================================
# DATA CLEANING PARAMETERS
# ============================================================================

# Year range for valid production years
MIN_VALID_YEAR = 1990
MAX_VALID_YEAR = CURRENT_YEAR + 1

# Maximum realistic mileage (km)
MAX_VALID_MILEAGE = 1_000_000

# IQR multiplier for outlier detection
IQR_MULTIPLIER = 1.5

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# Price segment quantile labels
PRICE_SEGMENTS = ["Budget", "Economy", "Mid-Range", "Premium"]

# Mileage bucket bins (in km) and labels
MILEAGE_BINS = [0, 25_000, 75_000, 150_000, float("inf")]
MILEAGE_LABELS = ["Low", "Medium", "High", "Very High"]

# Season mapping (month -> season)
SEASON_MAP = {
    1: "Winter", 2: "Winter", 3: "Spring",
    4: "Spring", 5: "Spring", 6: "Summer",
    7: "Summer", 8: "Summer", 9: "Autumn",
    10: "Autumn", 11: "Autumn", 12: "Winter",
}

# ============================================================================
# ML MODEL PARAMETERS
# ============================================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Features to use for ML training
ML_NUMERIC_FEATURES = [
    "year", "km_driven", "car_age", "mileage_per_year",
    "log_price", "log_mileage",
    "brand_average_price", "model_average_price",
]

ML_ENCODED_FEATURES = [
    "fuel_type_encoded", "transmission_encoded",
    "seller_type_encoded", "owner_type_encoded",
]

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"
