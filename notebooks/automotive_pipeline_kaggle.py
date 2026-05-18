"""
Automotive Data Preparation & Forecasting Pipeline - Kaggle Notebook
=====================================================================
This script contains the full pipeline ready to run on Kaggle.
Copy each section into separate Kaggle notebook cells.

To use on Kaggle:
1. Create a new notebook
2. Upload the automotive-data-pipeline folder as a Kaggle dataset
3. Copy this code into cells (split at '# %% [markdown]' markers)
4. Run All
"""

# %% [markdown]
# # Automotive Data Preparation & Forecasting Pipeline
# End-to-end data engineering pipeline for used-car price prediction and demand forecasting.

# %% Setup & Imports
import os, sys, json, time, logging, sqlite3, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("pipeline")

# Detect environment
IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    BASE_DIR = "/kaggle/working/automotive-data-pipeline"
    os.makedirs(BASE_DIR, exist_ok=True)
    # Try to import from uploaded dataset
    dataset_paths = [
        "/kaggle/input/automotive-data-pipeline/automotive-data-pipeline/src",
        "/kaggle/input/automotive-data-pipeline/src",
    ]
    for p in dataset_paths:
        if os.path.exists(p):
            sys.path.insert(0, os.path.dirname(p))
            break
    else:
        logger.info("Source modules not found as Kaggle dataset - using inline code")
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create output directories
for d in ["data/raw", "data/processed", "data/final", "models", "reports"]:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"Base directory: {BASE_DIR}")

# %% Try importing from src/ modules, fall back to inline
try:
    from src.config import *
    from src.generate_dataset import generate_dataset
    from src.ingestion.ingest import run_ingestion
    from src.cleaning.clean import run_cleaning
    from src.validation.validate import run_validation
    from src.features.engineer_features import run_feature_engineering
    from src.features.prepare_forecasting import run_forecasting_preparation
    from src.modeling.train_model import run_model_training
    from src.transformation.transform import run_full_pipeline
    MODULES_AVAILABLE = True
    print("Loaded pipeline modules from src/")
except ImportError:
    MODULES_AVAILABLE = False
    print("Running with inline pipeline code")

# %% Configuration (used when modules not available)
CURRENT_YEAR = datetime.now().year
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "used_cars_raw.csv")
CLEANED_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_cars.csv")
ML_FEATURES_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "ml_features.csv")
FORECASTING_CSV_PATH = os.path.join(BASE_DIR, "data", "final", "automotive_monthly_forecasting_dataset.csv")
DATABASE_PATH = os.path.join(BASE_DIR, "data", "automotive_pipeline.db")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
VALIDATION_REPORT_PATH = os.path.join(BASE_DIR, "reports", "validation_report.json")
MODEL_METRICS_PATH = os.path.join(BASE_DIR, "reports", "model_metrics.json")

VALID_FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
VALID_TRANSMISSIONS = ["Manual", "Automatic"]
VALID_SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]
VALID_OWNER_TYPES = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
SEASON_MAP = {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn",12:"Winter"}

# %% [markdown]
# ## Stage 1: Data Generation / Loading

# %% Stage 1 - Generate or Load Dataset
if MODULES_AVAILABLE and not os.path.exists(RAW_CSV_PATH):
    df_raw = generate_dataset()
elif os.path.exists(RAW_CSV_PATH):
    df_raw = pd.read_csv(RAW_CSV_PATH)
    print(f"Loaded existing dataset: {df_raw.shape}")
else:
    # Inline dataset generation
    print("Generating synthetic dataset inline...")
    from src.generate_dataset import generate_dataset
    # If import still fails, the full generate code is in src/generate_dataset.py
    df_raw = generate_dataset()

print(f"\nRaw dataset shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
print(f"Missing values:\n{df_raw.isnull().sum()}")
print(f"Duplicates: {df_raw.duplicated().sum()}")
df_raw.head()

# %% [markdown]
# ## Stage 2: Run Full Pipeline

# %% Stage 2 - Run Pipeline
if MODULES_AVAILABLE:
    results = run_full_pipeline(RAW_CSV_PATH)
    print("\nPipeline Results:")
    for stage, info in results.items():
        if isinstance(info, dict) and "status" in info:
            print(f"  {stage}: {info['status']}")
else:
    print("Modules not available - run the pipeline locally first")

# %% [markdown]
# ## Stage 3: Explore Results

# %% Load cleaned data
df_cleaned = pd.read_csv(CLEANED_CSV_PATH)
print(f"Cleaned dataset: {df_cleaned.shape}")
df_cleaned.head()

# %% Load ML features
df_features = pd.read_csv(ML_FEATURES_CSV_PATH)
print(f"ML features dataset: {df_features.shape}")
print(f"Feature columns: {list(df_features.columns)}")

# %% Load forecasting data
df_forecast = pd.read_csv(FORECASTING_CSV_PATH)
print(f"Forecasting dataset: {df_forecast.shape}")
df_forecast.head()

# %% [markdown]
# ## Stage 4: EDA Visualizations

# %% Price Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_cleaned['selling_price'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Selling Price Distribution')
axes[0].set_xlabel('Price (EUR)')
axes[1].hist(np.log1p(df_cleaned['selling_price']), bins=50, color='coral', edgecolor='white')
axes[1].set_title('Log(Selling Price) Distribution')
axes[1].set_xlabel('Log Price')
plt.tight_layout()
plt.show()

# %% Top Brands
brand_stats = df_cleaned.groupby('brand').agg(
    count=('selling_price', 'count'),
    avg_price=('selling_price', 'mean')
).sort_values('count', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(brand_stats.index, brand_stats['count'], color='steelblue')
ax.set_title('Top 10 Brands by Listing Count')
ax.set_ylabel('Number of Listings')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% Price by Fuel Type
fig, ax = plt.subplots(figsize=(10, 5))
df_cleaned.boxplot(column='selling_price', by='fuel', ax=ax)
ax.set_title('Price Distribution by Fuel Type')
ax.set_ylabel('Price (EUR)')
plt.suptitle('')
plt.tight_layout()
plt.show()

# %% Monthly Trends
if 'year_month' in df_forecast.columns or 'avg_price' in df_forecast.columns:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(df_forecast.index, df_forecast['avg_price'], marker='o', markersize=3, color='steelblue')
    axes[0].set_title('Average Price Over Time')
    axes[0].set_ylabel('Avg Price (EUR)')
    axes[1].plot(df_forecast.index, df_forecast['listing_count'], marker='o', markersize=3, color='coral')
    axes[1].set_title('Monthly Listing Count')
    axes[1].set_ylabel('Count')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Stage 5: ML Results

# %% Model Metrics
with open(MODEL_METRICS_PATH, 'r') as f:
    metrics = json.load(f)

print("=" * 50)
print("ML BASELINE RESULTS")
print("=" * 50)
for m in metrics['models']:
    print(f"\n{m['model']}:")
    print(f"  MAE:  EUR {m['mae']:,.0f}")
    print(f"  RMSE: EUR {m['rmse']:,.0f}")
    print(f"  R2:   {m['r2_score']:.4f}")

print(f"\nBest Model: {metrics['best_model']['model']} (R2 = {metrics['best_model']['r2_score']})")

# %% Validation Report
with open(VALIDATION_REPORT_PATH, 'r') as f:
    val_report = json.load(f)

print(f"\nValidation: {val_report['summary']['overall_status']}")
print(f"Checks: {val_report['summary']['passed']}/{val_report['summary']['total_checks']} passed")
for check in val_report['checks']:
    print(f"  [{check['status']}] {check['rule']}: {check['details']}")

# %% [markdown]
# ## Stage 6: API Documentation
# 
# The FastAPI service provides these endpoints (run locally or via Docker):
# 
# ```bash
# # Start server
# uvicorn src.api.app:app --reload --port 8000
# 
# # Predict price
# curl -X POST http://localhost:8000/predict-price \
#   -H "Content-Type: application/json" \
#   -d '{"year": 2018, "km_driven": 45000, "fuel": "Petrol", 
#        "transmission": "Manual", "seller_type": "Individual",
#        "owner": "First Owner"}'
# ```

# %% Summary
print("\n" + "=" * 60)
print("PIPELINE COMPLETE - Output Files:")
print("=" * 60)
for name, path in [
    ("Raw Data", RAW_CSV_PATH),
    ("Cleaned Data", CLEANED_CSV_PATH),
    ("ML Features", ML_FEATURES_CSV_PATH),
    ("Forecasting", FORECASTING_CSV_PATH),
    ("Database", DATABASE_PATH),
    ("Model", BEST_MODEL_PATH),
    ("Validation", VALIDATION_REPORT_PATH),
    ("Metrics", MODEL_METRICS_PATH),
]:
    exists = "[OK]" if os.path.exists(path) else "[--]"
    print(f"  {exists} {name:20s} -> {path}")
