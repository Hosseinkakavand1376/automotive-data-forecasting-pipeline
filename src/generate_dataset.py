"""
Synthetic Used-Car Dataset Generator
=====================================
Generates a realistic dataset of ~8,000 used-car listings modeled after
European market data. This ensures the project is fully
self-contained and reproducible without needing external data downloads.

Features:
- Correlated price distributions (luxury brands → higher prices)
- Realistic mileage based on car age
- Synthetic listing_date for time-series features
- Intentionally injected noise: ~3% missing values, ~2% duplicates, some outliers
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_CSV_PATH, RAW_DATA_DIR, SYNTHETIC_NUM_ROWS

logger = logging.getLogger(__name__)

# ============================================================================
# BRAND AND MODEL DEFINITIONS
# ============================================================================

# Brand data: {brand: (market_share_weight, base_price_thousands_eur, models_list)}
BRAND_DATA = {
    "Maruti": (0.22, 5.0, [
        "Swift", "Baleno", "Alto", "WagonR", "Dzire", "Vitara Brezza",
        "Ciaz", "Ertiga", "S-Presso", "Celerio", "Ignis",
    ]),
    "Hyundai": (0.18, 7.0, [
        "i20", "Creta", "Venue", "Grand i10", "Verna", "Tucson",
        "Santro", "Aura", "Xcent", "Elite i20",
    ]),
    "Tata": (0.12, 6.5, [
        "Nexon", "Punch", "Harrier", "Safari", "Tiago", "Tigor",
        "Altroz", "Hexa",
    ]),
    "Mahindra": (0.10, 8.0, [
        "XUV500", "Scorpio", "Thar", "Bolero", "XUV300", "Marazzo",
        "KUV100", "TUV300",
    ]),
    "Honda": (0.08, 8.5, [
        "City", "Amaze", "WR-V", "Jazz", "Civic", "CR-V", "BR-V",
    ]),
    "Toyota": (0.07, 10.0, [
        "Innova Crysta", "Fortuner", "Glanza", "Urban Cruiser",
        "Yaris", "Etios", "Camry",
    ]),
    "Ford": (0.05, 7.5, [
        "EcoSport", "Endeavour", "Figo", "Aspire", "Freestyle",
    ]),
    "Kia": (0.05, 9.0, [
        "Seltos", "Sonet", "Carnival",
    ]),
    "Volkswagen": (0.04, 9.0, [
        "Polo", "Vento", "Taigun", "Tiguan",
    ]),
    "Renault": (0.03, 5.5, [
        "Kwid", "Triber", "Kiger", "Duster",
    ]),
    "BMW": (0.02, 30.0, [
        "3 Series", "5 Series", "X1", "X3", "X5", "7 Series",
    ]),
    "Mercedes-Benz": (0.02, 35.0, [
        "C-Class", "E-Class", "GLC", "GLE", "A-Class", "S-Class",
    ]),
    "Audi": (0.01, 28.0, [
        "A4", "A6", "Q3", "Q5", "Q7", "A3",
    ]),
    "Skoda": (0.01, 10.0, [
        "Rapid", "Superb", "Octavia", "Kushaq",
    ]),
}


def _generate_base_records(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the base records with correlated features."""
    brands = list(BRAND_DATA.keys())
    weights = [BRAND_DATA[b][0] for b in brands]

    # Normalize weights
    weights = np.array(weights) / sum(weights)

    # Sample brands based on market share
    sampled_brands = rng.choice(brands, size=n_rows, p=weights)

    records = []
    for brand in sampled_brands:
        _, base_price, models = BRAND_DATA[brand]
        model = rng.choice(models)

        # Generate production year (weighted toward recent years)
        year_weights = np.array([1.0 + 0.3 * i for i in range(22)])  # 2003-2024
        year_weights /= year_weights.sum()
        year = rng.choice(range(2003, 2025), p=year_weights)

        # Car age
        car_age = 2024 - year

        # Generate mileage based on car age (avg ~12,000 km/year with variance)
        avg_annual_km = rng.normal(12000, 4000)
        avg_annual_km = max(1000, avg_annual_km)
        km_driven = int(avg_annual_km * max(car_age, 0.5))
        km_driven = max(500, km_driven)

        # Fuel type (brand-dependent)
        if brand in ["BMW", "Mercedes-Benz", "Audi"]:
            fuel = rng.choice(
                ["Petrol", "Diesel"],
                p=[0.45, 0.55],
            )
        else:
            fuel = rng.choice(
                ["Petrol", "Diesel", "CNG", "LPG", "Electric"],
                p=[0.50, 0.38, 0.07, 0.03, 0.02],
            )

        # Transmission (luxury brands more likely automatic)
        if brand in ["BMW", "Mercedes-Benz", "Audi"]:
            transmission = rng.choice(["Manual", "Automatic"], p=[0.15, 0.85])
        else:
            transmission = rng.choice(["Manual", "Automatic"], p=[0.72, 0.28])

        # Seller type
        seller_type = rng.choice(
            ["Individual", "Dealer", "Trustmark Dealer"],
            p=[0.55, 0.35, 0.10],
        )

        # Owner type
        owner = rng.choice(
            ["First Owner", "Second Owner", "Third Owner",
             "Fourth & Above Owner", "Test Drive Car"],
            p=[0.58, 0.25, 0.10, 0.05, 0.02],
        )

        # Price calculation (in thousands EUR) - correlated with brand, age, fuel, etc.
        depreciation = max(0.1, 1.0 - car_age * 0.08)  # ~8% per year
        fuel_factor = {"Petrol": 1.0, "Diesel": 1.15, "CNG": 0.85,
                       "LPG": 0.80, "Electric": 1.30}.get(fuel, 1.0)
        trans_factor = 1.15 if transmission == "Automatic" else 1.0
        owner_factor = {"First Owner": 1.0, "Second Owner": 0.88,
                        "Third Owner": 0.75, "Fourth & Above Owner": 0.65,
                        "Test Drive Car": 0.95}.get(owner, 0.8)

        # Base price with random noise
        price_k = base_price * depreciation * fuel_factor * trans_factor * owner_factor
        price_k *= rng.normal(1.0, 0.15)  # 15% random variance
        price_k = max(0.3, price_k)  # Minimum 300 EUR

        # Convert to actual selling price (in EUR)
        selling_price = int(price_k * 1_000)

        # Construct car name (brand + model + year suffix)
        name = f"{brand} {model}"

        # Listing date (2020-01-01 to 2024-12-31)
        days_range = (datetime(2024, 12, 31) - datetime(2020, 1, 1)).days
        listing_date = datetime(2020, 1, 1) + timedelta(days=int(rng.integers(0, days_range)))

        records.append({
            "name": name,
            "year": year,
            "selling_price": selling_price,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "listing_date": listing_date.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(records)


def _inject_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject realistic noise: missing values, duplicates, and outliers."""
    df = df.copy()
    n = len(df)

    # --- Inject ~3% missing values across selected columns ---
    cols_for_nulls = ["selling_price", "km_driven", "fuel", "seller_type", "owner"]
    for col in cols_for_nulls:
        null_mask = rng.random(n) < 0.03
        df.loc[null_mask, col] = np.nan

    # --- Inject ~2% duplicate rows ---
    n_duplicates = int(n * 0.02)
    duplicate_indices = rng.choice(n, size=n_duplicates, replace=True)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # --- Inject some price outliers (unrealistically high) ---
    n_outliers = int(n * 0.005)  # 0.5% outliers
    outlier_indices = rng.choice(len(df), size=n_outliers, replace=False)
    df.loc[outlier_indices, "selling_price"] = df.loc[outlier_indices, "selling_price"].apply(
        lambda x: x * rng.uniform(5, 15) if pd.notna(x) else x
    )

    # --- Inject a few negative/zero mileage values ---
    n_bad_mileage = int(n * 0.003)
    bad_mileage_idx = rng.choice(len(df), size=n_bad_mileage, replace=False)
    df.loc[bad_mileage_idx, "km_driven"] = rng.choice([-100, -500, 0], size=n_bad_mileage)

    # --- Inject a few invalid years ---
    n_bad_years = int(n * 0.002)
    bad_year_idx = rng.choice(len(df), size=n_bad_years, replace=False)
    df.loc[bad_year_idx, "year"] = rng.choice([1950, 1975, 2030, 2035], size=n_bad_years)

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def generate_dataset(
    n_rows: int = SYNTHETIC_NUM_ROWS,
    output_path: str = RAW_CSV_PATH,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic used-car dataset and save to CSV.

    Args:
        n_rows: Number of base records to generate.
        output_path: File path to save the CSV.
        seed: Random seed for reproducibility.

    Returns:
        The generated DataFrame (with noise injected).
    """
    logger.info(f"Generating synthetic dataset with {n_rows} base rows...")
    rng = np.random.default_rng(seed)

    # Generate base records
    df = _generate_base_records(n_rows, rng)
    logger.info(f"Base records generated: {len(df)} rows, {len(df.columns)} columns")

    # Inject noise for realistic data engineering demonstration
    df = _inject_noise(df, rng)
    logger.info(f"After noise injection: {len(df)} rows")
    logger.info(f"  Missing values: {df.isnull().sum().sum()}")
    logger.info(f"  Duplicate rows: {df.duplicated().sum()}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to: {output_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    df = generate_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nSample rows:\n{df.head()}")
