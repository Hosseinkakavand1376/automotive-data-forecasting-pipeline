-- ============================================================================
-- Automotive Data Pipeline - Database Schema
-- ============================================================================
-- Creates all tables used by the pipeline.
-- Database: SQLite
-- ============================================================================

-- Table 1: Raw car listings (as ingested from CSV)
CREATE TABLE IF NOT EXISTS raw_car_listings (
    name            TEXT,
    year            INTEGER,
    selling_price   REAL,
    km_driven       INTEGER,
    fuel            TEXT,
    seller_type     TEXT,
    transmission    TEXT,
    owner           TEXT,
    listing_date    TEXT
);

-- Table 2: Cleaned car listings (after cleaning pipeline)
CREATE TABLE IF NOT EXISTS cleaned_car_listings (
    listing_id      INTEGER PRIMARY KEY,
    name            TEXT,
    year            INTEGER NOT NULL,
    selling_price   REAL NOT NULL,
    km_driven       INTEGER NOT NULL,
    fuel            TEXT NOT NULL,
    seller_type     TEXT NOT NULL,
    transmission    TEXT NOT NULL,
    owner           TEXT NOT NULL,
    brand           TEXT NOT NULL,
    model           TEXT NOT NULL,
    listing_date    TEXT
);

-- Table 3: ML features (engineered features for model training)
CREATE TABLE IF NOT EXISTS ml_features (
    listing_id              INTEGER PRIMARY KEY,
    name                    TEXT,
    year                    INTEGER,
    selling_price           REAL,
    km_driven               INTEGER,
    fuel                    TEXT,
    seller_type             TEXT,
    transmission            TEXT,
    owner                   TEXT,
    brand                   TEXT,
    model                   TEXT,
    listing_date            TEXT,
    car_age                 INTEGER,
    mileage_per_year        REAL,
    log_price               REAL,
    log_mileage             REAL,
    price_segment           TEXT,
    mileage_bucket          TEXT,
    brand_average_price     REAL,
    model_average_price     REAL,
    fuel_type_encoded       INTEGER,
    transmission_encoded    INTEGER,
    seller_type_encoded     INTEGER,
    owner_type_encoded      INTEGER,
    listing_month           INTEGER,
    listing_year            INTEGER,
    listing_season          TEXT,
    monthly_average_price   REAL,
    monthly_listing_count   INTEGER,
    brand_monthly_average_price   REAL,
    brand_monthly_listing_count   INTEGER
);

-- Table 4: Monthly forecasting data (aggregated time series)
CREATE TABLE IF NOT EXISTS monthly_forecasting_data (
    year_month      TEXT PRIMARY KEY,
    avg_price       REAL,
    listing_count   INTEGER,
    avg_mileage     REAL,
    avg_car_age     REAL,
    median_price    REAL,
    min_price       REAL,
    max_price       REAL,
    std_price       REAL
);

-- Table 5: Brand-monthly forecasting data
CREATE TABLE IF NOT EXISTS brand_monthly_forecasting_data (
    brand           TEXT,
    year_month      TEXT,
    avg_price       REAL,
    listing_count   INTEGER,
    avg_mileage     REAL,
    median_price    REAL,
    PRIMARY KEY (brand, year_month)
);
