"""
ML Baseline Model Training
============================
Trains simple baseline models for used-car price prediction:
- Linear Regression
- Random Forest Regressor

Evaluates with MAE, RMSE, and R² score.
Saves the best model and metrics report.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    BEST_MODEL_PATH, MODEL_METRICS_PATH, MODELS_DIR, REPORTS_DIR,
    ML_NUMERIC_FEATURES, ML_ENCODED_FEATURES,
    TEST_SIZE, RANDOM_STATE, RF_PARAMS,
)

logger = logging.getLogger(__name__)


def prepare_ml_data(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for ML training.

    Args:
        df: DataFrame with engineered features.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    # Define feature columns (only use those that exist in the dataframe)
    all_features = ML_NUMERIC_FEATURES + ML_ENCODED_FEATURES
    available_features = [f for f in all_features if f in df.columns]

    # Remove log_price from features if we're predicting selling_price
    # (it's a direct transformation of the target)
    if "log_price" in available_features:
        available_features.remove("log_price")

    logger.info(f"Using {len(available_features)} features: {available_features}")

    # Target variable
    target = "selling_price"

    # Drop rows with any NaN in features or target
    subset = df[available_features + [target]].dropna()
    logger.info(f"Training data after dropping NaN: {len(subset)} rows")

    X = subset[available_features]
    y = subset[target]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info(f"Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")

    return X_train, X_test, y_train, y_test, available_features


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a model and return metrics.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test target.
        model_name: Name of the model for logging.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "model": model_name,
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2_score": round(float(r2), 4),
    }

    logger.info(f"  {model_name} Results:")
    logger.info(f"    MAE:  {mae:>12,.2f}")
    logger.info(f"    RMSE: {rmse:>12,.2f}")
    logger.info(f"    R²:   {r2:>12.4f}")

    return metrics


def run_model_training(df: pd.DataFrame) -> dict:
    """
    Train baseline models, evaluate them, and save the best one.

    Args:
        df: DataFrame with engineered features.

    Returns:
        Dictionary with model metrics and results.
    """
    logger.info("=" * 60)
    logger.info("STARTING ML BASELINE TRAINING")
    logger.info("=" * 60)

    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_ml_data(df)

    # --- Model 1: Linear Regression ---
    logger.info("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Linear Regression")

    # --- Model 2: Random Forest Regressor ---
    logger.info("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # --- Select best model ---
    all_metrics = [lr_metrics, rf_metrics]
    best = max(all_metrics, key=lambda x: x["r2_score"])
    best_model = rf_model if best["model"] == "Random Forest" else lr_model

    logger.info(f"\nBest model: {best['model']} (R² = {best['r2_score']})")

    # --- Save best model ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({
        "model": best_model,
        "feature_names": feature_names,
        "model_name": best["model"],
    }, BEST_MODEL_PATH)
    logger.info(f"Saved best model to: {BEST_MODEL_PATH}")

    # --- Save metrics report ---
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        "training_info": {
            "total_samples": len(X_train) + len(X_test),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_size": TEST_SIZE,
            "features_used": feature_names,
            "target": "selling_price",
        },
        "models": all_metrics,
        "best_model": best,
    }

    with open(MODEL_METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved model metrics to: {MODEL_METRICS_PATH}")

    logger.info("=" * 60)
    logger.info("ML BASELINE TRAINING COMPLETE")
    logger.info("=" * 60)

    return report
