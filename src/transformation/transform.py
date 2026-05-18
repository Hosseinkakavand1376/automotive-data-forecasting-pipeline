"""
Data Transformation Orchestrator
==================================
Coordinates the full data pipeline from ingestion to feature engineering.
This module ties together all pipeline stages in the correct order.
"""

import logging
import time

from src.config import LOG_FORMAT, LOG_DATE_FORMAT, RAW_CSV_PATH
from src.ingestion.ingest import run_ingestion
from src.cleaning.clean import run_cleaning
from src.validation.validate import run_validation
from src.features.engineer_features import run_feature_engineering
from src.features.prepare_forecasting import run_forecasting_preparation
from src.modeling.train_model import run_model_training

logger = logging.getLogger(__name__)


def run_full_pipeline(raw_csv_path: str = RAW_CSV_PATH) -> dict:
    """
    Execute the complete data pipeline end-to-end.

    Pipeline stages:
    1. Data Ingestion — Load CSV, inspect schema, save to DB
    2. Data Cleaning — Deduplicate, normalize, handle outliers
    3. Data Validation — Automated quality checks
    4. Feature Engineering — Create ML-ready features
    5. Forecasting Preparation — Monthly aggregations
    6. ML Baseline Training — Train and evaluate models

    Args:
        raw_csv_path: Path to the raw CSV file.

    Returns:
        Dictionary with results from each pipeline stage.
    """
    pipeline_start = time.time()
    results = {}

    logger.info("=" * 60)
    logger.info("   AUTOMOTIVE DATA PREPARATION & FORECASTING PIPELINE")
    logger.info("=" * 60)

    # ================================================================
    # STAGE 1: DATA INGESTION
    # ================================================================
    try:
        stage_start = time.time()
        logger.info("\n>> STAGE 1/6: DATA INGESTION")
        raw_df = run_ingestion(raw_csv_path)
        results["ingestion"] = {
            "status": "success",
            "rows": len(raw_df),
            "columns": len(raw_df.columns),
            "duration_sec": round(time.time() - stage_start, 2),
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        results["ingestion"] = {"status": "failed", "error": str(e)}
        raise

    # ================================================================
    # STAGE 2: DATA CLEANING
    # ================================================================
    try:
        stage_start = time.time()
        logger.info("\n>> STAGE 2/6: DATA CLEANING")
        cleaned_df = run_cleaning(raw_df)
        results["cleaning"] = {
            "status": "success",
            "rows_before": len(raw_df),
            "rows_after": len(cleaned_df),
            "rows_removed": len(raw_df) - len(cleaned_df),
            "duration_sec": round(time.time() - stage_start, 2),
        }
    except Exception as e:
        logger.error(f"Cleaning failed: {e}")
        results["cleaning"] = {"status": "failed", "error": str(e)}
        raise

    # ================================================================
    # STAGE 3: DATA VALIDATION
    # ================================================================
    try:
        stage_start = time.time()
        logger.info("\n>> STAGE 3/6: DATA VALIDATION")
        validation_report = run_validation(cleaned_df)
        results["validation"] = {
            "status": "success",
            "overall": validation_report["summary"]["overall_status"],
            "checks_passed": validation_report["summary"]["passed"],
            "checks_total": validation_report["summary"]["total_checks"],
            "duration_sec": round(time.time() - stage_start, 2),
        }
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        results["validation"] = {"status": "failed", "error": str(e)}
        raise

    # ================================================================
    # STAGE 4: FEATURE ENGINEERING
    # ================================================================
    try:
        stage_start = time.time()
        logger.info("\n>> STAGE 4/6: FEATURE ENGINEERING")
        features_df = run_feature_engineering(cleaned_df)
        results["feature_engineering"] = {
            "status": "success",
            "total_features": len(features_df.columns),
            "new_features": len(features_df.columns) - len(cleaned_df.columns),
            "duration_sec": round(time.time() - stage_start, 2),
        }
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        results["feature_engineering"] = {"status": "failed", "error": str(e)}
        raise

    # ================================================================
    # STAGE 5: FORECASTING PREPARATION
    # ================================================================
    try:
        stage_start = time.time()
        logger.info("\n>> STAGE 5/6: FORECASTING PREPARATION")
        forecast_data = run_forecasting_preparation(features_df)
        results["forecasting"] = {
            "status": "success",
            "monthly_records": len(forecast_data["monthly"]),
            "brand_monthly_records": len(forecast_data["brand_monthly"]),
            "duration_sec": round(time.time() - stage_start, 2),
        }
    except Exception as e:
        logger.error(f"Forecasting preparation failed: {e}")
        results["forecasting"] = {"status": "failed", "error": str(e)}
        raise

    # ================================================================
    # STAGE 6: ML BASELINE TRAINING
    # ================================================================
    try:
        stage_start = time.time()
        logger.info("\n>> STAGE 6/6: ML BASELINE TRAINING")
        ml_report = run_model_training(features_df)
        results["ml_training"] = {
            "status": "success",
            "best_model": ml_report["best_model"]["model"],
            "best_r2": ml_report["best_model"]["r2_score"],
            "best_mae": ml_report["best_model"]["mae"],
            "duration_sec": round(time.time() - stage_start, 2),
        }
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        results["ml_training"] = {"status": "failed", "error": str(e)}
        raise

    # ================================================================
    # PIPELINE SUMMARY
    # ================================================================
    total_duration = round(time.time() - pipeline_start, 2)
    results["pipeline_duration_sec"] = total_duration

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    for stage, info in results.items():
        if isinstance(info, dict) and "status" in info:
            status = "[OK]" if info["status"] == "success" else "[FAIL]"
            duration = info.get("duration_sec", "N/A")
            logger.info(f"  {status} {stage:25s} | {info['status']:8s} | {duration}s")
    logger.info(f"\n  Total pipeline duration: {total_duration}s")
    logger.info("=" * 60)

    return results
