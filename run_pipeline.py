"""
Pipeline Runner
================
One-click script to execute the complete Automotive Data Pipeline.
Handles dataset generation (if needed) and runs all pipeline stages.

Usage:
    python run_pipeline.py
"""

import os
import sys
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import RAW_CSV_PATH, LOG_FORMAT, LOG_DATE_FORMAT


def main():
    """Run the complete pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )
    logger = logging.getLogger("pipeline")

    logger.info("=" * 60)
    logger.info("AUTOMOTIVE DATA PIPELINE - Starting...")
    logger.info("=" * 60)

    # Step 0: Generate dataset if it doesn't exist
    if not os.path.exists(RAW_CSV_PATH):
        logger.info("Raw dataset not found — generating synthetic data...")
        from src.generate_dataset import generate_dataset
        generate_dataset()
    else:
        logger.info(f"Using existing dataset: {RAW_CSV_PATH}")

    # Step 1-6: Run the full pipeline
    from src.transformation.transform import run_full_pipeline
    results = run_full_pipeline(RAW_CSV_PATH)

    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nTotal duration: {results.get('pipeline_duration_sec', 'N/A')}s")

    # Print key output files
    from src.config import (
        CLEANED_CSV_PATH, ML_FEATURES_CSV_PATH, FORECASTING_CSV_PATH,
        DATABASE_PATH, VALIDATION_REPORT_PATH, MODEL_METRICS_PATH,
        BEST_MODEL_PATH,
    )

    print("\nOutput files:")
    outputs = [
        ("Cleaned data", CLEANED_CSV_PATH),
        ("ML features", ML_FEATURES_CSV_PATH),
        ("Forecasting data", FORECASTING_CSV_PATH),
        ("Database", DATABASE_PATH),
        ("Validation report", VALIDATION_REPORT_PATH),
        ("Model metrics", MODEL_METRICS_PATH),
        ("Best model", BEST_MODEL_PATH),
    ]
    for name, path in outputs:
        exists = "[OK]" if os.path.exists(path) else "[--]"
        print(f"  {exists} {name:25s} -> {path}")

    print("\n" + "=" * 60)

    # Print ML results
    if "ml_training" in results and results["ml_training"]["status"] == "success":
        print(f"\nBest ML Model: {results['ml_training']['best_model']}")
        print(f"  R2 Score: {results['ml_training']['best_r2']}")
        print(f"  MAE: EUR {results['ml_training']['best_mae']:,.0f}")

    print("\nTo start the API server, run:")
    print("  uvicorn src.api.app:app --reload --port 8000")
    print()

    return results


if __name__ == "__main__":
    main()
