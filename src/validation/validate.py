"""
Data Validation Module
=======================
Automated validation checks to ensure data quality after cleaning.
Generates a validation report with pass/fail status for each rule.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime

from src.config import (
    VALIDATION_REPORT_PATH, REPORTS_DIR,
    VALID_FUEL_TYPES, VALID_TRANSMISSIONS, VALID_SELLER_TYPES, VALID_OWNER_TYPES,
    MIN_VALID_YEAR, MAX_VALID_YEAR,
)

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for a single validation check result."""

    def __init__(self, rule_name: str, passed: bool, details: str, violations: int = 0):
        self.rule_name = rule_name
        self.passed = passed
        self.details = details
        self.violations = violations

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_name,
            "status": "PASS" if self.passed else "FAIL",
            "details": self.details,
            "violations": self.violations,
        }


def check_positive_price(df: pd.DataFrame) -> ValidationResult:
    """Validate that all prices are positive."""
    violations = (df["selling_price"] <= 0).sum()
    return ValidationResult(
        rule_name="positive_price",
        passed=violations == 0,
        details=f"{violations} records with non-positive price",
        violations=violations,
    )


def check_valid_year(df: pd.DataFrame) -> ValidationResult:
    """Validate that years are within realistic limits."""
    violations = ((df["year"] < MIN_VALID_YEAR) | (df["year"] > MAX_VALID_YEAR)).sum()
    return ValidationResult(
        rule_name="valid_year_range",
        passed=violations == 0,
        details=f"{violations} records outside {MIN_VALID_YEAR}-{MAX_VALID_YEAR}",
        violations=violations,
    )


def check_non_negative_mileage(df: pd.DataFrame) -> ValidationResult:
    """Validate that mileage is non-negative."""
    violations = (df["km_driven"] < 0).sum()
    return ValidationResult(
        rule_name="non_negative_mileage",
        passed=violations == 0,
        details=f"{violations} records with negative mileage",
        violations=violations,
    )


def check_brand_not_empty(df: pd.DataFrame) -> ValidationResult:
    """Validate that brand is not empty or null."""
    violations = df["brand"].isnull().sum() + (df["brand"].astype(str).str.strip() == "").sum()
    return ValidationResult(
        rule_name="brand_not_empty",
        passed=violations == 0,
        details=f"{violations} records with empty brand",
        violations=violations,
    )


def check_valid_transmission(df: pd.DataFrame) -> ValidationResult:
    """Validate that transmission belongs to valid categories."""
    violations = (~df["transmission"].isin(VALID_TRANSMISSIONS)).sum()
    invalid = df.loc[~df["transmission"].isin(VALID_TRANSMISSIONS), "transmission"].unique()
    return ValidationResult(
        rule_name="valid_transmission",
        passed=violations == 0,
        details=f"{violations} invalid values: {list(invalid)[:5]}",
        violations=violations,
    )


def check_valid_fuel_type(df: pd.DataFrame) -> ValidationResult:
    """Validate that fuel type belongs to valid categories."""
    violations = (~df["fuel"].isin(VALID_FUEL_TYPES)).sum()
    invalid = df.loc[~df["fuel"].isin(VALID_FUEL_TYPES), "fuel"].unique()
    return ValidationResult(
        rule_name="valid_fuel_type",
        passed=violations == 0,
        details=f"{violations} invalid values: {list(invalid)[:5]}",
        violations=violations,
    )


def check_no_duplicate_ids(df: pd.DataFrame) -> ValidationResult:
    """Validate that listing IDs are unique."""
    if "listing_id" not in df.columns:
        return ValidationResult(
            rule_name="no_duplicate_ids",
            passed=True,
            details="No listing_id column — skipped",
            violations=0,
        )
    violations = df["listing_id"].duplicated().sum()
    return ValidationResult(
        rule_name="no_duplicate_ids",
        passed=violations == 0,
        details=f"{violations} duplicate listing IDs",
        violations=violations,
    )


def check_no_critical_nulls(df: pd.DataFrame) -> ValidationResult:
    """Validate that critical columns have no null values after cleaning."""
    critical_cols = ["selling_price", "year", "km_driven", "brand", "fuel", "transmission"]
    total_nulls = 0
    null_details = {}

    for col in critical_cols:
        if col in df.columns:
            n_null = df[col].isnull().sum()
            if n_null > 0:
                null_details[col] = n_null
                total_nulls += n_null

    return ValidationResult(
        rule_name="no_critical_nulls",
        passed=total_nulls == 0,
        details=f"{total_nulls} critical nulls: {null_details}" if total_nulls > 0 else "No critical nulls",
        violations=total_nulls,
    )


def run_validation(df: pd.DataFrame) -> dict:
    """
    Execute all validation checks and generate a report.

    Args:
        df: Cleaned DataFrame to validate.

    Returns:
        Dictionary containing the full validation report.
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA VALIDATION")
    logger.info("=" * 60)

    # Run all checks
    checks = [
        check_positive_price(df),
        check_valid_year(df),
        check_non_negative_mileage(df),
        check_brand_not_empty(df),
        check_valid_transmission(df),
        check_valid_fuel_type(df),
        check_no_duplicate_ids(df),
        check_no_critical_nulls(df),
    ]

    # Build report
    n_passed = sum(1 for c in checks if c.passed)
    n_failed = sum(1 for c in checks if not c.passed)

    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
        "summary": {
            "total_checks": len(checks),
            "passed": n_passed,
            "failed": n_failed,
            "overall_status": "PASS" if n_failed == 0 else "FAIL",
        },
        "checks": [c.to_dict() for c in checks],
    }

    # Log results
    for check in checks:
        status_icon = "[OK]" if check.passed else "[FAIL]"
        logger.info(f"  {status_icon} {check.rule_name}: {check.details}")

    logger.info("-" * 60)
    logger.info(f"VALIDATION RESULT: {n_passed}/{len(checks)} passed -- {report['summary']['overall_status']}")
    logger.info("=" * 60)

    # Save report (convert numpy types for JSON serialization)
    def _convert(obj):
        """Convert numpy types to Python native for JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(VALIDATION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=_convert)
    logger.info(f"Validation report saved to: {VALIDATION_REPORT_PATH}")

    return report
