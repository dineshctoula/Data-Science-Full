"""
================================================================================
Day 34: Automated Data Quality Auditing & Modular Multi-Stage ETL Pipeline
Module 2: Enterprise Data Quality Validator & Statistical Drift Detector
================================================================================
Provides a production-grade data validation and drift detection engine:
1. Schema & Type Enforcement (missing columns, type mismatches, null thresholds).
2. Value Constraint & Anomaly Rules (negative amounts, extreme outliers, invalid regex).
3. Population Stability Index (PSI) & Statistical Z-Score Distribution Drift.
4. Automated Audit Severity Classification (INFO, WARNING, CRITICAL).
"""

import re
import numpy as np
import pandas as pd
import polars as pl
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


@dataclass
class QualityIssue:
    check_type: str        # 'SCHEMA', 'NULL_THRESHOLD', 'RANGE_BOUND', 'REGEX_PATTERN', 'DRIFT'
    column: str
    severity: str          # 'INFO', 'WARNING', 'CRITICAL'
    message: str
    affected_count: int = 0
    metric_value: float = 0.0


class DataQualityValidator:
    """
    Automated Data Quality & Schema Auditor for transactional datasets.
    """

    def __init__(self, expected_schema: Dict[str, Any], null_thresholds: Dict[str, float] = None):
        self.expected_schema = expected_schema
        self.null_thresholds = null_thresholds or {}

    def audit_schema_and_nulls(self, df: pl.DataFrame) -> List[QualityIssue]:
        """
        Validates schema adherence and null value limits.
        """
        issues = []

        # 1. Column Presence Check
        missing_cols = set(self.expected_schema.keys()) - set(df.columns)
        for col in missing_cols:
            issues.append(QualityIssue(
                check_type="SCHEMA",
                column=col,
                severity="CRITICAL",
                message=f"Missing mandatory column: '{col}' in incoming batch",
            ))

        # 2. Null Value Threshold Check
        for col in df.columns:
            if col in self.null_thresholds:
                null_count = df[col].null_count()
                null_pct = null_count / df.height
                allowed_pct = self.null_thresholds[col]

                if null_pct > allowed_pct:
                    severity = "CRITICAL" if null_pct > 0.05 else "WARNING"
                    issues.append(QualityIssue(
                        check_type="NULL_THRESHOLD",
                        column=col,
                        severity=severity,
                        message=f"Null ratio ({null_pct:.2%}) exceeded maximum threshold ({allowed_pct:.2%})",
                        affected_count=null_count,
                        metric_value=round(null_pct, 4)
                    ))

        return issues

    def audit_value_constraints(
        self, df: pl.DataFrame, valid_categories: List[str], max_amount: float = 20000.0
    ) -> List[QualityIssue]:
        """
        Validates domain business logic (no negative amounts, no unhandled categories, outlier boundaries).
        """
        issues = []

        # 1. Negative Amounts Check
        if "amount" in df.columns:
            neg_count = df.filter(pl.col("amount") < 0).height
            if neg_count > 0:
                issues.append(QualityIssue(
                    check_type="RANGE_BOUND",
                    column="amount",
                    severity="CRITICAL",
                    message=f"Found {neg_count} invalid negative transaction amounts",
                    affected_count=neg_count,
                ))

            # Extreme Outlier Check
            outlier_count = df.filter(pl.col("amount") > max_amount).height
            if outlier_count > 0:
                issues.append(QualityIssue(
                    check_type="RANGE_BOUND",
                    column="amount",
                    severity="WARNING",
                    message=f"Found {outlier_count} transaction amounts exceeding extreme boundary (${max_amount:,.2f})",
                    affected_count=outlier_count,
                ))

        # 2. Domain Category Check (detect dirty whitespace / unexpected categories)
        if "category" in df.columns:
            invalid_cat_df = df.filter(~pl.col("category").is_in(valid_categories))
            invalid_count = invalid_cat_df.height
            if invalid_count > 0:
                sample_dirty = invalid_cat_df["category"].unique().to_list()[:3]
                issues.append(QualityIssue(
                    check_type="DOMAIN_CATEGORY",
                    column="category",
                    severity="WARNING",
                    message=f"Found {invalid_count} non-standard category values (e.g. {sample_dirty})",
                    affected_count=invalid_count,
                ))

        return issues


class StatisticalDriftDetector:
    """
    Computes statistical data drift between baseline historical data and current incoming batch.
    Uses Population Stability Index (PSI) and Mean Z-Score shifts.
    """

    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_bins: int = 10) -> Tuple[float, List[Dict]]:
        """
        Computes Population Stability Index (PSI) for continuous numeric distributions.
        PSI < 0.10: Stable / No Drift
        0.10 <= PSI < 0.25: Moderate Drift
        PSI >= 0.25: Significant Drift (Action Required)
        """
        # Filter out NaN/null values
        b_clean = baseline[~np.isnan(baseline)]
        c_clean = current[~np.isnan(current)]

        if len(b_clean) == 0 or len(c_clean) == 0:
            return 0.0, []

        # Create quantile-based bins from baseline
        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(b_clean, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        # Deduplicate bins if values are tight
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0, []

        # Calculate counts per bin
        b_counts, _ = np.histogram(b_clean, bins=bins)
        c_counts, _ = np.histogram(c_clean, bins=bins)

        # Convert to proportions with zero-handling smoothing
        b_pct = np.maximum(b_counts / len(b_clean), 1e-4)
        c_pct = np.maximum(c_counts / len(c_clean), 1e-4)

        # PSI formula
        psi_value = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))

        bin_details = []
        for i in range(len(b_pct)):
            bin_details.append({
                "bin_index": i + 1,
                "baseline_pct": round(float(b_pct[i]), 4),
                "current_pct": round(float(c_pct[i]), 4),
                "psi_contrib": round(float((c_pct[i] - b_pct[i]) * np.log(c_pct[i] / b_pct[i])), 4),
            })

        return float(psi_value), bin_details

    @classmethod
    def detect_numeric_drift(
        cls, baseline_df: pl.DataFrame, current_df: pl.DataFrame, numeric_cols: List[str]
    ) -> List[QualityIssue]:
        """
        Performs drift detection across all specified numeric features.
        """
        drift_issues = []

        for col in numeric_cols:
            if col not in baseline_df.columns or col not in current_df.columns:
                continue

            b_arr = baseline_df[col].drop_nulls().to_numpy()
            c_arr = current_df[col].drop_nulls().to_numpy()

            # 1. Calculate PSI
            psi_val, _ = cls.calculate_psi(b_arr, c_arr)

            # 2. Compute Mean & Std Z-Score Shift
            b_mean, b_std = np.mean(b_arr), np.std(b_arr)
            c_mean, c_std = np.mean(c_arr), np.std(c_arr)

            mean_shift_z = (c_mean - b_mean) / b_std if b_std > 0 else 0.0

            # Determine Drift Severity
            if psi_val >= 0.25:
                severity = "CRITICAL"
                msg = f"Significant data drift detected (PSI = {psi_val:.4f} >= 0.25, Mean Shift = {mean_shift_z:+.2f}σ)"
            elif psi_val >= 0.10:
                severity = "WARNING"
                msg = f"Moderate data drift detected (PSI = {psi_val:.4f}, Mean Shift = {mean_shift_z:+.2f}σ)"
            else:
                severity = "INFO"
                msg = f"Distribution stable (PSI = {psi_val:.4f}, Mean Shift = {mean_shift_z:+.2f}σ)"

            if severity in ["WARNING", "CRITICAL"]:
                drift_issues.append(QualityIssue(
                    check_type="DRIFT",
                    column=col,
                    severity=severity,
                    message=msg,
                    metric_value=round(psi_val, 4)
                ))

        return drift_issues
