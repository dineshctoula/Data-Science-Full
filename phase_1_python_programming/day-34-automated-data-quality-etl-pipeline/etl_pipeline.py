"""
================================================================================
Day 34: Automated Data Quality Auditing & Modular Multi-Stage ETL Pipeline
Module 3: Multi-Stage Hybrid ETL Transformation Engine & Data Mart Exporter
================================================================================
Performs automated anomaly remediation, multi-source metadata enrichment,
feature engineering, and multi-destination loading (Parquet & DuckDB).
"""

import os
import json
import duckdb
import polars as pl
import pandas as pd
from validator import DataQualityValidator, StatisticalDriftDetector, QualityIssue

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CURATED_PARQUET = os.path.join(DATA_DIR, "curated_transactions_mart.parquet")
DUCKDB_FILE = os.path.join(DATA_DIR, "staging_data_mart.duckdb")


class ModularETLPipeline:
    """
    Automated Multi-Stage ETL Engine for enterprise data processing.
    """

    def __init__(self, baseline_path: str, incoming_path: str, customers_json_path: str):
        self.baseline_path = baseline_path
        self.incoming_path = incoming_path
        self.customers_json_path = customers_json_path
        self.audit_issues = []

    def run_pipeline(self):
        """
        Executes full ETL lifecycle: Extract -> Audit -> Remediate -> Enrich -> Load.
        """
        print("\n" + "=" * 75)
        print("⚙️ EXECUTING AUTOMATED ETL & REMEDIATION PIPELINE")
        print("=" * 75)

        # --- STAGE 1: EXTRACT ---
        print("\n📥 Stage 1: Extracting raw data streams...")
        df_base = pl.read_parquet(self.baseline_path)
        df_raw = pl.read_parquet(self.incoming_path)

        with open(self.customers_json_path, "r") as f:
            customers_data = json.load(f)
        df_customers = pl.DataFrame(customers_data)
        print(f"  • Baseline records: {df_base.height:,}")
        print(f"  • Incoming raw batch: {df_raw.height:,}")
        print(f"  • Customer profiles: {df_customers.height:,}")

        # --- STAGE 2: AUDIT & DRIFT DETECTION ---
        print("\n🔎 Stage 2: Auditing schema, quality constraints & distribution drift...")
        expected_schema = {
            "transaction_id": str, "user_id": int, "timestamp": str,
            "category": str, "country": str, "amount": float,
            "risk_score": float, "is_completed": int
        }
        null_limits = {"user_id": 0.01, "amount": 0.005, "category": 0.01}
        valid_categories = ["Electronics", "Groceries", "Clothing", "Home", "Beauty", "Sports"]

        validator = DataQualityValidator(expected_schema, null_limits)
        schema_issues = validator.audit_schema_and_nulls(df_raw)
        value_issues = validator.audit_value_constraints(df_raw, valid_categories)
        drift_issues = StatisticalDriftDetector.detect_numeric_drift(
            df_base, df_raw, numeric_cols=["amount", "risk_score"]
        )

        self.audit_issues = schema_issues + value_issues + drift_issues

        print(f"  • Detected {len(self.audit_issues)} total data quality & drift alerts:")
        for issue in self.audit_issues:
            icon = "🔴" if issue.severity == "CRITICAL" else "🟡" if issue.severity == "WARNING" else "🔵"
            print(f"    {icon} [{issue.severity}] [{issue.check_type}] Col: {issue.column} - {issue.message}")

        # --- STAGE 3: TRANSFORM & AUTOMATED ANOMALY REMEDIATION ---
        print("\n🛠️ Stage 3: Performing automated anomaly remediation & feature engineering...")

        # 1. Clean dirty string categories (trim whitespace, title case)
        df_clean = df_raw.with_columns(
            pl.col("category").str.strip_chars().str.to_titlecase().alias("category_clean")
        )

        # Map unrecognized categories to 'Other'
        df_clean = df_clean.with_columns(
            pl.when(pl.col("category_clean").is_in(valid_categories))
            .then(pl.col("category_clean"))
            .otherwise(pl.lit("Other"))
            .alias("category_standardized")
        ).drop(["category", "category_clean"]).rename({"category_standardized": "category"})

        # 2. Remediate invalid negative amounts (replace negative amounts with null)
        df_clean = df_clean.with_columns(
            pl.when(pl.col("amount") < 0.0)
            .then(None)
            .otherwise(pl.col("amount"))
            .alias("amount_valid")
        )

        # 3. Impute missing amounts using category median
        category_medians = df_clean.group_by("category").agg(
            pl.col("amount_valid").median().alias("cat_median_amount")
        )
        df_clean = df_clean.join(category_medians, on="category", how="left")

        df_clean = df_clean.with_columns(
            pl.col("amount_valid").fill_null(pl.col("cat_median_amount")).alias("amount_imputed")
        ).drop(["amount", "amount_valid", "cat_median_amount"]).rename({"amount_imputed": "amount"})

        # 4. Impute missing user_id with fallback ID (-1)
        df_clean = df_clean.with_columns(
            pl.col("user_id").fill_null(-1).cast(pl.Int64)
        )

        # 5. Multi-Source Enrichment: Join Customer Profile Metadata
        df_enriched = df_clean.join(
            df_customers.select(["user_id", "email", "membership_tier"]),
            on="user_id",
            how="left"
        ).with_columns(
            pl.col("membership_tier").fill_null("Guest"),
            pl.col("email").fill_null("unregistered@unknown.com")
        )

        # 6. Feature Engineering & Risk Tier Mapping
        df_final = df_enriched.with_columns(
            (pl.col("amount") * 0.025).round(2).alias("processing_fee"),
            pl.when(pl.col("risk_score") >= 0.75).then(pl.lit("CRITICAL"))
            .when(pl.col("risk_score") >= 0.50).then(pl.lit("HIGH"))
            .when(pl.col("risk_score") >= 0.20).then(pl.lit("MEDIUM"))
            .otherwise(pl.lit("LOW"))
            .alias("risk_tier"),
            # Category average ratio via window expression (.over())
            (pl.col("amount") / pl.col("amount").mean().over("category")).round(2).alias("amount_to_cat_avg_ratio")
        )

        print(f"  • Successfully remediated nulls & negative amounts.")
        print(f"  • Enriched with Customer Profiles & engineered features: 'processing_fee', 'risk_tier', 'amount_to_cat_avg_ratio'.")

        # --- STAGE 4: LOAD TO DATA MART & DUCKDB STAGING ---
        print("\n💾 Stage 4: Loading curated dataset to Data Mart (Parquet & DuckDB)...")

        # Export to Parquet
        df_final.write_parquet(CURATED_PARQUET, compression="snappy")
        file_size_mb = os.path.getsize(CURATED_PARQUET) / (1024 * 1024)
        print(f"  • Curated Parquet Data Mart created: {CURATED_PARQUET} ({file_size_mb:.2f} MB)")

        # Export to DuckDB Staging DB
        if os.path.exists(DUCKDB_FILE):
            os.remove(DUCKDB_FILE)

        conn = duckdb.connect(DUCKDB_FILE)
        # Register Polars DataFrame in DuckDB via PyArrow zero-copy
        arrow_table = df_final.to_arrow()
        conn.register("curated_view", arrow_table)
        conn.execute("CREATE TABLE curated_transactions AS SELECT * FROM curated_view")

        # Create indexed view for analytics
        conn.execute("CREATE INDEX idx_user_cat ON curated_transactions(user_id, category)")
        conn.close()

        print(f"  • DuckDB Staging Database loaded with indexed analytical table: {DUCKDB_FILE}")
        return df_final, self.audit_issues
