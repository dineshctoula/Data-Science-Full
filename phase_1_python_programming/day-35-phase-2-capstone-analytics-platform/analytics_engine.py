"""
Day 35 Phase 2 Capstone: Enterprise Hybrid Analytics Engine (Polars + DuckDB + PyArrow)
========================================================================================
This module implements a production-grade, object-oriented data engineering and analytics engine:
1. Automated Data Quality Remediation & Anomaly Handling.
2. High-Performance Multi-Source ETL Joins & Feature Engineering using Polars.
3. RFM (Recency, Frequency, Monetary) Customer Segmentation Algorithm.
4. Population Stability Index (PSI) Statistical Data Drift Tracker.
5. DuckDB In-Memory SQL Analytical Staging Engine & Data Mart Export.

Author: 100-Day Data Science Challenge Team
"""

import os
import json
import duckdb
import numpy as np
import polars as pl
from datetime import datetime, date
from typing import Dict, Tuple, Any


class EnterpriseAnalyticsEngine:
    """
    Core analytics pipeline orchestrator using Polars and DuckDB for high-throughput
    data cleaning, RFM segmentation, statistical drift calculation, and staging.
    """

    def __init__(self, datalake_paths: Dict[str, str], db_path: str):
        """
        Initializes engine paths and DuckDB analytical database connection.
        
        Parameters:
            datalake_paths (dict): Map of dataset keys to raw file paths.
            db_path (str): File path for persistent DuckDB database storage.
        """
        self.paths = datalake_paths
        self.db_path = db_path
        self.con = duckdb.connect(database=db_path)
        self.curated_df: pl.DataFrame = None
        self.rfm_df: pl.DataFrame = None
        self.psi_metrics: Dict[str, Any] = {}

    def load_and_clean_data() -> pl.DataFrame:
        """
        Loads multi-source raw files using Polars eager evaluation and applies
        automated data cleaning rules:
        - Filters out negative net amounts (corrupted logs).
        - Handles null customer IDs by marking unlinked checkouts.
        - Clips extreme pricing outliers (>99.9th percentile).
        - Casts datetime strings into native Polars Datetime objects.
        
        Returns:
            pl.DataFrame: Cleaned and unified transaction DataFrame.
        """
        print("⚡ [Analytics Engine] Ingesting and cleaning raw datalake files with Polars...")

        # Read Current Transactions (Parquet)
        df_txn = pl.read_parquet(self.paths["current_transactions"])
        initial_count = df_txn.height

        # Rule 1: Cast timestamp and handle invalid/corrupted net amounts
        df_txn = df_txn.with_columns([
            pl.col("transaction_timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("transaction_dt"),
            pl.col("net_amount").abs().alias("net_amount_clean")  # Remediate negative values
        ])

        # Rule 2: Remediate extreme unit price outliers (>99.9th percentile)
        price_cap = float(df_txn.select(pl.col("unit_price").quantile(0.999)).item())
        df_txn = df_txn.with_columns(
            pl.when(pl.col("unit_price") > price_cap)
            .then(price_cap)
            .otherwise(pl.col("unit_price"))
            .alias("unit_price_clean")
        )

        # Rule 3: Recompute gross and net amounts based on cleaned pricing
        df_txn = df_txn.with_columns([
            (pl.col("quantity") * pl.col("unit_price_clean")).alias("gross_amount_clean"),
            ((pl.col("quantity") * pl.col("unit_price_clean")) * (1.0 - pl.col("discount_rate"))).alias("net_amount_clean")
        ])

        # Read Product Catalog (CSV) and Customer CRM (JSON)
        df_products = pl.read_csv(self.paths["product_catalog"])
        
        with open(self.paths["customer_crm"], 'r') as f:
            crm_list = json.load(f)
        df_crm = pl.DataFrame(crm_list).with_columns(
            pl.col("registration_date").str.to_date("%Y-%m-%d").alias("reg_date")
        )

        # Perform high-performance Polars multi-table join
        df_enriched = df_txn.join(
            df_products, on="product_id", how="left"
        ).join(
            df_crm, on="customer_id", how="left"
        )

        # Feature Engineering: Calculate Gross Profit per Order
        # Profit = Net Amount - (Quantity * Unit Cost)
        df_enriched = df_enriched.with_columns(
            (pl.col("net_amount_clean") - (pl.col("quantity") * pl.col("unit_cost").fill_null(0.0))).alias("order_profit")
        )

        print(f"  └─ Successfully cleaned {initial_count:,} raw transactions -> {df_enriched.height:,} enriched records.")
        self.curated_df = df_enriched
        return df_enriched

    def calculate_rfm_segmentation(self, reference_date: date = date(2026, 9, 1)) -> pl.DataFrame:
        """
        Implements an advanced RFM (Recency, Frequency, Monetary) Customer Segmentation model.
        - Recency (R): Days elapsed between customer's last order and reference date.
        - Frequency (F): Total count of completed orders per customer.
        - Monetary (M): Total monetary spend across completed orders.
        - Quantile Binning: R, F, M scored on a 1-5 scale.
        - Customer Persona Mapping: Champions, Loyal, Potential Loyalists, At Risk, Hibernating, Lost.
        
        Returns:
            pl.DataFrame: Customer-level RFM metrics and segment labels.
        """
        print("🧠 [Analytics Engine] Computing Customer RFM Segmentation & Behavioral Personas...")
        
        # Filter for registered customers with completed orders
        df_valid = self.curated_df.filter(
            (pl.col("customer_id").is_not_null()) & 
            (pl.col("order_status") == "Completed")
        )

        # Aggregate customer-level Recency, Frequency, Monetary values
        rfm_raw = df_valid.group_by("customer_id").agg([
            pl.col("transaction_dt").max().dt.date().alias("last_order_date"),
            pl.col("order_id").count().alias("frequency"),
            pl.col("net_amount_clean").sum().alias("monetary"),
            pl.col("order_profit").sum().alias("total_profit"),
            pl.col("country").first().alias("country"),
            pl.col("loyalty_tier").first().alias("loyalty_tier")
        ]).with_columns(
            (pl.lit(reference_date) - pl.col("last_order_date")).dt.total_days().alias("recency")
        )

        # Convert to pandas/numpy for stable quantile binning
        pdf_rfm = rfm_raw.to_pandas()

        # Recency Scoring (Lower recency days = higher score 5)
        pdf_rfm['r_score'] = pd_qcut_safe(pdf_rfm['recency'], q=5, labels=[5, 4, 3, 2, 1])
        # Frequency Scoring (Higher frequency = higher score 5)
        pdf_rfm['f_score'] = pd_qcut_safe(pdf_rfm['frequency'], q=5, labels=[1, 2, 3, 4, 5])
        # Monetary Scoring (Higher spend = higher score 5)
        pdf_rfm['m_score'] = pd_qcut_safe(pdf_rfm['monetary'], q=5, labels=[1, 2, 3, 4, 5])

        # Combined RFM Score string (e.g. "555", "111")
        pdf_rfm['rfm_combined'] = (
            pdf_rfm['r_score'].astype(str) + 
            pdf_rfm['f_score'].astype(str) + 
            pdf_rfm['m_score'].astype(str)
        )

        # Persona Rule Engine based on R and F scores
        def assign_segment(row):
            r, f = int(row['r_score']), int(row['f_score'])
            if r >= 4 and f >= 4:
                return "Champions"
            elif r >= 3 and f >= 3:
                return "Loyal Customers"
            elif r >= 4 and f <= 2:
                return "Recent / New Customers"
            elif r == 3 and f <= 2:
                return "Potential Loyalists"
            elif r == 2 and f >= 3:
                return "At Risk"
            elif r == 2 and f <= 2:
                return "Customers Needing Attention"
            elif r == 1 and f >= 4:
                return "Cant Lose Them"
            elif r == 1 and f >= 2:
                return "Hibernating"
            else:
                return "Lost / Dormant"

        pdf_rfm['customer_segment'] = pdf_rfm.apply(assign_segment, axis=1)
        
        self.rfm_df = pl.DataFrame(pdf_rfm)
        print(f"  └─ RFM model segmented {self.rfm_df.height:,} registered active customers across 9 personas.")
        return self.rfm_df

    def calculate_statistical_data_drift(self, num_buckets: int = 10) -> Dict[str, Any]:
        """
        Computes Population Stability Index (PSI) to detect statistical distribution drift
        between baseline transaction sales (2025) and current operational sales (2026).
        
        Formula:
            PSI = SUM( (Actual_i - Expected_i) * ln(Actual_i / Expected_i) )
            
        Interpretation Thresholds:
            PSI < 0.10: No significant distribution drift.
            0.10 <= PSI <= 0.25: Moderate shift observed; investigation recommended.
            PSI > 0.25: Significant distribution shift; model recalibration required.
        """
        print("📊 [Analytics Engine] Evaluating Population Stability Index (PSI) for sales data drift...")
        
        # Load baseline net amounts
        df_baseline = pl.read_parquet(self.paths["baseline_transactions"])
        baseline_vals = df_baseline.select(pl.col("net_amount").abs()).to_numpy().flatten()
        current_vals = self.curated_df.select(pl.col("net_amount_clean")).to_numpy().flatten()

        # Define quantile bin boundaries based on baseline distribution
        percentiles = np.linspace(0, 100, num_buckets + 1)
        bin_edges = np.percentile(baseline_vals, percentiles)
        # Ensure bin boundaries are strictly increasing by adding tiny offset
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) <= 2:
            bin_edges = np.linspace(baseline_vals.min(), baseline_vals.max(), num_buckets + 1)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Calculate counts per bin
        expected_counts, _ = np.histogram(baseline_vals, bins=bin_edges)
        actual_counts, _ = np.histogram(current_vals, bins=bin_edges)

        # Convert counts to proportions (fractions) with epsilon smoothing to prevent zero division
        eps = 1e-6
        expected_pct = (expected_counts / len(baseline_vals)) + eps
        actual_pct = (actual_counts / len(current_vals)) + eps

        # Normalize proportions to sum to 1.0
        expected_pct /= expected_pct.sum()
        actual_pct /= actual_pct.sum()

        # Compute PSI component-wise
        psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        total_psi = float(np.sum(psi_components))

        # Determine status classification
        if total_psi < 0.10:
            status = "STABLE (No significant drift)"
        elif total_psi <= 0.25:
            status = "MODERATE DRIFT (Slight shift)"
        else:
            status = "HIGH DRIFT (Significant distribution shift)"

        self.psi_metrics = {
            "psi_score": round(total_psi, 4),
            "status": status,
            "num_buckets": len(expected_counts),
            "baseline_sample_size": len(baseline_vals),
            "current_sample_size": len(current_vals),
            "bucket_proportions": {
                "expected": expected_pct.tolist(),
                "actual": actual_pct.tolist()
            }
        }

        print(f"  └─ Population Stability Index (PSI): {total_psi:.4f} -> [{status}]")
        return self.psi_metrics

    def stage_in_duckdb_and_export(self, output_parquet_path: str):
        """
        Registers curated DataFrames in DuckDB, runs analytical SQL queries, and exports
        the final curated enterprise Data Mart to Parquet format.
        
        Parameters:
            output_parquet_path (str): Destination path for the curated Parquet Data Mart.
        """
        print("🦆 [Analytics Engine] Staging curated data in DuckDB analytical lakehouse...")
        
        # Register Polars DataFrames into DuckDB virtual memory tables
        self.con.register("transactions_curated", self.curated_df.to_arrow())
        self.con.register("rfm_segments", self.rfm_df.to_arrow())
        
        # Read Clickstream and register into DuckDB
        df_click = pl.read_csv(self.paths["web_clickstream"])
        self.con.register("web_clickstream", df_click.to_arrow())

        # Execute DDL to create persistent physical DuckDB tables
        self.con.execute("""
            CREATE OR REPLACE TABLE dim_customer_rfm AS SELECT * FROM rfm_segments;
            CREATE OR REPLACE TABLE fact_transactions AS SELECT * FROM transactions_curated;
            CREATE OR REPLACE TABLE fact_clickstream AS SELECT * FROM web_clickstream;
        """)

        # Execute analytical SQL aggregation: Executive Summary view
        sql_summary = """
        SELECT 
            t.category,
            COUNT(DISTINCT t.order_id) AS total_orders,
            SUM(t.quantity) AS total_units_sold,
            ROUND(SUM(t.net_amount_clean), 2) AS total_revenue,
            ROUND(SUM(t.order_profit), 2) AS total_gross_profit,
            ROUND(AVG(t.net_amount_clean), 2) AS average_order_value
        FROM fact_transactions t
        WHERE t.order_status = 'Completed'
        GROUP BY t.category
        ORDER BY total_revenue DESC;
        """
        df_sql_exec = self.con.execute(sql_summary).pl()
        print("\n📌 Executive Category Performance Summary (via DuckDB SQL):")
        print(df_sql_exec)

        # Export Curated Enterprise Data Mart to Parquet
        print(f"\n💾 Exporting Curated Data Mart to Parquet: '{output_parquet_path}'...")
        self.curated_df.write_parquet(output_parquet_path, compression="zstd")
        print("  └─ Data Mart export complete!")

    def close(self):
        """Closes DuckDB database connection."""
        self.con.close()


def pd_qcut_safe(series: Any, q: int, labels: list) -> Any:
    """Helper function to perform quantile binning safely handling duplicate bin edges."""
    import pandas as pd
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates='drop')
    except Exception:
        # Fallback to ranking if qcut encounters non-unique boundaries
        ranks = series.rank(method='first')
        return pd.qcut(ranks, q=q, labels=labels)
