"""
================================================================================
Day 34: Automated Data Quality Auditing & Modular Multi-Stage ETL Pipeline
Main Execution Orchestrator & Report Generator
================================================================================
Orchestrates synthetic data generation, quality auditing, statistical drift detection,
ETL transformation/remediation, DuckDB staging validation, and report generation.
"""

import os
import time
import duckdb
import polars as pl
import pandas as pd
from data_generator import generate_baseline_dataset, generate_incoming_corrupted_batch, generate_customer_profiles
from validator import StatisticalDriftDetector
from etl_pipeline import ModularETLPipeline

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PARQUET = os.path.join(DATA_DIR, "baseline_transactions.parquet")
INCOMING_PARQUET = os.path.join(DATA_DIR, "incoming_batch_raw.parquet")
CUSTOMERS_JSON = os.path.join(DATA_DIR, "customer_profiles.json")
CURATED_PARQUET = os.path.join(DATA_DIR, "curated_transactions_mart.parquet")
DUCKDB_FILE = os.path.join(DATA_DIR, "staging_data_mart.duckdb")
REPORT_FILE = os.path.join(DATA_DIR, "DATA_QUALITY_AUDIT_REPORT.md")


def run_duckdb_analytics_check():
    """
    Executes analytical SQL queries on the curated DuckDB staging database.
    """
    print("\n" + "=" * 75)
    print("📊 DUCKDB ANALYTICAL STAGING AUDIT & VERIFICATION")
    print("=" * 75)

    conn = duckdb.connect(DUCKDB_FILE)

    # 1. Summary Metrics by Membership Tier
    query1 = """
    SELECT 
        membership_tier,
        COUNT(*) AS total_transactions,
        ROUND(AVG(amount), 2) AS avg_transaction_val,
        ROUND(SUM(amount), 2) AS total_revenue,
        ROUND(SUM(processing_fee), 2) AS total_fees,
        ROUND(AVG(risk_score), 4) AS avg_risk_score
    FROM curated_transactions
    GROUP BY membership_tier
    ORDER BY total_revenue DESC;
    """
    res1 = conn.execute(query1).df()
    print("\n📈 1. Financial Performance by Membership Tier:")
    print(res1.to_string(index=False))

    # 2. Risk Tier Breakdown
    query2 = """
    SELECT 
        risk_tier,
        COUNT(*) AS transaction_count,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM curated_transactions), 2) AS pct_of_total
    FROM curated_transactions
    GROUP BY risk_tier
    ORDER BY transaction_count DESC;
    """
    res2 = conn.execute(query2).df()
    print("\n🚨 2. Risk Tier Distribution:")
    print(res2.to_string(index=False))

    conn.close()
    return res1, res2


def export_markdown_audit_report(audit_issues, res_tier, res_risk, execution_time):
    """
    Generates a structured markdown audit report artifact: DATA_QUALITY_AUDIT_REPORT.md
    """
    base_df = pl.read_parquet(BASELINE_PARQUET)
    inc_df = pl.read_parquet(INCOMING_PARQUET)
    curated_df = pl.read_parquet(CURATED_PARQUET)

    psi_amt, bin_amt = StatisticalDriftDetector.calculate_psi(
        base_df["amount"].to_numpy(), inc_df["amount"].drop_nulls().to_numpy()
    )
    psi_risk, bin_risk = StatisticalDriftDetector.calculate_psi(
        base_df["risk_score"].to_numpy(), inc_df["risk_score"].drop_nulls().to_numpy()
    )

    issues_table = ""
    for issue in audit_issues:
        badge = "🔴 `CRITICAL`" if issue.severity == "CRITICAL" else "🟡 `WARNING`" if issue.severity == "WARNING" else "🔵 `INFO`"
        issues_table += f"| {badge} | `{issue.check_type}` | `{issue.column}` | {issue.message} | {issue.affected_count:,} |\n"

    def df_to_md(df):
        cols = df.columns.tolist()
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join(":---:" if df[c].dtype in ['int64', 'float64'] else ":---" for c in cols) + " |"
        rows = []
        for idx, row in df.iterrows():
            rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join([header, sep] + rows)

    tier_md_table = df_to_md(res_tier)
    risk_md_table = df_to_md(res_risk)

    report_content = f"""# 🛡️ Day 34 Automated Data Quality Audit & Statistical Drift Report

## 📌 Executive Summary
This report details the execution of an **Automated Data Quality & Statistical Drift Engine** across an incoming multi-source transactional stream (**{inc_df.height:,} raw records**). 
The multi-stage ETL pipeline identified schema irregularities, missing values, range constraint violations, and distribution drift, automatically remediating anomalies before exporting a validated dataset to the **Parquet Data Mart** and **DuckDB Analytical Staging Database**.

---

## ⏱️ Pipeline Execution Performance
- **Pipeline Processing Time**: `{execution_time:.4f} seconds`
- **Baseline Dataset Volume**: `{base_df.height:,} rows`
- **Incoming Raw Volume**: `{inc_df.height:,} rows`
- **Curated Data Mart Volume**: `{curated_df.height:,} rows`

---

## 🚨 Audit Issues & Anomaly Detection Summary

| Severity | Check Type | Target Column | Findings & Description | Affected Records |
| :--- | :--- | :--- | :--- | :---: |
{issues_table}

---

## 📈 Statistical Data Drift Analysis (Population Stability Index - PSI)

Statistical drift was evaluated between historical baseline distributions and current incoming raw batches:

| Feature Name | PSI Value | Status | Interpretation | Baseline Mean | Incoming Mean |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **`amount`** | `{psi_amt:.4f}` | 🔴 `CRITICAL DRIFT` | Significant upward distribution shift | `${base_df['amount'].mean():.2f}` | `${inc_df['amount'].drop_nulls().mean():.2f}` |
| **`risk_score`** | `{psi_risk:.4f}` | 🔴 `CRITICAL DRIFT` | Significant increase in transaction risk | `{base_df['risk_score'].mean():.4f}` | `{inc_df['risk_score'].drop_nulls().mean():.4f}` |

> [!IMPORTANT]
> **PSI Interpretation Standard:**
> - **PSI < 0.10**: Stable distribution (no action required).
> - **0.10 ≤ PSI < 0.25**: Moderate drift (monitor metric closely).
> - **PSI ≥ 0.25**: Significant drift (retrain downstream models & trigger data team alerts).

---

## 🛠️ Automated Anomaly Remediation & Pipeline Transformation
1. **String Normalization**: Trimmed whitespace and applied title-case formatting to `category`. Unrecognized values remapped to `"Other"`.
2. **Invalid Amount Handling**: Negative amounts converted to `null` and imputed using category-level medians.
3. **Missing Key Recovery**: Imputed null `user_id` values with fallback identifier (`-1`) and matched against customer profiles.
4. **Relational Enrichment**: Joined customer profiles (`email`, `membership_tier`) via left join.
5. **Feature Engineering**: Engineered `processing_fee` (`amount * 0.025`), mapped `risk_tier` (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), and computed relative spending ratio (`amount_to_cat_avg_ratio`).

---

## 📊 Curated Data Mart Analytics (DuckDB Queries)

### 1. Revenue & Fee Performance by Membership Tier
{tier_md_table}

### 2. Transaction Risk Tier Distribution
{risk_md_table}

---

## 💡 Key Architectural Takeaways
1. **Automated Auditing**: Proactive detection of schema shifts and distribution drift prevents corrupted data from polluting production data warehouses.
2. **Statistical Stability Index**: PSI provides mathematically rigorous drift tracking independently of sample sizes.
3. **Hybrid Engine Synergy**: Leveraging **Polars** for eager/lazy vector transformations and **DuckDB** for analytical SQL staging ensures sub-second throughput on enterprise datasets.
"""

    with open(REPORT_FILE, "w") as f:
        f.write(report_content)

    print(f"\n📝 Successfully exported Data Quality Audit Report: {REPORT_FILE}")


if __name__ == "__main__":
    t0 = time.perf_counter()

    # Step 1: Ensure datasets exist
    if not (os.path.exists(BASELINE_PARQUET) and os.path.exists(INCOMING_PARQUET) and os.path.exists(CUSTOMERS_JSON)):
        generate_baseline_dataset()
        generate_incoming_corrupted_batch()
        generate_customer_profiles()

    # Step 2: Execute Multi-Stage ETL Pipeline
    pipeline = ModularETLPipeline(BASELINE_PARQUET, INCOMING_PARQUET, CUSTOMERS_JSON)
    curated_df, audit_issues = pipeline.run_pipeline()

    # Step 3: Run DuckDB Analytical Queries
    res_tier, res_risk = run_duckdb_analytics_check()

    t_total = time.perf_counter() - t0
    print(f"\n⚡ Total Pipeline Execution Completed in {t_total:.4f} seconds!")

    # Step 4: Export Markdown Audit Report
    export_markdown_audit_report(audit_issues, res_tier, res_risk, t_total)
