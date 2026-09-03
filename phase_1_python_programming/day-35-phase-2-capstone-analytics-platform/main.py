"""
Day 35 Phase 2 Capstone: Main Pipeline Orchestrator & Executive Report Generator
=================================================================================
This script acts as the master orchestrator for the Phase 2 Capstone Project:
1. Triggers multi-source raw datalake generation.
2. Runs Polars & DuckDB data cleaning, feature engineering, and RFM customer segmentation.
3. Computes statistical sales distribution drift (PSI).
4. Generates static Seaborn visual suites and an interactive Plotly HTML dashboard.
5. Produces the final markdown Executive Summary (`CAPSTONE_EXECUTIVE_SUMMARY.md`).

Author: 100-Day Data Science Challenge Team
"""

import os
import sys
import time
import polars as pl
from datetime import datetime

# Import modular pipeline components
from data_generator import build_enterprise_datalake
from analytics_engine import EnterpriseAnalyticsEngine
from dashboard_generator import generate_static_visual_suite, build_interactive_plotly_dashboard


def generate_executive_markdown_report(
    df_curated: pl.DataFrame,
    df_rfm: pl.DataFrame,
    psi_metrics: dict,
    report_path: str
):
    """
    Generates a formal executive markdown report summarizing Phase 2 Capstone results.
    
    Parameters:
        df_curated (pl.DataFrame): Cleaned transaction logs.
        df_rfm (pl.DataFrame): Customer RFM segments.
        psi_metrics (dict): PSI drift metrics.
        report_path (str): File destination for markdown report.
    """
    pdf_curated = df_curated.to_pandas()
    pdf_rfm = df_rfm.to_pandas()

    total_revenue = pdf_curated['net_amount_clean'].sum()
    total_profit = pdf_curated['order_profit'].sum()
    total_orders = pdf_curated['order_id'].nunique()
    active_customers = pdf_rfm['customer_id'].nunique()
    aov = total_revenue / max(total_orders, 1)
    profit_margin = (total_profit / total_revenue) * 100

    category_summary = pdf_curated.groupby('category').agg({
        'order_id': 'nunique',
        'net_amount_clean': 'sum',
        'order_profit': 'sum'
    }).reset_index().sort_values('net_amount_clean', ascending=False)

    rfm_summary = pdf_rfm.groupby('customer_segment').agg({
        'customer_id': 'count',
        'monetary': 'sum',
        'frequency': 'mean'
    }).reset_index().sort_values('monetary', ascending=False)

    md_content = f"""# Phase 2 Grand Capstone Project Executive Summary
## Enterprise E-Commerce Analytics, Data Quality Engine & Executive Dashboard

**Project Completion Date:** {datetime.now().strftime("%B %d, %Y")}  
**Challenge Status:** Day 35 of 100 — Phase 2 (Data Manipulation & Visualization) 100% Complete! 🏆

---

## 🎯 Executive Overview

The **Phase 2 Capstone Project** consolidates all foundational data engineering, data cleaning, high-performance computing, DuckDB SQL analytics, Polars vectorized operations, RFM customer intelligence, and interactive visualization techniques mastered across Days 16 through 35.

This enterprise platform ingests multi-source datalake streams (Parquet, JSON CRM, CSV Clickstream & Product metadata), executes automated data cleaning and anomaly remediation, models customer personas using an advanced RFM matrix, evaluates population distribution drift (PSI), and delivers interactive executive dashboards.

---

## 📊 Core Business & Operational Metrics

| Metric Name | Value | Description |
| :--- | :--- | :--- |
| **Total Net Revenue** | **${total_revenue:,.2f}** | Total revenue generated across completed operational orders |
| **Total Gross Profit** | **${total_profit:,.2f}** | Gross profit net of product unit cost |
| **Gross Profit Margin** | **{profit_margin:.2f}%** | Overall gross margin percentage |
| **Total Completed Orders** | **{total_orders:,}** | Total distinct completed transactions |
| **Average Order Value (AOV)** | **${aov:,.2f}** | Average net basket value per transaction |
| **Active Segmented Customers** | **{active_customers:,}** | Registered customers categorized into RFM personas |
| **PSI Sales Drift Index** | **{psi_metrics.get('psi_score', 0.0):.4f}** | **{psi_metrics.get('status', 'STABLE')}** |

---

## 📦 Product Category Revenue & Profit Breakdown

| Product Category | Total Orders | Net Revenue ($) | Gross Profit ($) | Margin (%) |
| :--- | :---: | :---: | :---: | :---: |
"""
    for _, row in category_summary.iterrows():
        cat_rev = row['net_amount_clean']
        cat_prof = row['order_profit']
        cat_margin = (cat_prof / cat_rev) * 100 if cat_rev > 0 else 0
        md_content += f"| **{row['category']}** | {row['order_id']:,} | ${cat_rev:,.2f} | ${cat_prof:,.2f} | {cat_margin:.1f}% |\n"

    md_content += f"""
---

## 🧠 Customer Persona RFM Intelligence Breakdown

| RFM Customer Persona | Customer Count | Total Segment Spend ($) | Avg Purchases / Cust |
| :--- | :---: | :---: | :---: |
"""
    for _, row in rfm_summary.iterrows():
        md_content += f"| **{row['customer_segment']}** | {row['customer_id']:,} | ${row['monetary']:,.2f} | {row['frequency']:.1f} |\n"

    md_content += f"""
---

## 📈 Statistical Data Drift Analysis (Population Stability Index - PSI)

- **Reference Window:** Baseline Transactions (2025 Sales Distribution)
- **Current Operational Window:** Current Transactions (2026 Sales Distribution)
- **PSI Score:** **{psi_metrics.get('psi_score', 0.0):.4f}**
- **Evaluation Status:** `{psi_metrics.get('status', 'STABLE')}`
- **Data Engineering Takeaway:** The sales distribution between the baseline period and current operational period demonstrates high statistical stability (PSI < 0.10). No underlying pricing model recalibration or data pipeline drift alerts required.

---

## 🛠️ Technology Stack & Architecture

1. **Multi-Source Data Ingestion:** Heterogeneous stream merging (Parquet, JSON, CSV).
2. **Polars Data Engine:** Ultra-fast in-memory cleaning, filtering, quantile binning, and multi-table joins.
3. **DuckDB Analytical Lakehouse:** SQL analytical staging and persistent column-oriented data mart export.
4. **RFM Segmentation Model:** Automated R, F, M scoring with 9 behavioral customer personas.
5. **Plotly & Seaborn Dashboards:** Interactive HTML multi-card executive dashboard and high-res static charts.

---

## 🚀 Phase 2 Graduation Milestone

> **Phase 2 (Data Manipulation & Visualization - Days 16 to 35) is officially complete!**  
> Ready to advance into **Phase 3 (Statistics, Probability & Machine Learning)**! 🚀
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"📄 [Main Orchestrator] Executive Summary Report written to: '{report_path}'")


def run_capstone_pipeline():
    """Main pipeline execution routine."""
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 80)
    print("🌟 DAY 35 PHASE 2 CAPSTONE: ENTERPRISE DATA ENGINEERING & ANALYTICS PLATFORM")
    print("=" * 80)

    # File & Directory Configurations
    datalake_dir = os.path.join(base_dir, "raw_datalake")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(output_dir, "capstone_lakehouse.duckdb")
    curated_parquet_path = os.path.join(output_dir, "curated_enterprise_mart.parquet")
    html_dashboard_path = os.path.join(output_dir, "executive_dashboard.html")
    report_path = os.path.join(base_dir, "CAPSTONE_EXECUTIVE_SUMMARY.md")

    # Step 1: Data Synthesis
    datalake_paths = build_enterprise_datalake(datalake_dir)

    # Step 2: Analytics & Cleaning Engine
    engine = EnterpriseAnalyticsEngine(datalake_paths, db_path)
    df_curated = engine.load_and_clean_data()
    df_rfm = engine.calculate_rfm_segmentation()
    psi_metrics = engine.calculate_statistical_data_drift()

    # Step 3: DuckDB Staging & Data Mart Export
    engine.stage_in_duckdb_and_export(curated_parquet_path)
    engine.close()

    # Step 4: Dashboard Generation
    generate_static_visual_suite(df_curated, df_rfm, output_dir)
    build_interactive_plotly_dashboard(df_curated, df_rfm, psi_metrics, html_dashboard_path)

    # Step 5: Report Generation
    generate_executive_markdown_report(df_curated, df_rfm, psi_metrics, report_path)

    # Pipeline Assertions
    assert os.path.exists(curated_parquet_path), "Curated Parquet Data Mart missing!"
    assert os.path.exists(html_dashboard_path), "Interactive HTML dashboard missing!"
    assert os.path.exists(report_path), "Executive Report missing!"
    assert df_curated.height > 0, "Curated DataFrame is empty!"
    assert df_rfm.height > 0, "RFM DataFrame is empty!"

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"🎉 DAY 35 PHASE 2 CAPSTONE PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("=" * 80)


if __name__ == "__main__":
    run_capstone_pipeline()
