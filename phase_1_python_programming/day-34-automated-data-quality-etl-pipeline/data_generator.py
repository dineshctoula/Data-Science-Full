"""
================================================================================
Day 34: Automated Data Quality Auditing & Modular Multi-Stage ETL Pipeline
Module 1: Synthetic Data Stream Generator (Baseline vs Incoming Drifted Data)
================================================================================
Generates multi-source synthetic datasets simulating enterprise transactional logs:
1. Baseline Reference Dataset (clean historical benchmark).
2. Incoming Raw Batch Dataset (with injected schema anomalies, missing values,
   corrupted data types, out-of-bound outliers, and statistical distribution drift).
3. Dimension Table (Customer profiles for relational enrichment).
"""

import os
import json
import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PARQUET = os.path.join(DATA_DIR, "baseline_transactions.parquet")
INCOMING_PARQUET = os.path.join(DATA_DIR, "incoming_batch_raw.parquet")
CUSTOMERS_JSON = os.path.join(DATA_DIR, "customer_profiles.json")


def generate_baseline_dataset(num_rows=100_000, seed=42):
    """
    Generates clean baseline historical transaction data representing expected distribution.
    """
    np.random.seed(seed)
    print(f"📦 Generating baseline historical dataset ({num_rows:,} rows)...")

    categories = ["Electronics", "Groceries", "Clothing", "Home", "Beauty", "Sports"]
    countries = ["US", "CA", "UK", "DE", "JP", "AU"]

    timestamps = pd.date_range("2025-01-01", periods=num_rows, freq="3s")

    data = {
        "transaction_id": [f"TXN-{i:07d}" for i in range(1, num_rows + 1)],
        "user_id": np.random.randint(10000, 20000, size=num_rows),
        "timestamp": timestamps,
        "category": np.random.choice(categories, size=num_rows, p=[0.25, 0.30, 0.20, 0.10, 0.10, 0.05]),
        "country": np.random.choice(countries, size=num_rows),
        "amount": np.round(np.random.gamma(shape=2.0, scale=40.0, size=num_rows) + 10.0, 2), # Mean ~90
        "risk_score": np.round(np.random.beta(a=2, b=10, size=num_rows), 4), # Mean ~0.16
        "is_completed": np.random.choice([1, 0], size=num_rows, p=[0.95, 0.05]),
    }

    df_base = pl.DataFrame(data)
    df_base.write_parquet(BASELINE_PARQUET, compression="snappy")
    print(f"✅ Baseline dataset saved: {BASELINE_PARQUET} ({df_base.height:,} rows)")
    return df_base


def generate_incoming_corrupted_batch(num_rows=100_000, seed=2026):
    """
    Generates raw incoming batch dataset with injected data quality issues:
    - Missing/Null values in user_id & amount
    - Out-of-bounds negative amounts & extreme outliers
    - Formatting issues (untrimmed strings, uppercase emails)
    - Statistical distribution drift in 'amount' and 'risk_score'
    """
    np.random.seed(seed)
    print(f"⚠️ Generating incoming raw batch dataset with injected anomalies ({num_rows:,} rows)...")

    categories = ["Electronics", "Groceries", "Clothing", "Home", "Beauty", "Sports", "UNKNOWN"]
    countries = ["US", "CA", "UK", "DE", "JP", "AU", "XX"]

    timestamps = pd.date_range("2026-09-01", periods=num_rows, freq="3s")

    # Injected Distribution Drift: Mean amount shifted up (shape=3.5, scale=60 -> Mean ~220 vs 90)
    amounts = np.random.gamma(shape=3.5, scale=60.0, size=num_rows) + 15.0
    
    # Injected Risk Score Drift: Higher mean risk score
    risk_scores = np.random.beta(a=3, b=7, size=num_rows) # Mean ~0.30 vs 0.16

    data = {
        "transaction_id": [f"TXN-INC-{i:07d}" for i in range(1, num_rows + 1)],
        "user_id": np.random.randint(10000, 20000, size=num_rows).astype(object),
        "timestamp": timestamps,
        "category": np.random.choice(categories, size=num_rows),
        "country": np.random.choice(countries, size=num_rows),
        "amount": amounts,
        "risk_score": risk_scores,
        "is_completed": np.random.choice([1, 0], size=num_rows, p=[0.92, 0.08]),
    }

    df_inc = pd.DataFrame(data)

    # 1. Inject Nulls (Missing Data Anomalies)
    null_idx_user = np.random.choice(num_rows, size=int(num_rows * 0.03), replace=False)
    df_inc.loc[null_idx_user, "user_id"] = None

    null_idx_amount = np.random.choice(num_rows, size=int(num_rows * 0.02), replace=False)
    df_inc.loc[null_idx_amount, "amount"] = None

    # 2. Inject Invalid Negative Amounts & Extreme Outliers
    neg_idx = np.random.choice(num_rows, size=150, replace=False)
    df_inc.loc[neg_idx, "amount"] = np.random.uniform(-500.0, -10.0, size=150)

    outlier_idx = np.random.choice(num_rows, size=50, replace=False)
    df_inc.loc[outlier_idx, "amount"] = np.random.uniform(25000.0, 50000.0, size=50)

    # 3. Inject String Dirty Formatting (leading/trailing whitespace)
    dirty_cats = ["  Electronics ", "groceries ", "CLOTHING", "  Home"]
    dirty_cat_idx = np.random.choice(num_rows, size=int(num_rows * 0.05), replace=False)
    df_inc.loc[dirty_cat_idx, "category"] = np.random.choice(dirty_cats, size=len(dirty_cat_idx))

    # Convert to Polars and export
    df_pl_inc = pl.DataFrame(df_inc)
    df_pl_inc.write_parquet(INCOMING_PARQUET, compression="snappy")
    print(f"✅ Incoming raw batch saved: {INCOMING_PARQUET} ({df_pl_inc.height:,} rows)")
    return df_pl_inc


def generate_customer_profiles(num_customers=10000, seed=123):
    """
    Generates reference JSON dataset for customer profile metadata.
    """
    np.random.seed(seed)
    print(f"👤 Generating customer profiles JSON dataset ({num_customers:,} records)...")

    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.org", "invalid-domain"]
    tiers = ["Standard", "Silver", "Gold", "Platinum"]

    customers = []
    for uid in range(10000, 10000 + num_customers):
        domain = np.random.choice(domains, p=[0.45, 0.25, 0.20, 0.08, 0.02])
        email = f"user_{uid}@{domain}" if domain != "invalid-domain" else f"user_{uid}_at_invalid"
        customers.append({
            "user_id": uid,
            "email": email,
            "membership_tier": np.random.choice(tiers, p=[0.50, 0.30, 0.15, 0.05]),
            "account_created": f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,28):02d}",
        })

    with open(CUSTOMERS_JSON, "w") as f:
        json.dump(customers, f, indent=2)

    print(f"✅ Customer profiles saved: {CUSTOMERS_JSON}")


if __name__ == "__main__":
    generate_baseline_dataset()
    generate_incoming_corrupted_batch()
    generate_customer_profiles()
