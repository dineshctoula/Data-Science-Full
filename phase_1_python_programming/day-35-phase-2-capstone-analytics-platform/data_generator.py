"""
Day 35 Phase 2 Capstone: Enterprise Multi-Source Synthetic Data Generator
========================================================================
This module generates realistic, multi-source enterprise data modeling an e-commerce ecosystem:
1. Transactional Data (Parquet) - Order details, pricing, discount, payment status, anomalies.
2. Customer CRM Data (JSON) - Customer demographic attributes, signup date, loyalty status.
3. Web Clickstream Data (CSV) - User session interactions, traffic sources, conversion tags.
4. Product Catalog (CSV) - Product hierarchy, category, unit cost, MSRP, inventory levels.

Author: 100-Day Data Science Challenge Team
"""

import os
import json
import random
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta

# Set deterministic random seeds for reproducible data generation
random.seed(42)
np.random.seed(42)

def generate_customer_crm(num_customers: int = 5000) -> pd.DataFrame:
    """
    Generates synthetic Customer Relationship Management (CRM) data in JSON structure.
    
    Parameters:
        num_customers (int): Total number of unique customer profiles to generate.
        
    Returns:
        pd.DataFrame: Customer profiles with demographic and loyalty attributes.
    """
    tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']
    tier_weights = [0.50, 0.30, 0.15, 0.05]  # Standard Pareto distribution of loyalty tiers
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia']
    country_weights = [0.45, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2026, 8, 31)
    time_span_days = (end_date - start_date).days

    customer_records = []
    for cid in range(1001, 1001 + num_customers):
        signup_offset = random.randint(0, time_span_days)
        signup_dt = start_date + timedelta(days=signup_offset)
        
        profile = {
            "customer_id": f"CUST-{cid}",
            "registration_date": signup_dt.strftime("%Y-%m-%d"),
            "country": np.random.choice(countries, p=country_weights),
            "age": int(np.clip(np.random.normal(38, 12), 18, 75)),  # Truncated normal distribution for age
            "loyalty_tier": np.random.choice(tiers, p=tier_weights),
            "email_opt_in": random.random() > 0.3,
            "credit_score": int(np.clip(np.random.normal(710, 50), 550, 850))
        }
        customer_records.append(profile)
        
    return pd.DataFrame(customer_records)


def generate_product_catalog(num_products: int = 250) -> pd.DataFrame:
    """
    Generates synthetic Product Catalog metadata.
    
    Parameters:
        num_products (int): Number of product SKUs to construct.
        
    Returns:
        pd.DataFrame: Product attributes including categories, MSRP, cost, and stock.
    """
    categories = {
        'Electronics': (50.0, 1200.0, 0.35),   # (min_price, max_price, profit_margin)
        'Fashion': (15.0, 300.0, 0.55),
        'Home & Kitchen': (20.0, 500.0, 0.45),
        'Books & Media': (8.0, 80.0, 0.40),
        'Sports & Outdoors': (25.0, 650.0, 0.40),
        'Beauty & Health': (10.0, 200.0, 0.60)
    }

    products = []
    cat_names = list(categories.keys())
    
    for pid in range(501, 501 + num_products):
        cat = random.choice(cat_names)
        min_p, max_p, margin = categories[cat]
        msrp = round(random.uniform(min_p, max_p), 2)
        unit_cost = round(msrp * (1.0 - margin), 2)
        
        product = {
            "product_id": f"PROD-{pid}",
            "product_name": f"{cat} Item #{pid}",
            "category": cat,
            "msrp": msrp,
            "unit_cost": unit_cost,
            "stock_quantity": random.randint(0, 1500),
            "is_active": random.random() > 0.05
        }
        products.append(product)

    return pd.DataFrame(products)


def generate_transactions(
    num_orders: int,
    customer_ids: list,
    product_ids: list,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Generates high-volume transactional data with intentional data quality issues:
    - Missing customer IDs (nulls) to simulate guest/unlinked checkouts.
    - Outlier high order values (anomalies).
    - Payment status distribution (Completed, Pending, Failed, Refunded).
    """
    payment_methods = ['Credit Card', 'PayPal', 'Apple Pay', 'Bank Transfer', 'Crypto']
    payment_weights = [0.55, 0.25, 0.12, 0.06, 0.02]

    statuses = ['Completed', 'Completed', 'Completed', 'Completed', 'Refunded', 'Pending', 'Failed']
    
    time_span_seconds = int((end_date - start_date).total_seconds())

    transactions = []
    for oid in range(100001, 100001 + num_orders):
        # Generate random timestamp with peak hours simulation
        random_secs = random.randint(0, time_span_seconds)
        txn_timestamp = start_date + timedelta(seconds=random_secs)

        # Inject ~2% missing customer IDs for unlinked checkouts
        cust_id = random.choice(customer_ids) if random.random() > 0.02 else None
        prod_id = random.choice(product_ids)
        
        quantity = np.random.negative_binomial(1, 0.4) + 1  # Skewed small purchase quantities
        unit_price = round(float(np.random.lognormal(mean=3.5, sigma=0.8)), 2)
        
        # Inject occasional extreme pricing outlier anomaly (~0.1%)
        if random.random() < 0.001:
            unit_price *= 25.0
            
        discount_rate = round(random.choice([0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30]), 2)
        gross_amount = round(quantity * unit_price, 2)
        net_amount = round(gross_amount * (1.0 - discount_rate), 2)
        
        # Inject occasional negative net amount bug (~0.05%) to test cleaning pipeline
        if random.random() < 0.0005:
            net_amount = -abs(net_amount)

        transactions.append({
            "order_id": f"ORD-{oid}",
            "transaction_timestamp": txn_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "customer_id": cust_id,
            "product_id": prod_id,
            "quantity": int(quantity),
            "unit_price": unit_price,
            "discount_rate": discount_rate,
            "gross_amount": gross_amount,
            "net_amount": net_amount,
            "payment_method": np.random.choice(payment_methods, p=payment_weights),
            "order_status": random.choice(statuses)
        })

    return pd.DataFrame(transactions)


def generate_clickstream_events(num_events: int, customer_ids: list) -> pd.DataFrame:
    """
    Generates web/mobile application clickstream logs.
    
    Parameters:
        num_events (int): Total number of session clickstream events.
        customer_ids (list): List of registered customer IDs.
        
    Returns:
        pd.DataFrame: Clickstream event records with session duration, traffic channel, and conversion.
    """
    channels = ['Organic Search', 'Direct', 'Paid Ads', 'Social Media', 'Email Campaign', 'Referral']
    channel_weights = [0.35, 0.25, 0.20, 0.10, 0.06, 0.04]

    devices = ['Desktop', 'Mobile Web', 'Mobile App', 'Tablet']
    device_weights = [0.45, 0.35, 0.15, 0.05]

    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 8, 31)
    time_span_secs = int((end_date - start_date).total_seconds())

    events = []
    for eid in range(500001, 500001 + num_events):
        session_dt = start_date + timedelta(seconds=random.randint(0, time_span_secs))
        cust_id = random.choice(customer_ids) if random.random() > 0.15 else None  # ~15% anonymous visitors
        
        pages_viewed = np.random.poisson(lam=4.5) + 1
        session_duration = int(pages_viewed * np.random.exponential(scale=45.0) + random.randint(5, 30))
        converted = random.random() < (0.04 if cust_id else 0.015)  # Higher conversion for logged-in users

        events.append({
            "session_id": f"SESS-{eid}",
            "timestamp": session_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "customer_id": cust_id,
            "traffic_channel": np.random.choice(channels, p=channel_weights),
            "device_type": np.random.choice(devices, p=device_weights),
            "pages_viewed": int(pages_viewed),
            "session_duration_seconds": session_duration,
            "has_converted": converted
        })

    return pd.DataFrame(events)


def build_enterprise_datalake(output_dir: str) -> dict:
    """
    Main orchestration routine for multi-source data generation.
    Saves datasets in appropriate formats (Parquet, JSON, CSV).
    
    Parameters:
        output_dir (str): Destination folder path.
        
    Returns:
        dict: Summary of generated file paths and record counts.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"📦 [Data Generator] Initializing multi-source dataset generation in '{output_dir}'...")

    # 1. Generate Customer CRM (JSON)
    df_customers = generate_customer_crm(num_customers=4000)
    customer_json_path = os.path.join(output_dir, "customer_crm.json")
    with open(customer_json_path, 'w') as f:
        json.dump(df_customers.to_dict(orient='records'), f, indent=2)
    print(f"  └─ Customer CRM: {len(df_customers):,} records saved to JSON.")

    # 2. Generate Product Catalog (CSV)
    df_products = generate_product_catalog(num_products=200)
    product_csv_path = os.path.join(output_dir, "product_catalog.csv")
    df_products.to_csv(product_csv_path, index=False)
    print(f"  └─ Product Catalog: {len(df_products):,} products saved to CSV.")

    customer_ids = df_customers['customer_id'].tolist()
    product_ids = df_products['product_id'].tolist()

    # 3. Generate Baseline & Current Transaction Batches (Parquet)
    # Baseline batch: Jan 2025 - Dec 2025 (Reference distribution for PSI drift detection)
    df_txn_baseline = generate_transactions(
        num_orders=35000,
        customer_ids=customer_ids,
        product_ids=product_ids,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 12, 31)
    )
    baseline_parquet_path = os.path.join(output_dir, "baseline_transactions.parquet")
    df_txn_baseline.to_parquet(baseline_parquet_path, index=False)
    print(f"  └─ Baseline Transactions: {len(df_txn_baseline):,} records saved to Parquet.")

    # Current batch: Jan 2026 - Aug 2026 (Operational distribution)
    df_txn_current = generate_transactions(
        num_orders=50000,
        customer_ids=customer_ids,
        product_ids=product_ids,
        start_date=datetime(2026, 1, 1),
        end_date=datetime(2026, 8, 31)
    )
    current_parquet_path = os.path.join(output_dir, "current_transactions.parquet")
    df_txn_current.to_parquet(current_parquet_path, index=False)
    print(f"  └─ Current Transactions: {len(df_txn_current):,} records saved to Parquet.")

    # 4. Generate Clickstream Web Event Logs (CSV)
    df_clickstream = generate_clickstream_events(num_events=60000, customer_ids=customer_ids)
    clickstream_csv_path = os.path.join(output_dir, "web_clickstream.csv")
    df_clickstream.to_csv(clickstream_csv_path, index=False)
    print(f"  └─ Web Clickstream: {len(df_clickstream):,} sessions saved to CSV.")

    print("✅ [Data Generator] Enterprise Data Lake initialization complete!\n")
    return {
        "customer_crm": customer_json_path,
        "product_catalog": product_csv_path,
        "baseline_transactions": baseline_parquet_path,
        "current_transactions": current_parquet_path,
        "web_clickstream": clickstream_csv_path
    }

if __name__ == "__main__":
    target_dir = os.path.join(os.path.dirname(__file__), "raw_datalake")
    build_enterprise_datalake(target_dir)
