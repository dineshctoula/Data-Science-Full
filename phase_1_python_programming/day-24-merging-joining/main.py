"""
Day 24 – Pandas Data Merging, Joining & Concatenating
=====================================================
100-Day Data Science Challenge | Phase 2: Data Manipulation & Visualization

Topics Covered:
  1. Concatenation (pd.concat) – vertical stacking, horizontal alignment, key hierarchies
  2. Database-Style Merges (pd.merge) – inner, left, right, outer joins & _merge indicators
  3. Multi-Key & Self-Joins – merging on multiple columns, custom suffixes, organizational hierarchies
  4. Index Merging & Validation – left_index/right_index, df.join(), combine_first(), validate parameter
  5. Executive Visualization – joining multi-table insights into clean visuals

Learning Goals:
  - Combine multiple datasets cleanly using SQL-like relational join strategies
  - Stacking periodic batch data efficiently with pd.concat()
  - Master self-joins to resolve hierarchical relationships (e.g. employee-manager trees)
  - Enforce schema constraints and data integrity using merge validation rules (1:1, 1:m, m:1)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Path helper for saving output plots inside the module folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────
# Relational Datasets Generator
# ──────────────────────────────────────────────────────────────
# Simulates a real-world enterprise database schema:
#   - customers_df : customer demographics & tier
#   - orders_q1_df  : Q1 transactions ledger
#   - orders_q2_df  : Q2 transactions ledger
#   - products_df   : product catalog with cost & retail price
#   - returns_df    : return log with return reasons
#   - employees_df  : staff directory with manager_id for self-joins

def make_enterprise_datasets(seed: int = 42) -> dict:
    """
    Generate synthetic relational enterprise datasets.

    Returns:
        dict: Dictionary containing DataFrames:
              'customers', 'orders_q1', 'orders_q2', 'products', 'returns', 'employees'
    """
    rng = np.random.default_rng(seed)

    # 1. Customer Directory (100 customers, IDs 101-200)
    customer_ids = np.arange(101, 201)
    regions = ["North", "South", "East", "West"]
    tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    
    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "name": [f"Customer_{cid}" for cid in customer_ids],
        "region": rng.choice(regions, size=len(customer_ids)),
        "tier": rng.choice(tiers, size=len(customer_ids), p=[0.4, 0.3, 0.2, 0.1]),
        "signup_year": rng.choice([2021, 2022, 2023, 2024], size=len(customer_ids)),
    })

    # 2. Product Catalog (8 products)
    products_data = {
        "product_id": ["P10", "P20", "P30", "P40", "P50", "P60", "P70", "P80"],
        "product_name": ["Laptop", "Monitor", "Keyboard", "Mouse", "Desk Lamp", "Headphones", "Ergonomic Chair", "Webcam"],
        "category": ["Electronics", "Electronics", "Accessories", "Accessories", "Furniture", "Electronics", "Furniture", "Electronics"],
        "unit_cost": [600.0, 180.0, 35.0, 15.0, 25.0, 70.0, 150.0, 40.0],
        "retail_price": [999.99, 299.99, 59.99, 29.99, 49.99, 119.99, 249.99, 79.99],
    }
    products_df = pd.DataFrame(products_data)

    # Helper function to build transaction batch logs
    def make_order_batch(start_order_id: int, count: int, start_date: str) -> pd.DataFrame:
        o_ids = np.arange(start_order_id, start_order_id + count)
        # Randomly select active customers (some customers buy multiple times, some none)
        c_ids = rng.choice(customer_ids[:85], size=count)
        p_ids = rng.choice(products_df["product_id"], size=count)
        qty = rng.integers(1, 6, size=count)
        
        dates = pd.date_range(start=start_date, periods=count, freq="12h")
        
        return pd.DataFrame({
            "order_id": o_ids,
            "customer_id": c_ids,
            "product_id": p_ids,
            "order_date": dates,
            "quantity": qty,
        })

    # 3. Orders Batches (Q1 & Q2)
    orders_q1_df = make_order_batch(1001, 120, "2024-01-01")
    orders_q2_df = make_order_batch(1121, 130, "2024-04-01")

    # 4. Returns Log (Subset of orders returned)
    all_order_ids = np.concatenate([orders_q1_df["order_id"], orders_q2_df["order_id"]])
    returned_order_ids = rng.choice(all_order_ids, size=25, replace=False)
    reasons = ["Defective", "Wrong Size", "Changed Mind", "Late Delivery"]
    
    returns_df = pd.DataFrame({
        "return_id": np.arange(501, 501 + len(returned_order_ids)),
        "order_id": returned_order_ids,
        "return_date": pd.date_range(start="2024-01-15", periods=len(returned_order_ids), freq="5D"),
        "reason": rng.choice(reasons, size=len(returned_order_ids)),
    })

    # 5. Employees Directory for Self-Joins (Organizational Tree)
    employees_df = pd.DataFrame({
        "emp_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "emp_name": ["Alice (CEO)", "Bob (VP Sales)", "Carol (VP Tech)", "Dave (Sales Mgr)",
                     "Eve (Sales Rep)", "Frank (Sales Rep)", "Grace (Tech Lead)", "Hank (Dev)",
                     "Ivy (Dev)", "Jack (QA)"],
        "manager_id": [np.nan, 1, 1, 2, 4, 4, 3, 7, 7, 3],
        "department": ["Exec", "Sales", "Tech", "Sales", "Sales", "Sales", "Tech", "Tech", "Tech", "Tech"],
        "salary": [180000, 130000, 140000, 95000, 65000, 62000, 110000, 85000, 82000, 70000],
    })

    return {
        "customers": customers_df,
        "orders_q1": orders_q1_df,
        "orders_q2": orders_q2_df,
        "products": products_df,
        "returns": returns_df,
        "employees": employees_df,
    }


# ===========================================================
# ENTRY POINT
# ===========================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  Day 24 – Pandas Data Merging, Joining & Concatenating")
    print("=" * 62)

    # Initialize enterprise relational datasets
    datasets = make_enterprise_datasets()
    print("\n✅ Enterprise Datasets Loaded:")
    for name, df in datasets.items():
        print(f"   • {name:<12}: {df.shape[0]:>3} rows × {df.shape[1]} columns")
