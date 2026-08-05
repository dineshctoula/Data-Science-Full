"""
Day 23 – Pandas GroupBy & Aggregation
======================================
100-Day Data Science Challenge | Phase 2: Data Manipulation & Visualization

Topics Covered:
  1. GroupBy Basics        – split → apply → combine pattern
  2. Aggregation Functions – sum, mean, count, min, max, custom lambdas
  3. Multi-Level GroupBy   – group by two or more columns simultaneously
  4. Transform & Filter    – group-aware row-level transformations
  5. Pivot Tables          – cross-tabulation summaries for quick insight

Learning Goals:
  - Understand the split-apply-combine paradigm at the core of GroupBy
  - Use agg() with dicts to compute multiple metrics in one step
  - Apply transform() to create group-normalised features
  - Build pivot tables and cross-tabs for executive-style summaries
  - Combine GroupBy results with Matplotlib for labelled bar charts
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Shared Dataset – used throughout all sections of this module
# ──────────────────────────────────────────────────────────────
# We simulate a retail sales ledger with realistic attributes so
# every groupby example answers a genuine business question.

def make_sales_df(seed: int = 42) -> pd.DataFrame:
    """
    Build and return a synthetic retail sales DataFrame.

    Columns:
      - date       : transaction date (Jan–Dec 2024)
      - region     : sales region (North / South / East / West)
      - category   : product category (Electronics / Clothing / Food)
      - product    : specific product name within the category
      - rep        : sales representative name
      - units_sold : number of units sold per transaction
      - unit_price : price per unit in USD
      - discount   : discount percentage applied (0–30 %)
      - revenue    : units_sold × unit_price × (1 – discount/100)
    """
    rng = np.random.default_rng(seed)

    # ── Dimension values ────────────────────────────────────────
    regions    = ["North", "South", "East", "West"]
    categories = ["Electronics", "Clothing", "Food"]

    # Each category has three specific products
    products = {
        "Electronics": ["Laptop", "Headphones", "Tablet"],
        "Clothing":    ["Jacket", "Jeans", "Sneakers"],
        "Food":        ["Coffee", "Granola", "Protein Bar"],
    }

    # Sales representatives assigned to each region
    reps_by_region = {
        "North": ["Alice", "Bob"],
        "South": ["Carol", "Dave"],
        "East":  ["Eve",   "Frank"],
        "West":  ["Grace", "Hank"],
    }

    n = 600   # total number of transactions

    # ── Generate random categorical columns ────────────────────
    region_col   = rng.choice(regions,    size=n)
    category_col = rng.choice(categories, size=n)

    # Product must be consistent with category
    product_col  = np.array([
        rng.choice(products[cat]) for cat in category_col
    ])

    # Rep must be consistent with region
    rep_col = np.array([
        rng.choice(reps_by_region[reg]) for reg in region_col
    ])

    # ── Generate numeric columns ────────────────────────────────
    units      = rng.integers(1, 20, size=n)          # 1–19 units per sale
    base_price = {
        "Electronics": 350, "Clothing": 65, "Food": 12
    }
    # Add category-specific noise to price
    unit_price = np.array([
        base_price[cat] + rng.normal(0, base_price[cat] * 0.15)
        for cat in category_col
    ]).clip(5, None).round(2)

    discount   = rng.integers(0, 31, size=n)          # 0–30 % discount
    revenue    = (units * unit_price * (1 - discount / 100)).round(2)

    # ── Random dates spread across 2024 ────────────────────────
    start = np.datetime64("2024-01-01")
    end   = np.datetime64("2024-12-31")
    days_range = (end - start).astype(int)
    dates = start + rng.integers(0, days_range, size=n)

    df = pd.DataFrame({
        "date":       dates,
        "region":     region_col,
        "category":   category_col,
        "product":    product_col,
        "rep":        rep_col,
        "units_sold": units,
        "unit_price": unit_price,
        "discount":   discount,
        "revenue":    revenue,
    }).sort_values("date").reset_index(drop=True)

    return df


# ===========================================================
# ENTRY POINT
# ===========================================================

if __name__ == "__main__":
    print("=" * 58)
    print("  Day 23 – Pandas GroupBy & Aggregation")
    print("=" * 58)

    # Build the shared dataset once; pass it to each section
    df = make_sales_df()
    print(f"\n✅ Dataset created: {len(df):,} rows × {df.shape[1]} columns")
    print(df.head())
    print(f"\nColumn dtypes:\n{df.dtypes}")

    print("\n✅ Day 23 scaffold complete – dataset ready!")
