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
# SECTION 1 – GROUPBY BASICS (Split → Apply → Combine)
# ===========================================================

def groupby_basics(df: pd.DataFrame) -> None:
    """
    Introduces the three-step GroupBy paradigm in Pandas:
      1. Split   – divide the DataFrame into groups by one column
      2. Apply   – apply an aggregation function to each group
      3. Combine – merge the per-group results into a new DataFrame

    Key API:
      - df.groupby("col")          → creates a DataFrameGroupBy object
      - .size()                    → number of rows per group
      - .sum() / .mean() / .min()  → scalar aggregate per group
      - .agg("func")               → same but with a string shorthand
      - as_index=False             → keeps group keys as columns (not index)
    """
    print("\n" + "─" * 55)
    print("SECTION 1 – GroupBy Basics")
    print("─" * 55)

    # ── How many transactions does each region have? ──────────
    # groupby("region") splits the df into 4 sub-DataFrames (one per region).
    # .size() counts rows in each group and returns a Series indexed by region.
    txn_count = df.groupby("region").size()
    print("\n📌 Transaction count by region:")
    print(txn_count.to_string())          # .to_string() prevents truncation

    # ── Total revenue per region ──────────────────────────────
    # .sum() is applied to every numeric column; we then select only 'revenue'.
    rev_by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    print("\n📌 Total revenue by region (sorted descending):")
    print(rev_by_region.apply(lambda v: f"${v:,.2f}").to_string())

    # ── Average discount per product category ─────────────────
    # as_index=False returns a regular DataFrame instead of a grouped Series,
    # making it easier to pass to plotting functions downstream.
    avg_discount = (
        df.groupby("category", as_index=False)["discount"]
        .mean()
        .rename(columns={"discount": "avg_discount_pct"})
        .sort_values("avg_discount_pct", ascending=False)
    )
    print("\n📌 Average discount (%) by category:")
    print(avg_discount.to_string(index=False))

    # ── Min and Max unit price per product ────────────────────
    # You can chain multiple aggregation methods or use .agg() with a list.
    # Here we use a list to compute both in a single pass over the data.
    price_range = (
        df.groupby("product")["unit_price"]
        .agg(["min", "max"])            # returns a DataFrame with two columns
        .rename(columns={"min": "min_price", "max": "max_price"})
        .sort_values("max_price", ascending=False)
    )
    print("\n📌 Unit price range (min / max) by product:")
    print(price_range.round(2).to_string())

    # ── Visualise: Revenue by region (simple bar chart) ───────
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    ax.bar(rev_by_region.index, rev_by_region.values, color=colours)
    ax.set_title("Total Revenue by Region (2024)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v/1_000:,.0f}K")
    )
    for bar in ax.patches:
        # Annotate each bar with the exact dollar value above it
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f"${bar.get_height():,.0f}",
            ha="center", va="bottom", fontsize=9
        )
    plt.tight_layout()
    plt.savefig("s1_revenue_by_region.png", dpi=150)
    plt.show()
    print("\n   ✅ s1_revenue_by_region.png saved")


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

    # Section 1 – GroupBy basics: split → apply → combine
    groupby_basics(df)

    print("\n✅ Day 23 – Section 1 complete!")

