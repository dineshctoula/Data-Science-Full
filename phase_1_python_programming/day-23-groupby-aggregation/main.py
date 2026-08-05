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
# SECTION 2 – AGGREGATION FUNCTIONS (agg with dicts)
# ===========================================================

def multi_metric_agg(df: pd.DataFrame) -> None:
    """
    Shows how to compute MULTIPLE aggregation metrics in a single
    GroupBy call using the .agg() method with a dictionary.

    Key concepts:
      - .agg({"col": ["func1", "func2"]})  → dict-based multi-metric agg
      - pd.NamedAgg(column=, aggfunc=)     → named aggregation for clean output
      - Custom lambda inside .agg()        → apply any arbitrary function
      - .round() chained on grouped result → round all numeric columns at once
    """
    print("\n" + "─" * 55)
    print("SECTION 2 – Multi-Metric Aggregation with agg()")
    print("─" * 55)

    # ── Multi-metric agg on a single column ───────────────────
    # Compute total revenue, average revenue, and transaction count per category.
    # When given a list, .agg() returns a MultiIndex DataFrame.
    rev_stats = (
        df.groupby("category")["revenue"]
        .agg(["sum", "mean", "count", "std"])
        .rename(columns={
            "sum":   "total_revenue",
            "mean":  "avg_revenue",
            "count": "transactions",
            "std":   "std_revenue",
        })
        .round(2)
        .sort_values("total_revenue", ascending=False)
    )
    print("\n📌 Revenue statistics by category:")
    print(rev_stats.to_string())

    # ── Named aggregation with pd.NamedAgg ────────────────────
    # pd.NamedAgg gives each output column a clean, explicit name.
    # This avoids MultiIndex columns and is the recommended modern approach.
    rep_summary = df.groupby("rep").agg(
        total_revenue  = pd.NamedAgg(column="revenue",    aggfunc="sum"),
        deals_closed   = pd.NamedAgg(column="revenue",    aggfunc="count"),
        avg_deal_size  = pd.NamedAgg(column="revenue",    aggfunc="mean"),
        avg_discount   = pd.NamedAgg(column="discount",   aggfunc="mean"),
        units_moved    = pd.NamedAgg(column="units_sold", aggfunc="sum"),
    ).round(2).sort_values("total_revenue", ascending=False)
    print("\n📌 Sales rep performance summary (NamedAgg):")
    print(rep_summary.to_string())

    # ── Custom lambda aggregation ─────────────────────────────
    # Lambda functions let us compute any metric not covered by built-ins.
    # Here: range (max-min) and coefficient of variation (std/mean * 100).
    custom_stats = df.groupby("category")["revenue"].agg(
        revenue_range = lambda s: s.max() - s.min(),     # peak-to-trough spread
        coeff_of_var  = lambda s: s.std() / s.mean() * 100,  # relative variability
    ).round(2)
    print("\n📌 Custom agg – revenue range & coefficient of variation:")
    print(custom_stats.to_string())

    # ── Visualise: stacked bar – total revenue by category + region ──
    # First create a pivot: rows = region, cols = category
    pivot = df.pivot_table(
        index="region", columns="category", values="revenue", aggfunc="sum"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=["#4C72B0", "#DD8452", "#55A868"],
               edgecolor="white", linewidth=0.5)
    ax.set_title("Total Revenue by Region & Category (Stacked)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v/1_000:,.0f}K")
    )
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("s2_stacked_bar.png", dpi=150)
    plt.show()
    print("\n   ✅ s2_stacked_bar.png saved")


# ===========================================================
# SECTION 3 – MULTI-LEVEL GROUPBY (Group by Two+ Columns)
# ===========================================================

def multi_level_groupby(df: pd.DataFrame) -> None:
    """
    Demonstrates grouping by more than one column at the same time.
    Multi-level GroupBy returns a hierarchically indexed (MultiIndex) result.

    Key concepts:
      - df.groupby(["col1", "col2"])  → creates a two-level group
      - .unstack()                    → pivots the inner index to columns
      - .swaplevel() + .sort_index()  → rearrange MultiIndex levels
      - Useful pattern: group by (time period + category) for trend analysis
    """
    print("\n" + "─" * 55)
    print("SECTION 3 – Multi-Level GroupBy")
    print("─" * 55)

    # ── Extract month from the date column for time-series grouping ──
    # Pandas datetime accessor (.dt) gives access to year, month, day, etc.
    df = df.copy()                          # avoid mutating the caller's df
    df["month"] = df["date"].astype("datetime64[ns]").dt.month

    # ── Group by region + category ───────────────────────────────────
    # Result is a Series with a 2-level MultiIndex: (region, category)
    regional_cat = (
        df.groupby(["region", "category"])["revenue"]
        .sum()
        .round(2)
    )
    print("\n📌 Total revenue by region × category (MultiIndex Series):")
    print(regional_cat.to_string())

    # ── Unstack inner level → pivot MultiIndex into wide DataFrame ───
    # .unstack() moves the innermost index level ("category") to columns.
    # This makes it easy to compare categories side-by-side for each region.
    wide = regional_cat.unstack(level="category").fillna(0).round(2)
    print("\n📌 Wide-format (regions as rows, categories as columns):")
    print(wide.to_string())

    # ── Monthly revenue trend per category ──────────────────────────
    # Grouping by month + category gives a 12×3 pivot-ready result.
    monthly_cat = (
        df.groupby(["month", "category"])["revenue"]
        .sum()
        .unstack("category")           # months as rows, categories as columns
        .fillna(0)
        .round(2)
    )
    print("\n📌 Monthly revenue by category (first 3 months shown):")
    print(monthly_cat.head(3).to_string())

    # ── Group by region + rep to rank individual performance ────────
    rep_perf = (
        df.groupby(["region", "rep"]).agg(
            total_revenue = pd.NamedAgg("revenue",    "sum"),
            deals         = pd.NamedAgg("revenue",    "count"),
        )
        .round(2)
        .sort_values("total_revenue", ascending=False)
    )
    print("\n📌 Revenue by region → rep (MultiIndex, sorted by revenue):")
    print(rep_perf.to_string())

    # ── Visualise: monthly revenue trend per category ────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    for col, colour in zip(monthly_cat.columns, colours):
        ax.plot(monthly_cat.index, monthly_cat[col],
                marker="o", linewidth=2, color=colour, label=col)

    ax.set_title("Monthly Revenue Trend by Category (2024)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v/1_000:,.0f}K")
    )
    ax.legend(title="Category", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("s3_monthly_trend.png", dpi=150)
    plt.show()
    print("\n   ✅ s3_monthly_trend.png saved")


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

    # Section 2 – Multi-metric aggregation with agg() and NamedAgg
    multi_metric_agg(df)

    # Section 3 – Multi-level GroupBy by two or more columns
    multi_level_groupby(df)

    print("\n✅ Day 23 – Sections 1–3 complete!")



