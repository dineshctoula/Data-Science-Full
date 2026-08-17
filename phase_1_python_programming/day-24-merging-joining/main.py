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
    # We use numpy's arange to generate sequential IDs.
    customer_ids = np.arange(101, 201)
    # Define possible regions and loyalty tiers for random assignment
    regions = ["North", "South", "East", "West"]
    tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    
    # Create the customers DataFrame
    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "name": [f"Customer_{cid}" for cid in customer_ids],
        "region": rng.choice(regions, size=len(customer_ids)),
        # Tiers are weighted to simulate a typical distribution (mostly Bronze/Silver)
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
# SECTION 1 – CONCATENATION (pd.concat)
# ===========================================================

def concatenation_basics(q1_df: pd.DataFrame, q2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrates combining DataFrames using pd.concat().

    Key concepts:
      - pd.concat([df1, df2], axis=0)       → vertical stacking (rows)
      - ignore_index=True                   → reindex from 0 to N-1
      - keys=['Q1', 'Q2']                   → creates hierarchical MultiIndex
      - axis=1                              → horizontal side-by-side join (columns)
      - handling mismatched columns        → outer join keeps all, inner join keeps common
    """
    print("\n" + "─" * 58)
    print("SECTION 1 – Concatenation with pd.concat()")
    print("─" * 58)

    # Simulate Q2 having an extra operational column 'promotion_code'
    q2_enhanced = q2_df.copy()
    q2_enhanced["promotion_code"] = np.random.choice(["SAVE10", "WELCOME5", None], size=len(q2_df))

    # ── Vertical Stacking (Default: axis=0) ───────────────────
    # Combine Q1 and Q2 order logs into a unified annual ledger.
    # Without ignore_index=True, original index numbers (0..119, 0..129) are preserved.
    # This might lead to duplicate index values, which can cause issues during lookups.
    raw_concat = pd.concat([q1_df, q2_enhanced], axis=0)
    print(f"\n📌 Simple vertical concat (mismatched cols aligned, NaNs inserted):")
    print(f"   Shape: {raw_concat.shape} | Duplicate index values present: {raw_concat.index.has_duplicates}")

    # Reset index to maintain continuous integer index 0..249
    # Setting ignore_index=True ensures a clean, unique sequential index for the new DataFrame.
    all_orders = pd.concat([q1_df, q2_enhanced], axis=0, ignore_index=True)
    print(f"\n📌 Clean vertical concat (ignore_index=True):")
    print(f"   Shape: {all_orders.shape} | Continuous index: {all_orders.index.is_monotonic_increasing}")
    print(all_orders.head(3).to_string())

    # ── Hierarchical Indexing (keys parameter) ─────────────────
    # Tag each batch with its period source to preserve origin provenance.
    tagged_orders = pd.concat([q1_df, q2_df], keys=["Q1", "Q2"])
    print(f"\n📌 MultiIndex Concat (keys=['Q1', 'Q2']):")
    print(tagged_orders.loc[("Q1", 0):("Q1", 2)].to_string())

    # ── Inner vs Outer Join on Concatenation ───────────────────
    # inner join drops columns not present in both DataFrames
    inner_concat = pd.concat([q1_df, q2_enhanced], axis=0, join="inner", ignore_index=True)
    print(f"\n📌 Inner Concat (drops 'promotion_code' column present only in Q2):")
    print(f"   Columns retained: {list(inner_concat.columns)}")

    # ── Horizontal Concatenation (axis=1) ──────────────────────
    # Combine summary statistics side-by-side by matching row index positions
    q1_summary = q1_df.groupby("customer_id")["quantity"].sum().rename("q1_qty")
    q2_summary = q2_df.groupby("customer_id")["quantity"].sum().rename("q2_qty")
    
    side_by_side = pd.concat([q1_summary, q2_summary], axis=1).fillna(0)
    print(f"\n📌 Horizontal Concat (axis=1) – Q1 vs Q2 Customer Quantities (sample):")
    print(side_by_side.head(5).to_string())

    # ── Visualise: Monthly order batch volumes ─────────────────
    all_orders["month_name"] = all_orders["order_date"].dt.strftime("%b %Y")
    monthly_counts = all_orders.groupby("month_name", sort=False).size()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(monthly_counts.index, monthly_counts.values, color="#34495E", width=0.55, edgecolor="black")
    ax.set_title("Concatenated Order Ledger – Monthly Batch Volume", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Order Count")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "s1_concat_batches.png"), dpi=150)
    plt.show()
    print("\n   ✅ s1_concat_batches.png saved")

    return all_orders



# ===========================================================
# SECTION 2 – DATABASE-STYLE MERGES (pd.merge)
# ===========================================================

def database_merges(orders_df: pd.DataFrame, customers_df: pd.DataFrame,
                    products_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrates SQL-style joins using pd.merge().

    Key concepts:
      - how='inner'   → keep keys present in BOTH DataFrames
      - how='left'    → keep ALL rows from left DataFrame, match right
      - how='right'   → keep ALL rows from right DataFrame, match left
      - how='outer'   → keep ALL rows from BOTH DataFrames (fill missing with NaN)
      - indicator=True → adds '_merge' column ('left_only', 'right_only', 'both')
      - Chaining multi-table merges for financial revenue/profit analysis
    """
    print("\n" + "─" * 58)
    print("SECTION 2 – Relational Joins with pd.merge()")
    print("─" * 58)

    # ── 1. Inner Join (orders + customers) ──────────────────────
    # Only returns orders where customer_id exists in both datasets.
    inner_df = pd.merge(orders_df, customers_df, on="customer_id", how="inner")
    print(f"\n📌 Inner Join (Orders ⋈ Customers):")
    print(f"   Total joined orders: {len(inner_df):,} | Columns: {list(inner_df.columns)}")

    # ── 2. Left Join & Inactive Customer Discovery ─────────────
    # Keep ALL customers regardless of whether they placed an order.
    # NaNs in order_id indicate inactive customers who haven't bought yet.
    cust_orders_left = pd.merge(customers_df, orders_df, on="customer_id", how="left", indicator=True)
    inactive_custs = cust_orders_left[cust_orders_left["_merge"] == "left_only"]
    print(f"\n📌 Left Join (Customers ⟕ Orders):")
    print(f"   Total customer records: {len(cust_orders_left):,}")
    print(f"   Active customers: {cust_orders_left['customer_id'].nunique() - len(inactive_custs)}")
    print(f"   Inactive customers (0 purchases): {len(inactive_custs)} (e.g. IDs: {inactive_custs['customer_id'].head(3).tolist()})")

    # ── 3. Outer Join with Merge Indicator ──────────────────────
    # Reconcile orders log against returns log.
    orders_returns_outer = pd.merge(
        orders_df, returns_df, on="order_id", how="outer", indicator="return_status"
    )
    status_counts = orders_returns_outer["return_status"].value_counts()
    print(f"\n📌 Outer Join with Indicator (Orders ⟗ Returns):")
    print(f"   Status breakdown:")
    print(f"   - Kept (Not Returned - left_only)  : {status_counts.get('left_only', 0):>3}")
    print(f"   - Matched (Returned - both)        : {status_counts.get('both', 0):>3}")
    print(f"   - Orphan Returns (right_only)       : {status_counts.get('right_only', 0):>3}")

    # ── 4. Chained Multi-Table Merge (Orders → Products → Customers) ───
    # Build complete business transaction view: calculates gross profit per order.
    full_ledger = (
        orders_df
        .merge(products_df, on="product_id", how="left")
        .merge(customers_df, on="customer_id", how="left")
    )
    
    # Calculate computed financial metrics
    full_ledger["gross_revenue"] = full_ledger["quantity"] * full_ledger["retail_price"]
    full_ledger["total_cost"]    = full_ledger["quantity"] * full_ledger["unit_cost"]
    full_ledger["gross_profit"]  = full_ledger["gross_revenue"] - full_ledger["total_cost"]

    print(f"\n📌 Chained Multi-Table Ledger (Orders ⋈ Products ⋈ Customers):")
    print(f"   Total Gross Revenue : ${full_ledger['gross_revenue'].sum():,.2f}")
    print(f"   Total Gross Profit  : ${full_ledger['gross_profit'].sum():,.2f}")
    print(full_ledger[["order_id", "name", "product_name", "quantity", "gross_revenue", "gross_profit"]].head(4).to_string(index=False))

    # ── Visualise: Join Type Record Count Comparison ───────────
    join_metrics = {
        "Inner (Orders & Cust)": len(inner_df),
        "Left (Cust & Orders)": len(cust_orders_left),
        "Outer (Orders & Returns)": len(orders_returns_outer),
        "Returns (Matched)": status_counts.get("both", 0),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#2980B9", "#27AE60", "#8E44AD", "#C0392B"]
    ax.bar(join_metrics.keys(), join_metrics.values(), color=colours, width=0.5, edgecolor="black")
    ax.set_title("Record Count Comparison Across Join Types", fontsize=12, fontweight="bold")
    ax.set_ylabel("Record Count")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 3, f"{int(height)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "s2_join_types.png"), dpi=150)
    plt.show()
    print("\n   ✅ s2_join_types.png saved")

    return full_ledger



# ===========================================================
# SECTION 3 – MULTI-KEY MERGES, SUFFIXES & SELF-JOINS
# ===========================================================

def advanced_merges_and_self_joins(full_ledger: pd.DataFrame, returns_df: pd.DataFrame,
                                   employees_df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrates multi-key joins, suffix resolution, and self-joins.

    Key concepts:
      - on=['col1', 'col2']                 → multi-key join matching multiple criteria
      - left_on='a', right_on='b'           → joining on differently named keys
      - suffixes=('_order', '_return')      → resolve name collisions on shared columns
      - Self-Join (df.merge(df, ...))       → join a table to itself to resolve hierarchical trees
    """
    print("\n" + "─" * 58)
    print("SECTION 3 – Multi-Key Merges, Suffixes & Self-Joins")
    print("─" * 58)

    # ── 1. Suffix Disambiguation on Overlapping Column Names ────
    # Both full_ledger and returns_df have date/id columns.
    returns_merged = pd.merge(
        full_ledger, returns_df, on="order_id", how="left", suffixes=("_order", "_return")
    )
    print(f"\n📌 Suffix Resolution (suffixes=('_order', '_return')):")
    print(f"   Columns with suffixes: {[c for c in returns_merged.columns if '_order' in c or '_return' in c]}")
    print(returns_merged[["order_id", "order_date", "return_date", "reason"]].dropna(subset=["return_date"]).head(3).to_string(index=False))

    # ── 2. Multi-Key Join Example ──────────────────────────────
    # Match transactions by both customer_id and region to find localized regional sales.
    # We construct a regional promo budget DataFrame to join on (customer_id, region)
    regional_promos = full_ledger[["customer_id", "region"]].drop_duplicates().copy()
    regional_promos["promo_tier"] = np.random.choice(["Tier_A", "Tier_B"], size=len(regional_promos))

    multi_key_df = pd.merge(full_ledger, regional_promos, on=["customer_id", "region"], how="left")
    print(f"\n📌 Multi-Key Join (on=['customer_id', 'region']):")
    print(f"   Total rows: {len(multi_key_df):,} | Sample promo tier assignment:")
    print(multi_key_df[["order_id", "customer_id", "region", "promo_tier"]].head(3).to_string(index=False))

    # ── 3. Self-Join on Employee Hierarchy ──────────────────────
    # Maps employees to their respective direct managers within the same DataFrame.
    org_chart = pd.merge(
        employees_df,
        employees_df[["emp_id", "emp_name", "salary", "department"]],
        left_on="manager_id",
        right_on="emp_id",
        how="left",
        suffixes=("_emp", "_mgr"),
    )

    # Compute salary differential between manager and employee
    org_chart["salary_diff"] = org_chart["salary_mgr"] - org_chart["salary_emp"]

    print(f"\n📌 Self-Join – Employee Organizational Tree:")
    print(org_chart[["emp_name_emp", "department_emp", "salary_emp", "emp_name_mgr", "salary_mgr", "salary_diff"]].to_string(index=False))

    # ── Visualise: Salary Distribution by Department & Manager Differential ──
    fig, ax = plt.subplots(figsize=(8, 4))
    dept_sal = org_chart.groupby("department_emp")["salary_emp"].mean()
    
    bars = ax.bar(dept_sal.index, dept_sal.values, color=["#E67E22", "#2980B9", "#27AE60"], width=0.5, edgecolor="black")
    ax.set_title("Average Salary by Department (Resolved via Self-Join Hierarchy)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Salary ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1_000:,.0f}K"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2000, f"${height:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "s3_self_join_org.png"), dpi=150)
    plt.show()
    print("\n   ✅ s3_self_join_org.png saved")

    return org_chart



# ===========================================================
# SECTION 4 – INDEX MERGING, COMBINE_FIRST & VALIDATION
# ===========================================================

def index_merges_and_validation(customers_df: pd.DataFrame, products_df: pd.DataFrame,
                                full_ledger: pd.DataFrame) -> None:
    """
    Demonstrates index-based joins, combine_first patching, and merge validation.

    Key concepts:
      - left_index=True / right_index=True  → join using index keys instead of columns
      - df.join(other_df, how=...)          → index-focused join convenience method
      - df.combine_first(backup_df)         → patch missing values from a fallback dataset
      - validate='1:m' / 'm:1' / '1:1'      → enforce relational schema integrity checks
    """
    print("\n" + "─" * 58)
    print("SECTION 4 – Index Merging, combine_first() & Validation")
    print("─" * 58)

    # Prepare index-indexed DataFrames
    cust_indexed = customers_df.set_index("customer_id")
    prod_indexed = products_df.set_index("product_id")

    # ── 1. Index-based Join using df.join() ────────────────────
    # Join orders with indexed customer directory using customer_id key
    orders_sample = full_ledger.head(5).copy()
    joined_by_index = orders_sample.join(cust_indexed, on="customer_id", lsuffix="_ledger")
    print(f"\n📌 Index Join via df.join(on='customer_id'):")
    print(joined_by_index[["order_id", "customer_id", "name", "region", "gross_revenue"]].to_string(index=False))

    # ── 2. combine_first() for Fallback Data Patching ───────────
    # Simulate a main product catalog with missing unit prices, patched from a secondary backup
    incomplete_catalog = prod_indexed[["product_name", "retail_price"]].copy()
    incomplete_catalog.loc[["P10", "P30"], "retail_price"] = np.nan

    backup_catalog = prod_indexed[["product_name", "retail_price"]].copy()

    patched_catalog = incomplete_catalog.combine_first(backup_catalog)
    print(f"\n📌 combine_first() Data Patching:")
    print(f"   Missing values before patch: {incomplete_catalog['retail_price'].isna().sum()}")
    print(f"   Missing values after patch:  {patched_catalog['retail_price'].isna().sum()}")

    # ── 3. Merge Validation Rules (Data Integrity Enforcement) ─
    # validate='1:m' ensures customer_id is unique on the left side (1 customer -> many orders)
    try:
        valid_merge = pd.merge(customers_df, full_ledger, on="customer_id", how="inner", validate="1:m")
        print(f"\n📌 Merge Validation (validate='1:m'): Success ✅")
        print(f"   Schema constraint verified: customer_id is unique in left DataFrame.")
    except Exception as e:
        print(f"\n❌ Merge Validation Failed: {e}")

    # ── 4. Visualise: Executive Multi-Panel Revenue & Profit Dashboard ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Gross Revenue by Customer Tier
    tier_revenue = full_ledger.groupby("tier")["gross_revenue"].sum().reindex(["Bronze", "Silver", "Gold", "Platinum"])
    ax1.bar(tier_revenue.index, tier_revenue.values, color="#3498DB", edgecolor="black", width=0.5)
    ax1.set_title("Gross Revenue by Customer Tier", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Revenue ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1_000:,.0f}K"))
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in ax1.patches:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1000, f"${h/1_000:,.1f}K", ha="center", va="bottom", fontsize=8)

    # Panel 2: Profit Margin % by Product Category
    cat_financials = full_ledger.groupby("category")[["gross_revenue", "gross_profit"]].sum()
    cat_financials["profit_margin_pct"] = (cat_financials["gross_profit"] / cat_financials["gross_revenue"]) * 100

    ax2.bar(cat_financials.index, cat_financials["profit_margin_pct"], color="#E74C3C", edgecolor="black", width=0.45)
    ax2.set_title("Gross Profit Margin (%) by Product Category", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Margin %")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in ax2.patches:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "s4_executive_summary.png"), dpi=150)
    plt.show()
    print("\n   ✅ s4_executive_summary.png saved")


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

    # Section 1 – Concatenation (pd.concat)
    all_orders_df = concatenation_basics(datasets["orders_q1"], datasets["orders_q2"])

    # Section 2 – Database-Style Merges (pd.merge)
    full_ledger_df = database_merges(
        all_orders_df, datasets["customers"], datasets["products"], datasets["returns"]
    )

    # Section 3 – Multi-Key Merges, Suffixes & Self-Joins
    org_chart_df = advanced_merges_and_self_joins(
        full_ledger_df, datasets["returns"], datasets["employees"]
    )

    # Section 4 – Index Merging, combine_first() & Validation
    index_merges_and_validation(
        datasets["customers"], datasets["products"], full_ledger_df
    )

    print("\n" + "=" * 62)
    print("  ✅ Day 24 – All Sections Complete!")
    print("  Topics: pd.concat, pd.merge, Self-Joins, Index Joins, Validation")
    print("=" * 62)
