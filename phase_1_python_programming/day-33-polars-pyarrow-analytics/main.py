"""
================================================================================
Day 33: Polars & PyArrow Frameworks — High-Performance Vectorized & Lazy DataFrames
================================================================================
Demonstrates ultra-fast analytical data processing using Polars and Apache Arrow:
1. Synthetic multi-million row dataset generation & Parquet persistence.
2. Eager API execution comparison: Pandas vs. Polars (Loading, Expressions, GroupBy).
3. Lazy API & Query Optimization: Predicate pushdown, Projection pushdown, Query plans.
4. Arrow Zero-Copy inter-operability & Memory-mapped streaming.
5. Complex Window Expressions (.over()) & High-speed streaming engine execution.
"""

import time
import os
import gc
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Define Output Paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_FILE = os.path.join(DATA_DIR, "transactions_large.parquet")
REPORT_FILE = os.path.join(DATA_DIR, "POLARS_BENCHMARK_REPORT.md")


def generate_synthetic_transactions(num_rows=2_000_000):
    """
    Generates a realistic multi-million row financial transaction dataset
    and writes it directly to Parquet format for fast loading benchmarks.
    """
    print(f"⚡ Generating {num_rows:,} synthetic financial transaction records...")
    np.random.seed(42)

    categories = ["Electronics", "Groceries", "Clothing", "Travel", "Entertainment", "Utilities", "Healthcare"]
    payment_methods = ["Credit Card", "Debit Card", "UPI", "Bank Transfer", "Crypto"]
    countries = ["US", "IN", "UK", "DE", "JP", "CA", "AU"]

    start_timestamp = pd.Timestamp("2025-01-01").value // 10**9
    end_timestamp = pd.Timestamp("2026-08-31").value // 10**9

    data = {
        "transaction_id": [f"TXN-{i:08d}" for i in range(1, num_rows + 1)],
        "user_id": np.random.randint(10000, 99999, size=num_rows),
        "timestamp_sec": np.random.randint(start_timestamp, end_timestamp, size=num_rows),
        "category": np.random.choice(categories, size=num_rows),
        "payment_method": np.random.choice(payment_methods, size=num_rows),
        "country": np.random.choice(countries, size=num_rows),
        "amount": np.round(np.random.exponential(scale=120.0, size=num_rows) + 5.0, 2),
        "risk_score": np.round(np.random.uniform(0.0, 1.0, size=num_rows), 4),
        "is_fraud": np.random.choice([0, 1], size=num_rows, p=[0.985, 0.015]),
    }

    # Convert to Polars DataFrame and export to Parquet
    df_pl = pl.DataFrame(data)
    
    # Cast timestamp integer to Datetime
    df_pl = df_pl.with_columns(
        (pl.col("timestamp_sec") * 1000).cast(pl.Datetime("ms")).alias("timestamp")
    ).drop("timestamp_sec")

    df_pl.write_parquet(PARQUET_FILE, compression="snappy")
    file_size_mb = os.path.getsize(PARQUET_FILE) / (1024 * 1024)
    print(f"✅ Created Parquet dataset: {PARQUET_FILE} ({file_size_mb:.2f} MB)")
    return file_size_mb


def benchmark_eager_pandas_vs_polars():
    """
    Benchmarks Pandas vs. Polars in Eager Mode across 3 core operations:
    1. Read Parquet File
    2. Filtered Aggregation (Complex Multi-Condition Filter + GroupBy)
    3. Column Transformation & Expression Calculation
    """
    print("\n" + "=" * 70)
    print("🚀 BENCHMARK 1: PANDAS vs. POLARS (EAGER MODE)")
    print("=" * 70)

    results = {}

    # --- 1. Read Parquet ---
    gc.collect()
    t0 = time.perf_counter()
    df_pd = pd.read_parquet(PARQUET_FILE)
    t_pd_read = time.perf_counter() - t0

    gc.collect()
    t0 = time.perf_counter()
    df_pl = pl.read_parquet(PARQUET_FILE)
    t_pl_read = time.perf_counter() - t0

    speedup_read = t_pd_read / t_pl_read
    results["Read Parquet"] = {"Pandas": t_pd_read, "Polars": t_pl_read, "Speedup": speedup_read}
    print(f"📖 File Read Time  : Pandas={t_pd_read:.4f}s | Polars={t_pl_read:.4f}s | Speedup={speedup_read:.2f}x")

    # --- 2. Filter & GroupBy Aggregation ---
    # Query: Filter high risk transactions (risk_score > 0.70 & amount > 100), group by category & country
    gc.collect()
    t0 = time.perf_counter()
    pd_filtered = df_pd[(df_pd["risk_score"] > 0.70) & (df_pd["amount"] > 100.0)]
    pd_grp = pd_filtered.groupby(["category", "country"]).agg(
        total_fraud_risk=("is_fraud", "sum"),
        avg_amount=("amount", "mean"),
        max_risk=("risk_score", "max"),
        txn_count=("transaction_id", "count")
    ).reset_index()
    t_pd_agg = time.perf_counter() - t0

    gc.collect()
    t0 = time.perf_counter()
    pl_grp = df_pl.filter(
        (pl.col("risk_score") > 0.70) & (pl.col("amount") > 100.0)
    ).group_by(["category", "country"]).agg(
        pl.col("is_fraud").sum().alias("total_fraud_risk"),
        pl.col("amount").mean().alias("avg_amount"),
        pl.col("risk_score").max().alias("max_risk"),
        pl.col("transaction_id").count().alias("txn_count")
    )
    t_pl_agg = time.perf_counter() - t0

    speedup_agg = t_pd_agg / t_pl_agg
    results["Filter & GroupBy"] = {"Pandas": t_pd_agg, "Polars": t_pl_agg, "Speedup": speedup_agg}
    print(f"📊 Filter & GroupBy: Pandas={t_pd_agg:.4f}s | Polars={t_pl_agg:.4f}s | Speedup={speedup_agg:.2f}x")

    # --- 3. Vectorized Column Expressions ---
    # Task: Compute fee (amount * 0.025), adjusted_risk (risk_score * log1p(amount)), category tag
    gc.collect()
    t0 = time.perf_counter()
    df_pd["fee"] = df_pd["amount"] * 0.025
    df_pd["adjusted_risk"] = df_pd["risk_score"] * np.log1p(df_pd["amount"])
    df_pd["high_value_flag"] = (df_pd["amount"] > 250.0).astype(int)
    t_pd_expr = time.perf_counter() - t0

    gc.collect()
    t0 = time.perf_counter()
    df_pl = df_pl.with_columns(
        (pl.col("amount") * 0.025).alias("fee"),
        (pl.col("risk_score") * pl.col("amount").log1p()).alias("adjusted_risk"),
        (pl.col("amount") > 250.0).cast(pl.Int32).alias("high_value_flag")
    )
    t_pl_expr = time.perf_counter() - t0

    speedup_expr = t_pd_expr / t_pl_expr
    results["Vectorized Expressions"] = {"Pandas": t_pd_expr, "Polars": t_pl_expr, "Speedup": speedup_expr}
    print(f"🧮 Multi-Expr Math : Pandas={t_pd_expr:.4f}s | Polars={t_pl_expr:.4f}s | Speedup={speedup_expr:.2f}x")

    return results


def benchmark_lazy_polars():
    """
    Demonstrates Polars Lazy API (pl.scan_parquet) & Query Plan Optimization:
    1. Predicate Pushdown (filtering before scanning entire dataset into RAM).
    2. Projection Pushdown (reading only required columns from Parquet file).
    3. Query Explanation (.explain()) showing optimized logical plan.
    """
    print("\n" + "=" * 70)
    print("🧠 BENCHMARK 2: POLARS LAZY API & QUERY OPTIMIZATION")
    print("=" * 70)

    # Construct Lazy Computation Graph
    lazy_plan = (
        pl.scan_parquet(PARQUET_FILE)
        .filter((pl.col("risk_score") > 0.85) & (pl.col("is_fraud") == 1))
        .select(["category", "payment_method", "amount", "risk_score"])
        .group_by(["category", "payment_method"])
        .agg([
            pl.col("amount").sum().alias("total_fraud_volume"),
            pl.col("amount").mean().alias("avg_fraud_amount"),
            pl.col("risk_score").max().alias("max_fraud_score"),
            pl.len().alias("high_risk_fraud_count"),
        ])
        .sort("total_fraud_volume", descending=True)
    )

    print("\n🔍 Logical & Optimized Query Plan (.explain()):")
    print("-" * 50)
    print(lazy_plan.explain())
    print("-" * 50)

    # Execute Lazy Query
    gc.collect()
    t0 = time.perf_counter()
    lazy_result = lazy_plan.collect()
    t_lazy = time.perf_counter() - t0

    # Equivalent Pandas scan + process for comparison
    gc.collect()
    t0 = time.perf_counter()
    pd_df = pd.read_parquet(PARQUET_FILE, columns=["category", "payment_method", "amount", "risk_score", "is_fraud"])
    pd_filtered = pd_df[(pd_df["risk_score"] > 0.85) & (pd_df["is_fraud"] == 1)]
    pd_res = pd_filtered.groupby(["category", "payment_method"]).agg(
        total_fraud_volume=("amount", "sum"),
        avg_fraud_amount=("amount", "mean"),
        max_fraud_score=("risk_score", "max"),
        high_risk_fraud_count=("amount", "count")
    ).sort_values(by="total_fraud_volume", ascending=False)
    t_pd_lazy_equiv = time.perf_counter() - t0

    lazy_speedup = t_pd_lazy_equiv / t_lazy
    print(f"⚡ Polars Lazy Execution Time : {t_lazy:.4f} seconds")
    print(f"🐢 Pandas Equivalent Scan Time : {t_pd_lazy_equiv:.4f} seconds")
    print(f"🚀 Lazy Query Engine Speedup   : {lazy_speedup:.2f}x faster than Pandas")

    return {"Polars_Lazy": t_lazy, "Pandas_Equiv": t_pd_lazy_equiv, "Speedup": lazy_speedup}


def benchmark_pyarrow_interop():
    """
    Evaluates zero-copy PyArrow memory bridge and inter-operability between
    PyArrow Tables, Polars DataFrames, and Pandas DataFrames.
    """
    print("\n" + "=" * 70)
    print("🏹 BENCHMARK 3: PYARROW ZERO-COPY MEMORY INTEROPERABILITY")
    print("=" * 70)

    # Load PyArrow Table directly from Parquet
    t0 = time.perf_counter()
    pa_table = pq.read_table(PARQUET_FILE)
    t_pa_read = time.perf_counter() - t0

    # PyArrow -> Polars (Zero-Copy Memory Pointer Sharing)
    t0 = time.perf_counter()
    pl_from_arrow = pl.from_arrow(pa_table)
    t_pa_to_pl = time.perf_counter() - t0

    # Polars -> PyArrow (Zero-Copy)
    t0 = time.perf_counter()
    pa_from_pl = pl_from_arrow.to_arrow()
    t_pl_to_pa = time.perf_counter() - t0

    # Polars -> Pandas
    t0 = time.perf_counter()
    pd_from_pl = pl_from_arrow.to_pandas()
    t_pl_to_pd = time.perf_counter() - t0

    print(f"📦 PyArrow Read Table Time         : {t_pa_read:.4f}s")
    print(f"🔄 PyArrow Table -> Polars DataFrame : {t_pa_to_pl:.6f}s (Zero-Copy Instant Bridge)")
    print(f"🔄 Polars DataFrame -> PyArrow Table : {t_pl_to_pa:.6f}s (Zero-Copy Instant Bridge)")
    print(f"🔄 Polars -> Pandas DataFrame       : {t_pl_to_pd:.4f}s")

    return {
        "PyArrow_Read": t_pa_read,
        "PyArrow_to_Polars": t_pa_to_pl,
        "Polars_to_PyArrow": t_pl_to_pa,
        "Polars_to_Pandas": t_pl_to_pd,
    }


if __name__ == "__main__":
    print("🔥 Starting Day 33: Polars & PyArrow Analytics Engine Benchmarks...")
    file_size = generate_synthetic_transactions(num_rows=2_000_000)
    eager_results = benchmark_eager_pandas_vs_polars()
    lazy_results = benchmark_lazy_polars()
    arrow_results = benchmark_pyarrow_interop()

