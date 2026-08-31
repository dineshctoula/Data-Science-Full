"""
===============================================================================
DAY 32: MODERN BIG DATA STORAGE FORMATS (PARQUET, FEATHER) & DUCKDB SQL ANALYTICS
===============================================================================
Author: Dinesh (100-Day Data Science Challenge)
Phase: Phase 2 - Data Manipulation & Visualization
Topic: File Storage Benchmarks (CSV, Parquet, Feather, SQLite), Compression
       Types (Snappy, Gzip), DuckDB Zero-Copy In-Memory SQL Engine & Analytical Queries
===============================================================================
"""

import os
import time
import shutil
import warnings
import sqlite3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
import duckdb
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# Directory for temporary benchmark files
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage_benchmarks")

def ensure_data_dir() -> str:
    """Ensures storage_benchmarks directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return DATA_DIR


# ---------------------------------------------------------------------------
# 1. SYNTHETIC DATASET GENERATION
# ---------------------------------------------------------------------------

def generate_transaction_data(rows: int = 1_000_000) -> pd.DataFrame:
    """
    Generates a rich multi-column financial transaction dataset for storage
    and SQL benchmarking.

    Args:
        rows (int): Number of transaction rows to generate.

    Returns:
        pd.DataFrame: Synthetic dataset with datetime, categorical, float, int, and boolean fields.
    """
    print(f"\n[INFO] Generating transaction dataset with {rows:,} rows...")
    start_time = time.time()

    np.random.seed(42)

    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Automotive', 'Books', 'Health']
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Crypto', 'Bank Transfer']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Kathmandu', 'London', 'Tokyo']

    start_date = np.datetime64('2025-01-01')
    random_days = np.random.randint(0, 365, rows)
    timestamps = start_date + random_days.astype('timedelta64[D]') + np.random.randint(0, 86400, rows).astype('timedelta64[s]')

    data = {
        'transaction_id': np.arange(100_000, 100_000 + rows, dtype=np.int64),
        'timestamp': timestamps,
        'customer_id': np.random.randint(1000, 50000, rows, dtype=np.int32),
        'category': np.random.choice(categories, rows),
        'payment_method': np.random.choice(payment_methods, rows),
        'city': np.random.choice(cities, rows),
        'amount': np.round(np.random.exponential(scale=150.0, size=rows) + 5.0, 2),
        'discount_rate': np.round(np.random.uniform(0.0, 0.35, rows), 4),
        'quantity': np.random.randint(1, 20, rows, dtype=np.int16),
        'is_flagged': np.random.choice([True, False], rows, p=[0.02, 0.98]),
    }

    df = pd.DataFrame(data)

    # Cast string columns to category for optimal memory footprint
    for col in ['category', 'payment_method', 'city']:
        df[col] = df[col].astype('category')

    elapsed = time.time() - start_time
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[SUCCESS] Dataset generated in {elapsed:.2f}s | DataFrame RAM Footprint: {mem_mb:.2f} MB")

    return df


# ---------------------------------------------------------------------------
# 2. CSV vs PARQUET STORAGE BENCHMARKS
# ---------------------------------------------------------------------------

def benchmark_csv_vs_parquet_io(df: pd.DataFrame) -> Dict[str, float]:
    """
    Benchmarks I/O read/write speeds and file sizes between standard CSV and Parquet formats.

    Args:
        df (pd.DataFrame): Input transaction DataFrame.

    Returns:
        Dict[str, float]: Performance metrics for CSV vs Parquet.
    """
    print("\n[INFO] Benchmark 1 — CSV vs Apache Parquet (Snappy Compressed) I/O:")
    data_dir = ensure_data_dir()

    csv_path = os.path.join(data_dir, "transactions.csv")
    parquet_path = os.path.join(data_dir, "transactions.parquet")

    # Write CSV
    start = time.perf_counter()
    df.to_csv(csv_path, index=False)
    csv_write_time = time.perf_counter() - start
    csv_size_mb = os.path.getsize(csv_path) / (1024 ** 2)
    print(f"  CSV Write      : {csv_write_time:.4f}s  | File Size: {csv_size_mb:.2f} MB")

    # Read CSV
    start = time.perf_counter()
    _ = pd.read_csv(csv_path)
    csv_read_time = time.perf_counter() - start
    print(f"  CSV Read       : {csv_read_time:.4f}s")

    # Write Parquet (Snappy default)
    start = time.perf_counter()
    df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
    parquet_write_time = time.perf_counter() - start
    parquet_size_mb = os.path.getsize(parquet_path) / (1024 ** 2)
    print(f"  Parquet Write  : {parquet_write_time:.4f}s  | File Size: {parquet_size_mb:.2f} MB")

    # Read Parquet
    start = time.perf_counter()
    _ = pd.read_parquet(parquet_path, engine='pyarrow')
    parquet_read_time = time.perf_counter() - start
    print(f"  Parquet Read   : {parquet_read_time:.4f}s")

    # Comparison metrics
    size_reduction = (1 - parquet_size_mb / csv_size_mb) * 100
    read_speedup = csv_read_time / parquet_read_time if parquet_read_time > 0 else float('inf')
    write_speedup = csv_write_time / parquet_write_time if parquet_write_time > 0 else float('inf')

    print(f"\n  [RESULTS] Storage Saved: {size_reduction:.1f}% ({csv_size_mb:.2f}MB → {parquet_size_mb:.2f}MB)")
    print(f"            Read Speedup : {read_speedup:.2f}x faster")
    print(f"            Write Speedup: {write_speedup:.2f}x faster")

    return {
        'csv_write_sec': round(csv_write_time, 4),
        'csv_read_sec': round(csv_read_time, 4),
        'csv_size_mb': round(csv_size_mb, 2),
        'parquet_write_sec': round(parquet_write_time, 4),
        'parquet_read_sec': round(parquet_read_time, 4),
        'parquet_size_mb': round(parquet_size_mb, 2),
        'parquet_size_saving_pct': round(size_reduction, 2),
        'read_speedup': round(read_speedup, 2),
    }


if __name__ == "__main__":
    print("=" * 75)
    print("  DAY 32: BIG DATA STORAGE FORMATS & DUCKDB IN-MEMORY SQL ANALYTICS")
    print("  100-Day Data Science Challenge")
    print("=" * 75)

    df_transactions = generate_transaction_data(rows=1_000_000)
    csv_pq_stats = benchmark_csv_vs_parquet_io(df_transactions)
