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



# ---------------------------------------------------------------------------
# 3. FEATHER FORMAT & PARQUET COMPRESSION CODEC BENCHMARKS
# ---------------------------------------------------------------------------

def benchmark_feather_io(df: pd.DataFrame) -> Dict[str, float]:
    """
    Benchmarks Arrow Feather V2 format for ultra-fast IPC & in-memory layout storage.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Dict[str, float]: Performance metrics for Feather format.
    """
    print("\n[INFO] Benchmark 2 — Arrow Feather V2 Format I/O:")
    data_dir = ensure_data_dir()

    uncompressed_path = os.path.join(data_dir, "transactions_uncompressed.feather")
    zstd_path = os.path.join(data_dir, "transactions_zstd.feather")

    # Uncompressed Feather
    start = time.perf_counter()
    feather.write_feather(df, uncompressed_path, compression='uncompressed')
    feather_uncomp_write = time.perf_counter() - start
    feather_uncomp_size = os.path.getsize(uncompressed_path) / (1024 ** 2)

    start = time.perf_counter()
    _ = feather.read_feather(uncompressed_path)
    feather_uncomp_read = time.perf_counter() - start

    print(f"  Feather (Uncompressed) Write: {feather_uncomp_write:.4f}s | Read: {feather_uncomp_read:.4f}s | Size: {feather_uncomp_size:.2f} MB")

    # ZSTD Compressed Feather
    start = time.perf_counter()
    feather.write_feather(df, zstd_path, compression='zstd')
    feather_zstd_write = time.perf_counter() - start
    feather_zstd_size = os.path.getsize(zstd_path) / (1024 ** 2)

    start = time.perf_counter()
    _ = feather.read_feather(zstd_path)
    feather_zstd_read = time.perf_counter() - start

    print(f"  Feather (ZSTD)         Write: {feather_zstd_write:.4f}s | Read: {feather_zstd_read:.4f}s | Size: {feather_zstd_size:.2f} MB")

    return {
        'feather_uncomp_write_sec': round(feather_uncomp_write, 4),
        'feather_uncomp_read_sec': round(feather_uncomp_read, 4),
        'feather_uncomp_size_mb': round(feather_uncomp_size, 2),
        'feather_zstd_write_sec': round(feather_zstd_write, 4),
        'feather_zstd_read_sec': round(feather_zstd_read, 4),
        'feather_zstd_size_mb': round(feather_zstd_size, 2),
    }


def compare_compression_codecs(df: pd.DataFrame) -> List[Dict]:
    """
    Compares Parquet file size, write time, and read time across compression algorithms:
    snappy, gzip, zstd, and NONE (uncompressed).

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        List[Dict]: Comparison breakdown for each compression codec.
    """
    print("\n[INFO] Benchmark 3 — Parquet Compression Codec Comparison:")
    data_dir = ensure_data_dir()

    codecs = ['snappy', 'gzip', 'zstd', 'NONE']
    results = []

    print(f"  {'Codec':<12} {'Write (s)':<12} {'Read (s)':<12} {'File Size (MB)':<16} {'Compress Ratio':<15}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*16} {'-'*15}")

    # Baseline CSV size for ratio calculation
    csv_path = os.path.join(data_dir, "transactions.csv")
    csv_size_mb = os.path.getsize(csv_path) / (1024 ** 2) if os.path.exists(csv_path) else 80.0

    for codec in codecs:
        out_file = os.path.join(data_dir, f"transactions_{codec.lower()}.parquet")
        comp_arg = None if codec == 'NONE' else codec.lower()

        # Write benchmark
        start = time.perf_counter()
        df.to_parquet(out_file, engine='pyarrow', compression=comp_arg, index=False)
        w_time = time.perf_counter() - start

        file_size_mb = os.path.getsize(out_file) / (1024 ** 2)
        compress_ratio = csv_size_mb / file_size_mb if file_size_mb > 0 else 1.0

        # Read benchmark
        start = time.perf_counter()
        _ = pd.read_parquet(out_file, engine='pyarrow')
        r_time = time.perf_counter() - start

        print(f"  {codec:<12} {w_time:<12.4f} {r_time:<12.4f} {file_size_mb:<16.2f} {compress_ratio:<15.2f}x")

        results.append({
            'codec': codec,
            'write_sec': round(w_time, 4),
            'read_sec': round(r_time, 4),
            'file_size_mb': round(file_size_mb, 2),
            'compression_ratio': round(compress_ratio, 2),
        })

    return results



# ---------------------------------------------------------------------------
# 4. DUCKDB IN-MEMORY SQL ANALYTICS & PANDAS INTEGRATION BENCHMARKS
# ---------------------------------------------------------------------------

def benchmark_duckdb_sql_queries(df: pd.DataFrame) -> Dict[str, float]:
    """
    Demonstrates zero-copy querying of Pandas DataFrames and Parquet files
    using DuckDB's vectorized analytical SQL engine.

    Args:
        df (pd.DataFrame): Input transaction DataFrame.

    Returns:
        Dict[str, float]: Timing results for DuckDB queries.
    """
    print("\n[INFO] Benchmark 4 — DuckDB Vectorized SQL Engine Queries:")
    data_dir = ensure_data_dir()
    parquet_path = os.path.join(data_dir, "transactions.parquet")

    # Connect to in-memory DuckDB instance
    con = duckdb.connect(database=':memory:')

    # Query 1: Direct SQL query on Pandas DataFrame variable 'df'
    start = time.perf_counter()
    sql_agg = con.query("""
        SELECT 
            category,
            payment_method,
            COUNT(*) AS total_transactions,
            ROUND(SUM(amount * (1 - discount_rate)), 2) AS net_revenue,
            ROUND(AVG(amount), 2) AS avg_transaction_val
        FROM df
        WHERE is_flagged = False
        GROUP BY category, payment_method
        ORDER BY net_revenue DESC
    """).df()
    df_sql_time = time.perf_counter() - start
    print(f"  Query 1: DuckDB SQL on Pandas DataFrame  : {df_sql_time:.4f}s  (groups: {len(sql_agg)})")

    # Query 2: SQL Window Functions (Dense Rank & Category Ranking)
    start = time.perf_counter()
    sql_window = con.query("""
        WITH ranked_sales AS (
            SELECT 
                transaction_id,
                customer_id,
                category,
                amount,
                DENSE_RANK() OVER (PARTITION BY category ORDER BY amount DESC) AS rank_in_category
            FROM df
        )
        SELECT * FROM ranked_sales WHERE rank_in_category <= 3
    """).df()
    window_sql_time = time.perf_counter() - start
    print(f"  Query 2: SQL Window Function (Top 3 per Cat): {window_sql_time:.4f}s  (rows: {len(sql_window)})")

    # Query 3: Direct SQL Query on Parquet File on Disk (No Pandas load needed!)
    start = time.perf_counter()
    parquet_sql = con.query(f"""
        SELECT 
            city,
            COUNT(DISTINCT customer_id) AS unique_customers,
            ROUND(AVG(amount), 2) AS avg_spent
        FROM '{parquet_path}'
        GROUP BY city
        ORDER BY avg_spent DESC
    """).df()
    parquet_sql_time = time.perf_counter() - start
    print(f"  Query 3: Direct SQL Query on Parquet File : {parquet_sql_time:.4f}s  (cities: {len(parquet_sql)})")

    con.close()

    return {
        'duckdb_df_sql_sec': round(df_sql_time, 4),
        'duckdb_window_sql_sec': round(window_sql_time, 4),
        'duckdb_parquet_sql_sec': round(parquet_sql_time, 4),
    }


def benchmark_duckdb_vs_pandas_groupby(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compares pure Pandas GroupBy aggregation vs DuckDB SQL engine execution.

    Args:
        df (pd.DataFrame): Input transaction DataFrame.

    Returns:
        Dict[str, float]: Timing results and speedup factor.
    """
    print("\n[INFO] Benchmark 5 — Pandas GroupBy vs DuckDB SQL Performance:")

    # Pure Pandas GroupBy
    start = time.perf_counter()
    unflagged = df[df['is_flagged'] == False].copy()
    unflagged['net_val'] = unflagged['amount'] * (1 - unflagged['discount_rate'])
    pandas_res = unflagged.groupby(['category', 'city'], observed=False).agg(
        total_tx=('transaction_id', 'count'),
        total_revenue=('net_val', 'sum'),
        avg_amount=('amount', 'mean')
    ).reset_index().sort_values('total_revenue', ascending=False)
    pandas_time = time.perf_counter() - start
    print(f"  Pandas GroupBy Pipeline : {pandas_time:.4f}s  (groups: {len(pandas_res)})")

    # DuckDB SQL Pipeline
    start = time.perf_counter()
    duck_res = duckdb.query("""
        SELECT 
            category,
            city,
            COUNT(transaction_id) AS total_tx,
            SUM(amount * (1 - discount_rate)) AS total_revenue,
            AVG(amount) AS avg_amount
        FROM df
        WHERE is_flagged = False
        GROUP BY category, city
        ORDER BY total_revenue DESC
    """).df()
    duck_time = time.perf_counter() - start
    print(f"  DuckDB SQL Pipeline     : {duck_time:.4f}s  (groups: {len(duck_res)})")

    speedup = pandas_time / duck_time if duck_time > 0 else float('inf')
    print(f"\n  [RESULTS] DuckDB vs Pandas Speedup: {speedup:.2f}x faster")

    return {
        'pandas_groupby_sec': round(pandas_time, 4),
        'duckdb_groupby_sec': round(duck_time, 4),
        'duckdb_speedup_factor': round(speedup, 2),
    }



# ---------------------------------------------------------------------------
# 5. SQLITE RELATIONAL DATABASE BENCHMARKS
# ---------------------------------------------------------------------------

def benchmark_sqlite_io(df: pd.DataFrame) -> Dict[str, float]:
    """
    Benchmarks SQLite database write, query, and index-accelerated lookups.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Dict[str, float]: Timing and database size stats.
    """
    print("\n[INFO] Benchmark 6 — SQLite Relational Database I/O & Indexing:")
    data_dir = ensure_data_dir()
    db_path = os.path.join(data_dir, "transactions.db")

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    # Convert categoricals back to string for sqlite compatibility
    df_sqlite = df.copy()
    for col in df_sqlite.select_dtypes(include=['category']).columns:
        df_sqlite[col] = df_sqlite[col].astype(str)

    # Write to SQLite
    start = time.perf_counter()
    df_sqlite.to_sql('transactions', conn, if_exists='replace', index=False)
    write_time = time.perf_counter() - start
    db_size_mb = os.path.getsize(db_path) / (1024 ** 2)
    print(f"  SQLite Write   : {write_time:.4f}s  | Database Size: {db_size_mb:.2f} MB")

    # Unindexed query lookup
    start = time.perf_counter()
    _ = pd.read_sql_query("SELECT * FROM transactions WHERE customer_id = 25000", conn)
    unindexed_query_time = time.perf_counter() - start
    print(f"  Unindexed Query: {unindexed_query_time:.4f}s")

    # Create Index
    start = time.perf_counter()
    conn.execute("CREATE INDEX idx_customer_id ON transactions (customer_id);")
    conn.commit()
    index_create_time = time.perf_counter() - start

    # Indexed query lookup
    start = time.perf_counter()
    _ = pd.read_sql_query("SELECT * FROM transactions WHERE customer_id = 25000", conn)
    indexed_query_time = time.perf_counter() - start
    print(f"  Indexed Query  : {indexed_query_time:.4f}s  (index creation: {index_create_time:.4f}s)")

    conn.close()

    idx_speedup = unindexed_query_time / indexed_query_time if indexed_query_time > 0 else float('inf')
    print(f"\n  [RESULTS] SQLite B-Tree Index Speedup: {idx_speedup:.2f}x faster")

    return {
        'sqlite_write_sec': round(write_time, 4),
        'sqlite_db_size_mb': round(db_size_mb, 2),
        'sqlite_unindexed_query_sec': round(unindexed_query_time, 4),
        'sqlite_indexed_query_sec': round(indexed_query_time, 4),
        'sqlite_index_speedup': round(idx_speedup, 2),
    }


# ---------------------------------------------------------------------------
# 6. EXECUTIVE PERFORMANCE SUMMARY REPORT
# ---------------------------------------------------------------------------

def generate_performance_report(
    csv_pq_stats: Dict[str, float],
    feather_stats: Dict[str, float],
    codec_stats: List[Dict],
    duck_query_stats: Dict[str, float],
    duck_vs_pd_stats: Dict[str, float],
    sqlite_stats: Dict[str, float],
) -> None:
    """
    Prints a formatted consolidated performance report summarizing storage
    efficiency and query execution across all storage formats and SQL engines.
    """
    sep = "=" * 78
    print(f"\n{sep}")
    print("  DAY 32: STORAGE FORMATS & DUCKDB SQL — CONSOLIDATED BENCHMARK REPORT")
    print(sep)

    print("\n  1. FILE STORAGE & COMPRESSION EFFICENCY (1,000,000 Rows)")
    print(f"     CSV (Plain Text)     : Size: {csv_pq_stats['csv_size_mb']:>7.2f} MB | Read: {csv_pq_stats['csv_read_sec']:>6.4f}s | Write: {csv_pq_stats['csv_write_sec']:>6.4f}s")
    print(f"     Parquet (Snappy)     : Size: {csv_pq_stats['parquet_size_mb']:>7.2f} MB | Read: {csv_pq_stats['parquet_read_sec']:>6.4f}s | Write: {csv_pq_stats['parquet_write_sec']:>6.4f}s")
    print(f"     Feather (Uncompressed): Size: {feather_stats['feather_uncomp_size_mb']:>7.2f} MB | Read: {feather_stats['feather_uncomp_read_sec']:>6.4f}s | Write: {feather_stats['feather_uncomp_write_sec']:>6.4f}s")
    print(f"     Feather (ZSTD)       : Size: {feather_stats['feather_zstd_size_mb']:>7.2f} MB | Read: {feather_stats['feather_zstd_read_sec']:>6.4f}s | Write: {feather_stats['feather_zstd_write_sec']:>6.4f}s")
    print(f"     SQLite Database      : Size: {sqlite_stats['sqlite_db_size_mb']:>7.2f} MB | Write: {sqlite_stats['sqlite_write_sec']:>6.4f}s")

    print("\n  2. PARQUET COMPRESSION CODEC BREAKDOWN")
    for c in codec_stats:
        print(f"     - Codec {c['codec']:<7} : File Size: {c['file_size_mb']:>6.2f} MB | Write: {c['write_sec']:>6.4f}s | Read: {c['read_sec']:>6.4f}s | Ratio: {c['compression_ratio']:>4.2f}x")

    print("\n  3. DUCKDB IN-MEMORY VECTORIZED SQL ENGINE PERFORMANCE")
    print(f"     - SQL Aggregation on Pandas DataFrame : {duck_query_stats['duckdb_df_sql_sec']:.4f}s")
    print(f"     - SQL Window Function (DENSE_RANK)    : {duck_query_stats['duckdb_window_sql_sec']:.4f}s")
    print(f"     - Direct SQL Query on Parquet File    : {duck_query_stats['duckdb_parquet_sql_sec']:.4f}s")
    print(f"     - DuckDB vs Pandas GroupBy Speedup   : {duck_vs_pd_stats['duckdb_speedup_factor']:.2f}x faster")

    print("\n  4. ARCHITECTURAL RECOMMENDATIONS & BEST PRACTICES")
    print("     • Cold Storage & Data Lakes    → Apache Parquet + ZSTD / Snappy (76%+ RAM/disk savings)")
    print("     • Inter-Process & Fast Caching → Arrow Feather Uncompressed (Sub-0.05s read speed)")
    print("     • Interactive SQL Analytics    → DuckDB over Pandas/Parquet (Vectorized C++ Execution)")
    print("     • Transactional/Embedded Apps  → SQLite with B-Tree Indexes")
    print(f"\n{sep}")


# ---------------------------------------------------------------------------
# 7. EXPORT BENCHMARK RESULTS TO CSV
# ---------------------------------------------------------------------------

def save_benchmark_results_to_csv(
    csv_pq_stats: Dict[str, float],
    feather_stats: Dict[str, float],
    codec_stats: List[Dict],
    duck_query_stats: Dict[str, float],
    duck_vs_pd_stats: Dict[str, float],
    sqlite_stats: Dict[str, float],
    output_filename: str = "benchmark_results_day32.csv"
) -> str:
    """
    Serializes all timing, file size, and speedup metrics to a CSV artifact.
    """
    records = []

    def _add(category: str, metric: str, value: float):
        records.append({'category': category, 'metric': metric, 'value': value})

    for k, v in csv_pq_stats.items():
        _add('csv_vs_parquet', k, v)
    for k, v in feather_stats.items():
        _add('feather_format', k, v)
    for c in codec_stats:
        for k, v in c.items():
            if k != 'codec':
                _add(f"parquet_codec_{c['codec'].lower()}", k, v)
    for k, v in duck_query_stats.items():
        _add('duckdb_sql_queries', k, v)
    for k, v in duck_vs_pd_stats.items():
        _add('duckdb_vs_pandas', k, v)
    for k, v in sqlite_stats.items():
        _add('sqlite_benchmarks', k, v)

    df_results = pd.DataFrame(records)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    df_results.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Benchmark metrics exported → {out_path} ({len(df_results)} metric rows)")
    return out_path


# ---------------------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 75)
    print("  DAY 32: BIG DATA STORAGE FORMATS & DUCKDB IN-MEMORY SQL ANALYTICS")
    print("  100-Day Data Science Challenge")
    print("=" * 75)

    df_transactions = generate_transaction_data(rows=1_000_000)

    # Storage & Format Benchmarks
    csv_pq_stats = benchmark_csv_vs_parquet_io(df_transactions)
    feather_stats = benchmark_feather_io(df_transactions)
    codec_stats = compare_compression_codecs(df_transactions)

    # DuckDB SQL Analytics Benchmarks
    duck_query_stats = benchmark_duckdb_sql_queries(df_transactions)
    duck_vs_pd_stats = benchmark_duckdb_vs_pandas_groupby(df_transactions)

    # SQLite Relational Database Benchmarks
    sqlite_stats = benchmark_sqlite_io(df_transactions)

    # Consolidated Executive Report
    generate_performance_report(
        csv_pq_stats=csv_pq_stats,
        feather_stats=feather_stats,
        codec_stats=codec_stats,
        duck_query_stats=duck_query_stats,
        duck_vs_pd_stats=duck_vs_pd_stats,
        sqlite_stats=sqlite_stats,
    )

    # Export Benchmark Results
    save_benchmark_results_to_csv(
        csv_pq_stats=csv_pq_stats,
        feather_stats=feather_stats,
        codec_stats=codec_stats,
        duck_query_stats=duck_query_stats,
        duck_vs_pd_stats=duck_vs_pd_stats,
        sqlite_stats=sqlite_stats,
    )



