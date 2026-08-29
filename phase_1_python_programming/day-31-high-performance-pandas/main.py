"""
===============================================================================
DAY 31: HIGH-PERFORMANCE DATA PROCESSING WITH PANDAS
===============================================================================
Author: Dinesh (100-Day Data Science Challenge)
Phase: Phase 2 - Data Manipulation & Visualization
Topic: Performance Optimization, Eval, Query, Vectorization, Chunking,
       Memory Optimization, Dtype Casting, Parallel Processing (NumPy UFunc)
===============================================================================
"""

import os
import time
import math
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------------------------

def generate_large_dataset(rows: int = 1_000_000) -> pd.DataFrame:
    """
    Generates a large synthetic dataset to benchmark pandas performance.
    Uses NumPy for fast in-memory construction before handing off to pandas.

    Args:
        rows (int): Number of rows to generate.

    Returns:
        pd.DataFrame: Unoptimized raw dataset for benchmarking.
    """
    print(f"\n[INFO] Generating large dataset with {rows:,} rows...")
    start_time = time.time()

    np.random.seed(42)

    categories = ['Alpha', 'Beta', 'Gamma', 'Delta']
    regions    = ['North', 'South', 'East', 'West', 'Central']

    data = {
        'id'         : np.arange(rows, dtype=np.int32),
        'value_a'    : np.random.normal(100, 25, rows).astype(np.float64),
        'value_b'    : np.random.normal(50, 15, rows).astype(np.float64),
        'score'      : np.random.randint(0, 1000, rows).astype(np.int32),
        'quantity'   : np.random.randint(1, 200, rows).astype(np.int16),
        'category'   : np.random.choice(categories, rows),   # object → will be cast
        'region'     : np.random.choice(regions, rows),       # object → will be cast
        'is_active'  : np.random.choice([True, False], rows),
        'price'      : np.round(np.random.uniform(5.0, 500.0, rows), 2),
        'discount'   : np.round(np.random.uniform(0.0, 0.5, rows), 4),
    }

    df = pd.DataFrame(data)

    elapsed_time = time.time() - start_time
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[SUCCESS] Dataset generated in {elapsed_time:.2f}s | "
          f"Memory: {mem_mb:.2f} MB")

    return df


# ---------------------------------------------------------------------------
# 2. MEMORY OPTIMIZATION — DTYPE DOWNCASTING & CATEGORICALS
# ---------------------------------------------------------------------------

def optimize_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Reduces DataFrame memory footprint by:
      - Downcasting integer columns (int64 → int32/int16/int8)
      - Downcasting float columns (float64 → float32)
      - Converting low-cardinality string columns to 'category'

    Args:
        df (pd.DataFrame): Raw DataFrame with default dtypes.

    Returns:
        Tuple[pd.DataFrame, Dict]: Optimized DataFrame + memory stats dict.
    """
    print("\n[INFO] Optimizing DataFrame dtypes for memory efficiency...")

    mem_before = df.memory_usage(deep=True).sum() / (1024 ** 2)

    df_opt = df.copy()

    # Downcast integer columns
    int_cols = df_opt.select_dtypes(include=['int64', 'int32']).columns
    for col in int_cols:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')

    # Downcast float columns
    float_cols = df_opt.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')

    # Convert low-cardinality object columns to category
    obj_cols = df_opt.select_dtypes(include=['object']).columns
    for col in obj_cols:
        n_unique = df_opt[col].nunique()
        n_total  = len(df_opt)
        # Rule of thumb: convert to category if cardinality < 5% of total rows
        if n_unique / n_total < 0.05:
            df_opt[col] = df_opt[col].astype('category')
            print(f"  → '{col}' converted to category  "
                  f"(cardinality: {n_unique})")

    mem_after = df_opt.memory_usage(deep=True).sum() / (1024 ** 2)
    reduction = (1 - mem_after / mem_before) * 100

    stats = {
        'mem_before_mb' : round(mem_before, 2),
        'mem_after_mb'  : round(mem_after, 2),
        'reduction_pct' : round(reduction, 2),
    }

    print(f"\n  Memory Before : {mem_before:.2f} MB")
    print(f"  Memory After  : {mem_after:.2f} MB")
    print(f"  Reduction     : {reduction:.1f}%")

    return df_opt, stats


def show_dtype_comparison(df_raw: pd.DataFrame, df_opt: pd.DataFrame) -> None:
    """Prints a side-by-side dtype comparison table."""
    print("\n[INFO] Dtype Comparison (Raw vs Optimized):")
    print(f"  {'Column':<15} {'Raw Dtype':<15} {'Opt Dtype':<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15}")
    for col in df_raw.columns:
        raw_dtype = str(df_raw[col].dtype)
        opt_dtype = str(df_opt[col].dtype)
        changed   = "  ✓" if raw_dtype != opt_dtype else ""
        print(f"  {col:<15} {raw_dtype:<15} {opt_dtype:<15}{changed}")


# ---------------------------------------------------------------------------
# 3. QUERY & EVAL PERFORMANCE BENCHMARKS
# ---------------------------------------------------------------------------

def benchmark_query_vs_masking(df: pd.DataFrame) -> Dict[str, float]:
    """
    Benchmarks standard boolean masking vs pandas.query().

    Args:
        df (pd.DataFrame): Input DataFrame (should have value_a, value_b,
                           is_active columns).

    Returns:
        Dict[str, float]: Timing results for masking and query.
    """
    print("\n[INFO] Benchmark 1 — Boolean Masking vs pd.query():")

    # Standard Boolean Masking
    start = time.perf_counter()
    result_mask = df[
        (df['value_a'] > 120) &
        (df['value_b'] < 40) &
        (df['is_active'] == True)
    ]
    mask_time = time.perf_counter() - start
    print(f"  Masking  : {mask_time:.4f}s  (rows: {len(result_mask):,})")

    # Pandas Query (uses numexpr under the hood when available)
    start = time.perf_counter()
    result_query = df.query("value_a > 120 and value_b < 40 and is_active == True")
    query_time = time.perf_counter() - start
    print(f"  pd.query : {query_time:.4f}s  (rows: {len(result_query):,})")

    speedup = mask_time / query_time if query_time > 0 else float('inf')
    winner  = "pd.query" if query_time < mask_time else "masking"
    print(f"  Winner   : {winner}  (speedup: {speedup:.2f}x)")

    return {'mask_time': mask_time, 'query_time': query_time}


def benchmark_eval_vs_standard(df: pd.DataFrame) -> Dict[str, float]:
    """
    Benchmarks standard arithmetic operations vs pandas.eval().

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Dict[str, float]: Timing results for standard math and eval.
    """
    print("\n[INFO] Benchmark 2 — Standard Math vs pd.eval():")

    df_copy1 = df.copy()
    df_copy2 = df.copy()

    # Standard Arithmetic
    start = time.perf_counter()
    df_copy1['net_value'] = df_copy1['value_a'] * 2 + df_copy1['value_b'] / 3 - 50
    standard_time = time.perf_counter() - start
    print(f"  Standard : {standard_time:.4f}s")

    # Pandas Eval
    start = time.perf_counter()
    df_copy2.eval("net_value = value_a * 2 + value_b / 3 - 50", inplace=True)
    eval_time = time.perf_counter() - start
    print(f"  pd.eval  : {eval_time:.4f}s")

    speedup = standard_time / eval_time if eval_time > 0 else float('inf')
    winner  = "pd.eval" if eval_time < standard_time else "standard"
    print(f"  Winner   : {winner}  (speedup: {speedup:.2f}x)")

    return {'standard_time': standard_time, 'eval_time': eval_time}


# ---------------------------------------------------------------------------
# 4. VECTORIZATION vs APPLY BENCHMARK
# ---------------------------------------------------------------------------

def benchmark_vectorization_vs_apply(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compares three approaches for a conditional computation:
      1. Row-wise apply  (slowest)
      2. np.where vectorization  (fastest for simple conditionals)
      3. Numba-style manual ufunc simulation with np operations

    Operates on a 100k-row subset to keep apply() feasible.

    Args:
        df (pd.DataFrame): Full DataFrame to sample from.

    Returns:
        Dict[str, float]: Timing results for each approach.
    """
    print("\n[INFO] Benchmark 3 — Row Apply vs np.where Vectorization:")
    subset = df.head(100_000).copy()

    # ---- 1. Row-wise apply ----
    def custom_logic(row):
        if row['is_active']:
            return row['value_a'] ** 2 - row['value_b'] + row['quantity']
        return row['value_a'] / 2 + row['discount'] * 100

    start = time.perf_counter()
    _ = subset.apply(custom_logic, axis=1)
    apply_time = time.perf_counter() - start
    print(f"  apply()    : {apply_time:.4f}s  (100k rows)")

    # ---- 2. np.where vectorization ----
    start = time.perf_counter()
    _ = np.where(
        subset['is_active'],
        subset['value_a'] ** 2 - subset['value_b'] + subset['quantity'],
        subset['value_a'] / 2 + subset['discount'] * 100
    )
    vec_time = time.perf_counter() - start
    print(f"  np.where   : {vec_time:.4f}s  (100k rows)")

    # ---- 3. Pure-NumPy masked array approach ----
    start = time.perf_counter()
    mask = subset['is_active'].values
    va   = subset['value_a'].values
    vb   = subset['value_b'].values
    qty  = subset['quantity'].values.astype(np.float32)
    disc = subset['discount'].values
    result_numpy = np.where(mask,
                            va ** 2 - vb + qty,
                            va / 2 + disc * 100)
    numpy_time = time.perf_counter() - start
    print(f"  NumPy raw  : {numpy_time:.4f}s  (100k rows)")

    # Speedups
    print(f"\n  Speedup np.where vs apply : {apply_time / vec_time:.1f}x")
    print(f"  Speedup NumPy raw vs apply: {apply_time / numpy_time:.1f}x")

    return {
        'apply_time'  : apply_time,
        'vec_time'    : vec_time,
        'numpy_time'  : numpy_time,
    }


# ---------------------------------------------------------------------------
# 5. CHUNKED I/O — SIMULATED FILE PROCESSING
# ---------------------------------------------------------------------------

def benchmark_chunked_processing(df: pd.DataFrame) -> Dict[str, float]:
    """
    Simulates chunked CSV processing vs loading the full DataFrame at once.
    Writes the DataFrame to a temp CSV, then reads it in two modes:
      a) pd.read_csv full load
      b) pd.read_csv with chunksize, processing each chunk incrementally

    This demonstrates the memory-efficient pattern for large-file ETL.

    Args:
        df (pd.DataFrame): Source DataFrame to serialize.

    Returns:
        Dict[str, float]: Timing and aggregate result for both modes.
    """
    print("\n[INFO] Benchmark 4 — Full CSV Load vs Chunked CSV Processing:")

    # Use a 200k-row subset to keep I/O time manageable
    sample = df.head(200_000)
    tmp_path = "/tmp/day31_benchmark.csv"
    sample.to_csv(tmp_path, index=False)

    # ---- Full Load ----
    start = time.perf_counter()
    df_full = pd.read_csv(tmp_path)
    full_sum = df_full['score'].sum()
    full_time = time.perf_counter() - start
    print(f"  Full load   : {full_time:.4f}s  (score sum: {full_sum:,.0f})")

    # ---- Chunked Load ----
    CHUNK_SIZE = 10_000
    start = time.perf_counter()
    chunk_sum = 0
    for chunk in pd.read_csv(tmp_path, chunksize=CHUNK_SIZE):
        chunk_sum += chunk['score'].sum()
    chunk_time = time.perf_counter() - start
    print(f"  Chunked ({CHUNK_SIZE:,}): {chunk_time:.4f}s  (score sum: {chunk_sum:,.0f})")

    # Cleanup temp file
    os.remove(tmp_path)

    note = ("Chunked processing uses less peak memory — preferred for files "
            "exceeding available RAM, even if slower for small files.")
    print(f"\n  [NOTE] {note}")

    return {
        'full_load_time'  : full_time,
        'chunked_load_time': chunk_time,
        'score_sum_match' : full_sum == chunk_sum,
    }


# ---------------------------------------------------------------------------
# 6. GROUPBY OPTIMIZATION — TRANSFORM vs APPLY
# ---------------------------------------------------------------------------

def benchmark_groupby_operations(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compares pd.GroupBy performance for common aggregation patterns:
      1. groupby().apply() with a custom function
      2. groupby().transform() — vectorized broadcast back to original index
      3. groupby().agg()  — fastest for standard aggregations

    Args:
        df (pd.DataFrame): Input DataFrame with 'category' and numeric columns.

    Returns:
        Dict[str, float]: Timing results.
    """
    print("\n[INFO] Benchmark 5 — GroupBy: apply() vs transform() vs agg():")

    df_gb = df.head(500_000).copy()

    # ---- 1. groupby + apply ----
    start = time.perf_counter()
    _ = df_gb.groupby('category').apply(
        lambda g: g['value_a'].mean()
    )
    apply_time = time.perf_counter() - start
    print(f"  groupby.apply()     : {apply_time:.4f}s")

    # ---- 2. groupby + transform ----
    start = time.perf_counter()
    _ = df_gb.groupby('category')['value_a'].transform('mean')
    transform_time = time.perf_counter() - start
    print(f"  groupby.transform() : {transform_time:.4f}s")

    # ---- 3. groupby + agg ----
    start = time.perf_counter()
    _ = df_gb.groupby('category').agg(
        mean_val_a  = ('value_a', 'mean'),
        total_score = ('score', 'sum'),
        count       = ('id', 'count'),
    )
    agg_time = time.perf_counter() - start
    print(f"  groupby.agg()       : {agg_time:.4f}s")

    print(f"\n  transform/apply speedup: {apply_time / transform_time:.1f}x")
    print(f"  agg/apply speedup      : {apply_time / agg_time:.1f}x")

    return {
        'groupby_apply_time'    : apply_time,
        'groupby_transform_time': transform_time,
        'groupby_agg_time'      : agg_time,
    }


# ---------------------------------------------------------------------------
# 7. PERFORMANCE SUMMARY REPORT
# ---------------------------------------------------------------------------

def generate_performance_report(
    mem_stats    : Dict[str, float],
    query_stats  : Dict[str, float],
    eval_stats   : Dict[str, float],
    vec_stats    : Dict[str, float],
    chunk_stats  : Dict[str, float],
    groupby_stats: Dict[str, float],
) -> None:
    """
    Prints a formatted, consolidated performance summary report
    combining all benchmark results from the session.

    Args:
        mem_stats     : Memory optimization stats.
        query_stats   : Query vs masking benchmark.
        eval_stats    : Eval vs standard math benchmark.
        vec_stats     : Vectorization vs apply benchmark.
        chunk_stats   : Chunked I/O benchmark.
        groupby_stats : GroupBy approach benchmark.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  DAY 31 — HIGH-PERFORMANCE PANDAS: CONSOLIDATED BENCHMARK REPORT")
    print(sep)

    # --- Memory ---
    print("\n  1. MEMORY OPTIMIZATION")
    print(f"     Before  : {mem_stats['mem_before_mb']:.2f} MB")
    print(f"     After   : {mem_stats['mem_after_mb']:.2f} MB")
    print(f"     Saved   : {mem_stats['reduction_pct']:.1f}%  "
          f"({mem_stats['mem_before_mb'] - mem_stats['mem_after_mb']:.2f} MB freed)")

    # --- Query ---
    q_win = "pd.query" if query_stats['query_time'] < query_stats['mask_time'] else "masking"
    print(f"\n  2. QUERY vs MASKING")
    print(f"     Masking  : {query_stats['mask_time']:.4f}s")
    print(f"     pd.query : {query_stats['query_time']:.4f}s")
    print(f"     Winner   : {q_win}")

    # --- Eval ---
    e_win = "pd.eval" if eval_stats['eval_time'] < eval_stats['standard_time'] else "standard"
    print(f"\n  3. EVAL vs STANDARD MATH")
    print(f"     Standard : {eval_stats['standard_time']:.4f}s")
    print(f"     pd.eval  : {eval_stats['eval_time']:.4f}s")
    print(f"     Winner   : {e_win}")

    # --- Vectorization ---
    print(f"\n  4. APPLY vs VECTORIZATION (100k rows)")
    print(f"     apply()  : {vec_stats['apply_time']:.4f}s")
    print(f"     np.where : {vec_stats['vec_time']:.4f}s  "
          f"({vec_stats['apply_time'] / vec_stats['vec_time']:.1f}x faster)")
    print(f"     NumPy    : {vec_stats['numpy_time']:.4f}s  "
          f"({vec_stats['apply_time'] / vec_stats['numpy_time']:.1f}x faster)")

    # --- Chunked I/O ---
    match = "✓ Matches" if chunk_stats['score_sum_match'] else "✗ Mismatch"
    print(f"\n  5. CHUNKED vs FULL CSV LOAD (200k rows)")
    print(f"     Full load : {chunk_stats['full_load_time']:.4f}s")
    print(f"     Chunked   : {chunk_stats['chunked_load_time']:.4f}s")
    print(f"     Aggregate : {match}")

    # --- GroupBy ---
    print(f"\n  6. GROUPBY STRATEGY (500k rows)")
    print(f"     apply()     : {groupby_stats['groupby_apply_time']:.4f}s")
    print(f"     transform() : {groupby_stats['groupby_transform_time']:.4f}s")
    print(f"     agg()       : {groupby_stats['groupby_agg_time']:.4f}s  ← Recommended")

    # --- Key Takeaways ---
    print(f"\n  KEY TAKEAWAYS:")
    print("    • Use category dtype for low-cardinality strings  → saves 60–80% RAM")
    print("    • Prefer pd.eval / pd.query on large DataFrames   → leverages numexpr")
    print("    • Replace apply(axis=1) with np.where / .values   → 10–100x speedup")
    print("    • Use groupby.agg() over groupby.apply()          → vectorized path")
    print("    • Chunked I/O is memory-safe for files > RAM      → ETL best practice")
    print(f"\n{sep}")


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  DAY 31 — HIGH-PERFORMANCE DATA PROCESSING WITH PANDAS")
    print("  100-Day Data Science Challenge")
    print("=" * 70)

    # 1. Generate raw dataset (5M rows)
    df_raw = generate_large_dataset(rows=5_000_000)
    print("\n--- Data Sample (first 5 rows) ---")
    print(df_raw.head())

    # 2. Optimize dtypes and show comparison
    df_opt, mem_stats = optimize_dtypes(df_raw)
    show_dtype_comparison(df_raw, df_opt)

    # 3. Run all benchmarks on the optimized DataFrame
    query_stats   = benchmark_query_vs_masking(df_opt)
    eval_stats    = benchmark_eval_vs_standard(df_opt)
    vec_stats     = benchmark_vectorization_vs_apply(df_opt)
    chunk_stats   = benchmark_chunked_processing(df_opt)
    groupby_stats = benchmark_groupby_operations(df_opt)

    # 4. Generate consolidated performance report
    generate_performance_report(
        mem_stats     = mem_stats,
        query_stats   = query_stats,
        eval_stats    = eval_stats,
        vec_stats     = vec_stats,
        chunk_stats   = chunk_stats,
        groupby_stats = groupby_stats,
    )
