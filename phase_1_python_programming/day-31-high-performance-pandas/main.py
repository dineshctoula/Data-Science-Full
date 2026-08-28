"""
===============================================================================
DAY 31: HIGH-PERFORMANCE DATA PROCESSING WITH PANDAS
===============================================================================
Author: Dinesh (100-Day Data Science Challenge)
Phase: Phase 2 - Data Manipulation & Visualization
Topic: Performance Optimization, Eval, Query, Vectorization, Chunking
===============================================================================
"""

import os
import time
import numpy as np
import pandas as pd

def generate_large_dataset(rows: int = 1_000_000) -> pd.DataFrame:
    """
    Generates a large synthetic dataset to benchmark pandas performance.
    """
    print(f"[INFO] Generating large dataset with {rows:,} rows...")
    start_time = time.time()
    
    np.random.seed(42)
    
    data = {
        'id': np.arange(rows),
        'value_a': np.random.normal(100, 25, rows),
        'value_b': np.random.normal(50, 15, rows),
        'category': np.random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'], rows),
        'is_active': np.random.choice([True, False], rows)
    }
    
    df = pd.DataFrame(data)
    
    # Optimize memory usage with categorical data type
    df['category'] = df['category'].astype('category')
    
    elapsed_time = time.time() - start_time
    print(f"[SUCCESS] Dataset generated in {elapsed_time:.2f} seconds.")
    print(f"[INFO] Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    return df

def test_query_performance(df: pd.DataFrame) -> None:
    """
    Benchmarks standard boolean masking vs pandas.query().
    """
    print("\n[INFO] Benchmarking Boolean Masking vs pd.query()...")
    
    # 1. Standard Boolean Masking
    start_time = time.time()
    result_mask = df[(df['value_a'] > 120) & (df['value_b'] < 40) & (df['is_active'] == True)]
    mask_time = time.time() - start_time
    print(f"  Standard Masking Time: {mask_time:.4f} seconds (Rows: {len(result_mask)})")
    
    # 2. Pandas Query
    start_time = time.time()
    result_query = df.query("value_a > 120 and value_b < 40 and is_active == True")
    query_time = time.time() - start_time
    print(f"  Pandas Query Time:     {query_time:.4f} seconds (Rows: {len(result_query)})")
    print(f"  Speedup Factor:        {mask_time / query_time:.2f}x")

def test_eval_performance(df: pd.DataFrame) -> None:
    """
    Benchmarks standard arithmetic operations vs pandas.eval().
    """
    print("\n[INFO] Benchmarking Standard Math vs pd.eval()...")
    
    # 1. Standard Arithmetic
    start_time = time.time()
    df['result_standard'] = df['value_a'] * 2 + df['value_b'] / 3 - 50
    standard_time = time.time() - start_time
    print(f"  Standard Math Time: {standard_time:.4f} seconds")
    
    # 2. Pandas Eval
    start_time = time.time()
    df.eval("result_eval = value_a * 2 + value_b / 3 - 50", inplace=True)
    eval_time = time.time() - start_time
    print(f"  Pandas Eval Time:   {eval_time:.4f} seconds")
    print(f"  Speedup Factor:     {standard_time / eval_time:.2f}x")

if __name__ == "__main__":
    df = generate_large_dataset(5_000_000) # Increased to 5 million for better benchmarking
    print("\n--- Data Sample ---")
    print(df.head())
    
    test_query_performance(df)
    test_eval_performance(df)

