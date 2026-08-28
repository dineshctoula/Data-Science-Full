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

if __name__ == "__main__":
    df = generate_large_dataset(1_000_000)
    print("\n--- Data Sample ---")
    print(df.head())
