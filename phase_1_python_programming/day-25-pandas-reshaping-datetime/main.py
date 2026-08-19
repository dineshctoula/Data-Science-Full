#!/usr/bin/env python3
"""
================================================================================
100 Days of Data Science - Day 25
Topic: Advanced Pandas Reshaping, Pivoting, Melting & Time Series Analysis
================================================================================

This module covers:
1. Reshaping Data: Converting Wide Format to Long Format with `pd.melt()` and `pivot()`.
2. Hierarchical Reshaping: Multi-Index Stacking (`stack()`) and Unstacking (`unstack()`).
3. Datetime Operations: Parsing, DatetimeIndex, and temporal feature extraction via `.dt`.
4. Time Series Analysis: Resampling (`resample()`), Rolling Windows (`rolling()`), & Shifting (`shift()`).
5. Real-World Case Study: E-Commerce & Financial Time Series Analytics Pipeline.

Author: Dinesh Sitoula
================================================================================
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# 1. DATA RESHAPING: MELT AND PIVOT
# ------------------------------------------------------------------------------
def demonstrate_melt_and_pivot():
    """
    Demonstrates data reshaping between Wide and Long formats.

    Wide format: Data has separate columns for each measurement/time period.
    Long (Tidy) format: Each row represents a single observation; columns represent variables.
    """
    print("=" * 80)
    print("1. DATA RESHAPING: WIDE TO LONG (MELT) & LONG TO WIDE (PIVOT)")
    print("=" * 80)

    # Create a synthetic wide-format revenue dataset (e.g., quarterly revenue per store)
    wide_data = pd.DataFrame({
        'store_id': ['S01', 'S02', 'S03', 'S04'],
        'region': ['East', 'West', 'East', 'North'],
        'Q1_Revenue': [12000, 15000, 9500, 18000],
        'Q2_Revenue': [13500, 14800, 10200, 19200],
        'Q3_Revenue': [14100, 16200, 11000, 20100],
        'Q4_Revenue': [16000, 18500, 12500, 22000]
    })

    print("--- Original Wide Format DataFrame ---")
    print(wide_data)

    # --------------------------------------------------------------------------
    # Melt: Unpivot DataFrame from wide format to long format
    # id_vars: Columns to keep as identifier variables (not melted)
    # value_vars: Columns to unpivot (if None, uses all columns not in id_vars)
    # var_name: Name for the new column storing melted column names
    # value_name: Name for the new column storing melted values
    # --------------------------------------------------------------------------
    long_data = pd.melt(
        wide_data,
        id_vars=['store_id', 'region'],
        value_vars=['Q1_Revenue', 'Q2_Revenue', 'Q3_Revenue', 'Q4_Revenue'],
        var_name='quarter',
        value_name='revenue'
    )

    # Clean up quarter string (e.g., 'Q1_Revenue' -> 'Q1')
    long_data['quarter'] = long_data['quarter'].str.replace('_Revenue', '')

    print("\n--- Melted (Long/Tidy Format) DataFrame ---")
    print(long_data.head(8))

    # Verify rows count: 4 stores * 4 quarters = 16 long rows
    assert len(long_data) == 16, "Melted dataframe should have 16 rows!"

    # --------------------------------------------------------------------------
    # Pivot: Reshape long data back to wide format using index and columns
    # index: Column to use to make new frame's index
    # columns: Column to use to make new frame's columns
    # values: Column(s) to use for populating new frame's values
    # --------------------------------------------------------------------------
    pivoted_wide = long_data.pivot(
        index=['store_id', 'region'],
        columns='quarter',
        values='revenue'
    ).reset_index()

    # Rename columns to remove index name artifact
    pivoted_wide.columns.name = None

    print("\n--- Reshaped Back to Wide Format using .pivot() ---")
    print(pivoted_wide)

    # Verify exact numerical match with original totals
    assert (pivoted_wide['Q1'] == wide_data['Q1_Revenue']).all(), "Pivoted revenue matches original wide values!"

    print("\n[✓] Melt and Pivot operations completed successfully.")


if __name__ == "__main__":
    demonstrate_melt_and_pivot()
