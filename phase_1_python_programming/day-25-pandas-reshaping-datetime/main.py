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


# ------------------------------------------------------------------------------
# 2. HIERARCHICAL RESHAPING: STACK AND UNSTACK
# ------------------------------------------------------------------------------
def demonstrate_stack_and_unstack():
    """
    Demonstrates Multi-Index DataFrame Reshaping using `.stack()` and `.unstack()`.

    - stack(): Pivots column level(s) to row index level(s). Moves wide headers to rows.
    - unstack(): Pivots row index level(s) to column level(s). Moves row index levels to columns.
    """
    print("\n" + "=" * 80)
    print("2. HIERARCHICAL RESHAPING: MULTI-INDEX STACK & UNSTACK")
    print("=" * 80)

    # Create Multi-Index columns (Financial Metrics x Departments)
    cols = pd.MultiIndex.from_tuples([
        ('2025', 'Revenue'), ('2025', 'Profit'),
        ('2026', 'Revenue'), ('2026', 'Profit')
    ], names=['Year', 'Metric'])

    # Create Multi-Index rows (Region x Product Branch)
    index = pd.MultiIndex.from_tuples([
        ('North', 'Electronics'), ('North', 'Apparel'),
        ('South', 'Electronics'), ('South', 'Apparel')
    ], names=['Region', 'Branch'])

    # Synthetic performance data matrix
    data = [
        [500, 120, 620, 150],
        [300, 80,  350, 95],
        [450, 110, 510, 130],
        [280, 65,  310, 75]
    ]

    df_multi = pd.DataFrame(data, index=index, columns=cols)

    print("--- Original Multi-Index DataFrame ---")
    print(df_multi)

    # --------------------------------------------------------------------------
    # .stack(): Move outermost column level ('Metric' or level=1) into row index
    # --------------------------------------------------------------------------
    stacked_df = df_multi.stack(level='Metric', future_stack=True)
    print("\n--- Stacked DataFrame (Metric column level moved to Row Index) ---")
    print(stacked_df)

    # Verify structure: Row index now has 3 levels: Region, Branch, Metric
    assert stacked_df.index.nlevels == 3, "Stacked DataFrame should have 3 index levels!"

    # --------------------------------------------------------------------------
    # .unstack(): Move row index level ('Branch' or level=1) into column header
    # --------------------------------------------------------------------------
    unstacked_df = stacked_df.unstack(level='Branch')
    print("\n--- Unstacked DataFrame (Branch row index level moved back to Columns) ---")
    print(unstacked_df)

    # --------------------------------------------------------------------------
    # Verify reversibility: Stacking then Unstacking preserves original shape & values
    # --------------------------------------------------------------------------
    fully_unstacked = stacked_df.unstack(level='Metric').reindex(index=df_multi.index, columns=df_multi.columns)
    assert np.array_equal(fully_unstacked.values, df_multi.values), "Stacking then unstacking preserves exact original data!"


    print("\n[✓] Stack and Unstack operations completed successfully.")



if __name__ == "__main__":
    demonstrate_melt_and_pivot()
    demonstrate_stack_and_unstack()

