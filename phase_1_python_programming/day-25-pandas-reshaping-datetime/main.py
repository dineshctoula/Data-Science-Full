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


# ------------------------------------------------------------------------------
# 3. DATETIME OPERATIONS & TEMPORAL FEATURE EXTRACTION
# ------------------------------------------------------------------------------
def demonstrate_datetime_parsing_and_extraction():
    """
    Demonstrates string-to-datetime conversion (`pd.to_datetime`),
    DatetimeIndex creation, and feature extraction via the `.dt` accessor.
    """
    print("\n" + "=" * 80)
    print("3. DATETIME PARSING, INDEXING, AND TEMPORAL FEATURE EXTRACTION")
    print("=" * 80)

    # Synthetic transactional timestamp string data with varying formats
    raw_transactions = pd.DataFrame({
        'transaction_id': [101, 102, 103, 104, 105, 106],
        'timestamp_str': [
            '2026-01-15 08:30:00',
            '2026-01-31 23:45:00',
            '2026-02-14 12:15:30',
            '2026-02-28 18:00:00',
            '2026-03-15 09:20:10',
            '2026-03-31 21:10:00'
        ],
        'amount': [250.50, 1200.00, 450.75, 890.00, 310.20, 1500.00]
    })

    print("--- Original Raw Transactions DataFrame ---")
    print(raw_transactions)

    # --------------------------------------------------------------------------
    # pd.to_datetime(): Convert string column to Pandas datetime64[ns] dtype
    # --------------------------------------------------------------------------
    raw_transactions['datetime'] = pd.to_datetime(raw_transactions['timestamp_str'])

    # --------------------------------------------------------------------------
    # Temporal Feature Extraction using .dt Accessor
    # --------------------------------------------------------------------------
    df = raw_transactions.copy()
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['month_name'] = df['datetime'].dt.month_name()
    df['day'] = df['datetime'].dt.day
    df['day_name'] = df['datetime'].dt.day_name()
    df['quarter'] = df['datetime'].dt.quarter
    df['hour'] = df['datetime'].dt.hour
    df['is_month_end'] = df['datetime'].dt.is_month_end

    print("\n--- Extracted Temporal Features DataFrame ---")
    print(df[['transaction_id', 'datetime', 'month_name', 'day_name', 'quarter', 'is_month_end']])

    # Verify extracted features
    assert df.loc[1, 'is_month_end'] == True, "Jan 31 transaction should register is_month_end=True!"
    assert df.loc[3, 'day_name'] == 'Saturday', "Feb 28 2026 is a Saturday!"

    # --------------------------------------------------------------------------
    # DatetimeIndex Slicing and Filtering
    # --------------------------------------------------------------------------
    df_indexed = df.set_index('datetime').sort_index()

    # Slice transactions occurring specifically in February 2026
    feb_transactions = df_indexed.loc['2026-02-01':'2026-02-28']
    print("\n--- Filtered Transactions for February 2026 using DatetimeIndex Slicing ---")
    print(feb_transactions[['transaction_id', 'amount', 'day_name']])

    assert len(feb_transactions) == 2, "Should return exactly 2 transactions in February 2026!"

    print("\n[✓] Datetime parsing and feature extraction completed successfully.")


# ------------------------------------------------------------------------------
# 4. TIME SERIES RESAMPLING, ROLLING WINDOWS, AND SHIFT OPERATIONS
# ------------------------------------------------------------------------------
def demonstrate_time_series_resampling_and_rolling():
    """
    Demonstrates time series frequency conversion (`resample()`),
    moving average calculation (`rolling()`), and lag analysis (`shift()`, `pct_change()`).
    """
    print("\n" + "=" * 80)
    print("4. TIME SERIES RESAMPLING, ROLLING WINDOWS, AND SHIFT OPERATIONS")
    print("=" * 80)

    # Generate 90 days of synthetic daily revenue data starting Jan 1, 2026
    np.random.seed(42)
    date_range = pd.date_range(start='2026-01-01', periods=90, freq='D')
    base_revenue = 1000 + np.cumsum(np.random.normal(loc=15, scale=50, size=90))

    daily_series = pd.DataFrame({
        'revenue': np.round(base_revenue, 2)
    }, index=date_range)
    daily_series.index.name = 'date'

    print("--- First 5 Days of Daily Revenue Data ---")
    print(daily_series.head())

    # --------------------------------------------------------------------------
    # Resampling: Frequency conversion and rollup
    # 'W': Weekly frequency rollup
    # 'ME': Month-end frequency rollup
    # --------------------------------------------------------------------------
    weekly_revenue = daily_series.resample('W').agg({
        'revenue': ['sum', 'mean', 'count']
    })
    # Flatten MultiIndex columns created by agg
    weekly_revenue.columns = ['total_revenue', 'avg_daily_revenue', 'days_count']

    print("\n--- Weekly Resampled Summary (First 4 Weeks) ---")
    print(weekly_revenue.head(4))

    # --------------------------------------------------------------------------
    # Rolling Windows: Moving Statistics (7-Day Moving Average & Volatility)
    # window=7: 7-period trailing window
    # min_periods=1: Allow initial window partial calculation
    # --------------------------------------------------------------------------
    daily_series['sma_7d'] = daily_series['revenue'].rolling(window=7, min_periods=1).mean()
    daily_series['std_7d'] = daily_series['revenue'].rolling(window=7, min_periods=1).std().fillna(0)

    # --------------------------------------------------------------------------
    # Lag Analysis: Shifting & Daily Growth Rate (%)
    # shift(1): Lag value by 1 day (previous day's revenue)
    # pct_change(): (Current - Previous) / Previous
    # --------------------------------------------------------------------------
    daily_series['prev_day_revenue'] = daily_series['revenue'].shift(1)
    daily_series['daily_growth_pct'] = daily_series['revenue'].pct_change() * 100

    print("\n--- Daily Series with Rolling Windows & Shift Indicators (Days 6 to 12) ---")
    print(daily_series.iloc[5:12])

    # Assertions for mathematical consistency
    day7_rev = daily_series.iloc[6]['revenue']
    day6_rev = daily_series.iloc[6]['prev_day_revenue']
    expected_pct = ((day7_rev - day6_rev) / day6_rev) * 100
    assert np.isclose(daily_series.iloc[6]['daily_growth_pct'], expected_pct), "Daily growth % matches exact formula!"

    print("\n[✓] Time series resampling, rolling windows, and shifting completed successfully.")


if __name__ == "__main__":
    demonstrate_melt_and_pivot()
    demonstrate_stack_and_unstack()
    demonstrate_datetime_parsing_and_extraction()
    demonstrate_time_series_resampling_and_rolling()

