#!/usr/bin/env python3
"""
====================================================
100 Days of Data Science - Day 18
Topic: Pandas Basics - Data Manipulation

Topics Covered:
1. Selecting Data (loc and iloc)
2. Filtering Data (Boolean Indexing)
3. Handling Missing Values
4. Adding and Removing Columns

Author: Dinesh Sitoula
====================================================
"""

import numpy as np
import pandas as pd

def get_sample_dataframe():
    """Create a sample DataFrame for manipulation."""
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'Age': [25, np.nan, 35, 40, 22, 33],
        'Department': ['HR', 'IT', 'Finance', 'IT', 'Marketing', 'Finance'],
        'Salary': [50000, 60000, 75000, 80000, 45000, np.nan]
    }
    return pd.DataFrame(data)

# -----------------------------------
# 1. SELECTING DATA
# -----------------------------------
def demonstrate_selection():
    """Demonstrate data selection using .loc and .iloc"""
    print("=" * 60)
    print("1. SELECTING DATA (loc vs iloc)")
    print("=" * 60)

    df = get_sample_dataframe()
    print("Original DataFrame:\n", df)

    # .loc is label-based selection
    print("\nSelect 'Name' and 'Salary' for first 3 rows using .loc:")
    print(df.loc[0:2, ['Name', 'Salary']])

    # .iloc is integer-position based selection
    print("\nSelect first 3 rows and first 2 columns using .iloc:")
    print(df.iloc[0:3, 0:2])

def main():
    print("=== Day 18: Pandas Basics - Data Manipulation ===")
    demonstrate_selection()

if __name__ == "__main__":
    main()