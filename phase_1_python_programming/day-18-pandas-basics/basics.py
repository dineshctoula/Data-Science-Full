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

# -----------------------------------
# 2. FILTERING DATA
# -----------------------------------
def demonstrate_filtering():
    """Demonstrate boolean indexing for data filtering"""
    print("\n" + "=" * 60)
    print("2. FILTERING DATA (Boolean Indexing)")
    print("=" * 60)

    df = get_sample_dataframe()

    # Filter employees older than 30
    older_than_30 = df[df['Age'] > 30]
    print("Employees older than 30:\n", older_than_30)

    # Multiple conditions: IT department AND salary > 60000
    high_earning_it = df[(df['Department'] == 'IT') & (df['Salary'] > 60000)]
    print("\nHigh earning IT employees:\n", high_earning_it)

# -----------------------------------
# 3. HANDLING MISSING VALUES
# -----------------------------------
def demonstrate_handling_missing():
    """Demonstrate handling of NaN/missing values"""
    print("\n" + "=" * 60)
    print("3. HANDLING MISSING VALUES")
    print("=" * 60)

    df = get_sample_dataframe()
    
    # Check for missing values
    print("Missing values per column:\n", df.isna().sum())

    # Drop rows with any missing values
    df_dropped = df.dropna()
    print("\nDataFrame after dropna():\n", df_dropped)

    # Fill missing values
    # We will fill missing Age with the mean age, and Salary with 0
    df_filled = df.copy()
    mean_age = df_filled['Age'].mean()
    df_filled['Age'] = df_filled['Age'].fillna(mean_age)
    df_filled['Salary'] = df_filled['Salary'].fillna(0)
    
    print("\nDataFrame after filling missing values:\n", df_filled)

# -----------------------------------
# 4. ADDING AND REMOVING COLUMNS
# -----------------------------------
def demonstrate_adding_removing_columns():
    """Demonstrate adding and removing columns from a DataFrame"""
    print("\n" + "=" * 60)
    print("4. ADDING AND REMOVING COLUMNS")
    print("=" * 60)

    df = get_sample_dataframe()
    
    # Adding a new column (e.g. bonus, 10% of salary)
    df['Bonus'] = df['Salary'] * 0.10
    print("DataFrame after adding 'Bonus' column:\n", df)

    # Removing a column
    # axis=1 specifies we are dropping a column, not a row
    df_dropped_col = df.drop('Department', axis=1)
    print("\nDataFrame after removing 'Department' column:\n", df_dropped_col)

def main():
    print("=== Day 18: Pandas Basics - Data Manipulation ===")
    demonstrate_selection()
    demonstrate_filtering()
    demonstrate_handling_missing()
    demonstrate_adding_removing_columns()

if __name__ == "__main__":
    main()