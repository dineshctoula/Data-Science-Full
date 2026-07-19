#!/usr/bin/env python3
"""
=========================================
100 Days of Data Science - Day 19
Topic: Creating and Inspecting DataFrames
=========================================

Topics Covered
--------------
1. Creating DataFrames
2. Selecting Columns
3. Basic Information
4. DataFrame Shape
5. Data Types
"""

import pandas as pd

print("=" * 60)
print("DAY 19 - CREATING DATAFRAMES")
print("=" * 60)

# ----------------------------------------
# Create DataFrame using Dictionary
# ----------------------------------------

students = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [20, 22, 21, 23],
    "City": ["New York", "London", "Paris", "Tokyo"],
    "Marks": [85, 90, 78, 95]
}

df = pd.DataFrame(students)

print("\nComplete DataFrame")
print(df)

# ----------------------------------------
# Display first rows
# ----------------------------------------

print("\nFirst 3 Rows")
print(df.head(3))

# ----------------------------------------
# Display last rows
# ----------------------------------------

print("\nLast 2 Rows")
print(df.tail(2))

# ----------------------------------------
# Shape
# ----------------------------------------

print("\nShape of DataFrame")
print(df.shape)

# ----------------------------------------
# Columns
# ----------------------------------------

print("\nColumn Names")
print(df.columns)

# ----------------------------------------
# Data Types
# ----------------------------------------

print("\nData Types")
print(df.dtypes)

# ----------------------------------------
# General Information
# ----------------------------------------

print("\nInformation")
df.info()

print("\nCommit 1 Complete!")