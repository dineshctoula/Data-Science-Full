#!/usr/bin/env python3
"""
====================================================
100 Days of Data Science - Day 17
Topic: Introduction to Pandas: Series & DataFrames

Topics Covered:
1. Creating Pandas Series (from lists, dicts, and NumPy arrays)
2. Indexing & Label Alignment in Series
3. Creating Pandas DataFrames (from dicts of lists & lists of dicts)
4. Inspecting DataFrames (head, tail, info, describe, shape, dtypes)
5. Practical Exercise: Student Profiles Dataset

Author: Dinesh Sitoula
====================================================
"""

import numpy as np
import pandas as pd

# -----------------------------------
# 1. PANDAS SERIES CREATION
# -----------------------------------
def demonstrate_series():
    print("=" * 60)
    print("1. PANDAS SERIES CREATION")
    print("=" * 60)

    # A Series is a 1D labeled array capable of holding any data type.
    
    # 1.1 Creating Series from a List
    data_list = [10, 20, 30, 40]
    series_from_list = pd.Series(data_list, name="List_Series")
    print("Series from List (default integer index):\n", series_from_list)

    # 1.2 Creating Series with Custom Labels (Index)
    custom_labels = ['a', 'b', 'c', 'd']
    series_with_index = pd.Series(data_list, index=custom_labels, name="Custom_Index_Series")
    print("\nSeries with Custom Index:\n", series_with_index)

    # 1.3 Creating Series from a Dictionary
    data_dict = {'apple': 5, 'banana': 8, 'cherry': 15}
    series_from_dict = pd.Series(data_dict, name="Fruits")
    print("\nSeries from Dictionary (keys become index labels):\n", series_from_dict)

    # 1.4 Creating Series from a NumPy Array
    np_array = np.array([1.5, 2.5, 3.5])
    series_from_np = pd.Series(np_array, index=['x', 'y', 'z'], name="Float_Series")
    print("\nSeries from NumPy Array:\n", series_from_np)


# -----------------------------------
# 2. INDEXING & ALIGNMENT IN SERIES
# -----------------------------------
def demonstrate_indexing_alignment():
    print("\n" + "=" * 60)
    print("2. INDEXING & LABEL ALIGNMENT IN SERIES")
    print("=" * 60)

    s1 = pd.Series([1, 2, 3], index=['A', 'B', 'C'])
    s2 = pd.Series([10, 20, 30], index=['B', 'C', 'D'])

    print("Series s1:\n", s1)
    print("Series s2:\n", s2)

    # Indexing by label or position
    print("\ns1 value at index 'B' (label):", s1['B'])
    print("s1 value at index 1 (position):", s1.iloc[1])

    # Label alignment in arithmetic operations:
    # Pandas aligns indices automatically. Missing labels in either series result in NaN.
    summed_series = s1 + s2
    print("\nSum of s1 and s2 (Automatic label alignment):\n", summed_series)


# -----------------------------------
# 3. CREATING PANDAS DATAFRAMES
# -----------------------------------
def demonstrate_dataframe_creation():
    print("\n" + "=" * 60)
    print("3. CREATING PANDAS DATAFRAMES")
    print("=" * 60)

    # A DataFrame is a 2D labeled data structure with columns of potentially different types.

    # 3.1 Creating DataFrame from a Dictionary of Lists
    dict_data = {
        'Product': ['Laptop', 'Mouse', 'Keyboard'],
        'Price': [1200.00, 25.50, 75.00],
        'InStock': [True, True, False]
    }
    df_from_dict = pd.DataFrame(dict_data)
    print("DataFrame from Dictionary of Lists:\n", df_from_dict)

    # 3.2 Creating DataFrame from a List of Dictionaries
    list_of_dicts = [
        {'Name': 'Dinesh', 'Role': 'Data Scientist'},
        {'Name': 'Amit', 'Role': 'Data Engineer'},
        {'Name': 'Sofia', 'Role': 'ML Engineer'}
    ]
    df_from_list = pd.DataFrame(list_of_dicts)
    print("\nDataFrame from List of Dictionaries:\n", df_from_list)


# -----------------------------------
# 4. INSPECTING DATAFRAMES
# -----------------------------------
def demonstrate_dataframe_inspection():
    print("\n" + "=" * 60)
    print("4. INSPECTING DATAFRAMES")
    print("=" * 60)

    # Create a dummy DataFrame with multiple rows
    df = pd.DataFrame({
        'ID': range(101, 111),
        'Score': [85, 92, 78, 89, 95, 62, 74, 88, 91, 80],
        'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A']
    })

    print("DataFrame Shape (rows, columns):", df.shape)
    print("\nFirst 3 rows using .head(3):\n", df.head(3))
    print("\nLast 2 rows using .tail(2):\n", df.tail(2))
    print("\nColumn names:", df.columns.tolist())
    print("\nData Types of columns:\n", df.dtypes)
    
    print("\nSummary Statistics of numeric columns (.describe()):\n", df.describe())
    print("\nDataFrame structural details using .info() (printed to stderr/stdout):")
    df.info()


# -----------------------------------
# 5. PRACTICAL EXERCISE
# -----------------------------------
def practical_exercise():
    print("\n" + "=" * 60)
    print("5. PRACTICAL EXERCISE: STUDENT PROFILES DATA")
    print("=" * 60)
    print("Task: Create a 10-row student dataset, inspect statistics and types.")

    # 1. Define student data (10 records)
    student_data = {
        'Name': ['Dinesh', 'Elena', 'Karthik', 'Sophia', 'Kenji', 'Chloe', 'Raj', 'Yuki', 'Amina', 'Carlos'],
        'Age': [24, 22, 23, 21, 25, 22, 24, 23, 22, 26],
        'Grade': ['A', 'B', 'B+', 'A+', 'B-', 'A', 'C+', 'A', 'B', 'A-'],
        'City': ['Kathmandu', 'Moscow', 'Bangalore', 'London', 'Tokyo', 'Paris', 'Mumbai', 'Kyoto', 'Nairobi', 'Madrid'],
        'GPA': [3.85, 3.42, 3.56, 3.98, 3.10, 3.75, 2.95, 3.90, 3.50, 3.65]
    }

    # 2. Convert to Pandas DataFrame
    students_df = pd.DataFrame(student_data)
    print("\nCreated Students DataFrame:\n", students_df)

    # 3. Print the shape
    print("\nDataFrame Shape (Rows, Columns):", students_df.shape)

    # 4. Inspect columns and data types
    print("\nColumns:", students_df.columns.tolist())
    print("Data Types:\n", students_df.dtypes)

    # 5. View basic descriptive statistics
    print("\nSummary Statistics (Descriptive):\n", students_df.describe())
    
    # 6. Verify basic expectations
    assert students_df.shape == (10, 5), "DataFrame shape must be (10, 5)"
    assert abs(students_df['GPA'].mean() - 3.561) < 0.001, "Average GPA does not match expected value!"
    print("\nAll assertions passed successfully!")


def main():
    print("=== Day 17: Introduction to Pandas: Series & DataFrames ===")
    demonstrate_series()
    demonstrate_indexing_alignment()
    demonstrate_dataframe_creation()
    demonstrate_dataframe_inspection()
    practical_exercise()
    print("\nDay 17 Completed Successfully!")

if __name__ == "__main__":
    main()
