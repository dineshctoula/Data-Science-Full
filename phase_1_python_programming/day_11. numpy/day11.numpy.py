

#!/usr/bin/env python3
"""
====================================================
100 Days of Data Science - Day 11
Topic: NumPy Fundamentals

Topics Covered
--------------
1. Creating Arrays
2. Array Attributes
3. Indexing and Slicing
4. Mathematical Operations
5. Statistical Functions
6. Reshaping Arrays
7. Random Numbers
8. Practical Exercise

Author: Dinesh Sitoula
====================================================
"""

import numpy as np


# ==================================================
# 1. ARRAY CREATION
# ==================================================

def array_creation():
    """Demonstrate different ways to create NumPy arrays."""

    print("=" * 50)
    print("1. ARRAY CREATION")
    print("=" * 50)

    arr1 = np.array([10, 20, 30, 40, 50])
    arr2 = np.zeros((2, 3))
    arr3 = np.ones((3, 3))
    arr4 = np.arange(1, 11)
    arr5 = np.linspace(0, 100, 5)

    print("Array:", arr1)
    print("\nZeros:\n", arr2)
    print("\nOnes:\n", arr3)
    print("\nRange:", arr4)
    print("\nLinspace:", arr5)


# ==================================================
# 2. ARRAY ATTRIBUTES
# ==================================================

def array_attributes():
    """Demonstrate important NumPy array attributes."""

    print("\n" + "=" * 50)
    print("2. ARRAY ATTRIBUTES")
    print("=" * 50)

    arr = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    print("Array:\n", arr)

    print("\nShape:", arr.shape)
    print("Dimensions:", arr.ndim)
    print("Size:", arr.size)
    print("Data Type:", arr.dtype)


# ==================================================
# 3. INDEXING AND SLICING
# ==================================================

def indexing_and_slicing():
    """Demonstrate indexing and slicing operations."""

    print("\n" + "=" * 50)
    print("3. INDEXING & SLICING")
    print("=" * 50)

    arr = np.array([5, 10, 15, 20, 25, 30])

    print("Array:", arr)
    print("First Element:", arr[0])
    print("Last Element:", arr[-1])
    print("Slice:", arr[1:5])


# ==================================================
# 4. MATHEMATICAL OPERATIONS
# ==================================================

def mathematical_operations():
    """Demonstrate element-wise mathematical operations."""

    print("\n" + "=" * 50)
    print("4. MATHEMATICAL OPERATIONS")
    print("=" * 50)

    a = np.array([10, 20, 30])
    b = np.array([2, 4, 5])

    print("Array A:", a)
    print("Array B:", b)

    print("\nAddition:", a + b)
    print("Subtraction:", a - b)
    print("Multiplication:", a * b)
    print("Division:", a / b)


# ==================================================
# 5. STATISTICAL FUNCTIONS
# ==================================================

def statistical_functions():
    """Demonstrate basic statistical functions."""

    print("\n" + "=" * 50)
    print("5. STATISTICAL FUNCTIONS")
    print("=" * 50)

    marks = np.array([78, 82, 91, 65, 88, 95])

    print("Marks:", marks)

    print("\nMean:", np.mean(marks))
    print("Median:", np.median(marks))
    print("Maximum:", np.max(marks))
    print("Minimum:", np.min(marks))
    print("Standard Deviation:", np.std(marks))


# ==================================================
# 6. RESHAPING ARRAYS
# ==================================================

def reshape_demo():
    """Demonstrate how to reshape a NumPy array."""

    print("\n" + "=" * 50)
    print("6. RESHAPING ARRAYS")
    print("=" * 50)

    arr = np.arange(1, 13)
    matrix = arr.reshape(3, 4)

    print("Original Array:", arr)
    print("\nReshaped Array:\n", matrix)


# ==================================================
# 7. RANDOM NUMBERS
# ==================================================

def random_demo():
    """Generate random student marks."""

    print("\n" + "=" * 50)
    print("7. RANDOM NUMBERS")
    print("=" * 50)

    random_marks = np.random.randint(40, 101, 10)

    print("Random Marks:", random_marks)