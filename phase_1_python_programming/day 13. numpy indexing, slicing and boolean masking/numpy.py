#!/usr/bin/env python3
"""
===========================================
100 Days of Data Science
Day 13 - NumPy Indexing and Slicing
===========================================

Topics:
1. Indexing
2. Slicing
3. Reshape
4. Copy vs View
5. Boolean Masking
6. Fancy Indexing
"""

import numpy as np

print("=" * 60)
print("DAY 13 - NUMPY INDEXING & SLICING")
print("=" * 60)

# ---------------------------------------------------
# 1. Indexing in 1D Array
# ---------------------------------------------------

print("\n1. Indexing in 1D Array")

arr = np.array([10, 20, 30, 40, 50])

print("Array:", arr)

print("First element:", arr[0])
print("Second element:", arr[1])
print("Last element:", arr[-1])

# ---------------------------------------------------
# 2. Indexing in 2D Array
# ---------------------------------------------------

print("\n2. Indexing in 2D Array")

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix)

print("First row:", matrix[0])
print("Second row:", matrix[1])

print("Element at row 2 column 3:", matrix[1, 2])

print("Bottom-right element:", matrix[2, 2])

# ---------------------------------------------------
# 3. Slicing
# ---------------------------------------------------

print("\n3. Array Slicing")

numbers = np.array([10,20,30,40,50,60,70,80])

print(numbers)

print("First 3 elements:", numbers[:3])

print("Last 3 elements:", numbers[-3:])

print("Middle elements:", numbers[2:6])

print("Every second element:", numbers[::2])

print("Reverse array:", numbers[::-1])

# ---------------------------------------------------
# 4. Slicing in 2D Arrays
# ---------------------------------------------------

print("\n4. 2D Slicing")

print(matrix)

print("First row:")
print(matrix[0, :])

print("Second column:")
print(matrix[:, 1])

print("Top-left 2x2 matrix:")
print(matrix[:2, :2])

# ---------------------------------------------------
# 5. Reshape
# ---------------------------------------------------

print("\n5. Reshape")

a = np.arange(12)

print("Original:")
print(a)

reshaped = a.reshape(3,4)

print("Reshaped (3x4):")
print(reshaped)

# ---------------------------------------------------
# 6. Copy vs View
# ---------------------------------------------------

print("\n6. Copy vs View")

original = np.array([1,2,3,4])

view = original.view()

copy = original.copy()

original[0] = 100

print("Original:", original)

print("View:", view)

print("Copy:", copy)

# ---------------------------------------------------
# 7. Boolean Masking
# ---------------------------------------------------

print("\n7. Boolean Masking")

marks = np.array([45,67,82,91,39,55,76])

print("Marks:", marks)

passed = marks >= 50

print("Boolean Mask:")
print(passed)

print("Passed Students:")
print(marks[passed])

print("Students scoring above 80:")
print(marks[marks > 80])

# ---------------------------------------------------
# 8. Fancy Indexing
# ---------------------------------------------------

print("\n8. Fancy Indexing")

data = np.array([100,200,300,400,500])

print(data)

print("Selecting index 0,2,4")

print(data[[0,2,4]])

# ---------------------------------------------------
# 9. Updating Values
# ---------------------------------------------------

print("\n9. Updating Values")

salary = np.array([25000,30000,28000,35000])

print("Original:", salary)

salary[salary < 30000] += 5000

print("Updated:", salary)

# ---------------------------------------------------
# 10. Real-Life Example
# ---------------------------------------------------

print("\n10. Real-Life Example")

ages = np.array([12,18,21,15,30,45,16])

adults = ages >= 18

print("Ages:", ages)

print("Adults:", ages[adults])

print("Min Age:", ages.min())

print("Max Age:", ages.max())

print("Average Age:", ages.mean())

print("=" * 60)
print("Day 13 Completed Successfully!")
print("=" * 60)