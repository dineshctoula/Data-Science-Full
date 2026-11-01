#!/usr/bin/env python3
"""
==========================================
100 Days of Data Science
Day 12 - NumPy Arrays
(Indexing, Slicing, Reshaping & Iteration)
==========================================

Topics:
1. Indexing
2. Slicing
3. Multi-dimensional Arrays
4. Reshaping
5. Flattening
6. Iteration

Author: Dinesh Sitoula
"""

import numpy as np

print("=" * 60)
print("DAY 12 - NUMPY ARRAY OPERATIONS")
print("=" * 60)

# ----------------------------------------------------
# 1. Creating Arrays
# ----------------------------------------------------

print("\n1. Creating Arrays")

arr = np.array([10, 20, 30, 40, 50])

print(arr)

# ----------------------------------------------------
# 2. Indexing
# ----------------------------------------------------

print("\n2. Indexing")

print("First Element :", arr[0])
print("Third Element :", arr[2])
print("Last Element :", arr[-1])
print("Second Last :", arr[-2])

# ----------------------------------------------------
# 3. Slicing
# ----------------------------------------------------

print("\n3. Slicing")

print("First 3 Elements :", arr[:3])
print("Last 2 Elements :", arr[-2:])
print("Middle Elements :", arr[1:4])
print("Every Second Element :", arr[::2])

# ----------------------------------------------------
# 4. Two-Dimensional Arrays
# ----------------------------------------------------

print("\n4. Two-Dimensional Arrays")

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix)

print("\nElement at Row 2 Column 3:", matrix[1, 2])
print("First Row:", matrix[0])
print("Second Column:", matrix[:, 1])
print("Last Row:", matrix[-1])

# ----------------------------------------------------
# 5. Reshape Arrays
# ----------------------------------------------------

print("\n5. Reshape")

numbers = np.arange(1, 13)

print("Original:")
print(numbers)

reshaped = numbers.reshape(3, 4)

print("\nReshaped into 3 x 4")
print(reshaped)

# ----------------------------------------------------
# 6. Flatten Arrays
# ----------------------------------------------------

print("\n6. Flatten")

flat = reshaped.flatten()

print(flat)

# ----------------------------------------------------
# 7. Iterating Arrays
# ----------------------------------------------------

print("\n7. Iterating 1D Array")

for item in arr:
    print(item)

print("\nIterating 2D Array")

for row in matrix:
    print(row)

print("\nEach Individual Element")

for value in np.nditer(matrix):
    print(value)

# ----------------------------------------------------
# 8. Shape Information
# ----------------------------------------------------

print("\n8. Shape Information")

print("Shape :", reshaped.shape)
print("Dimensions :", reshaped.ndim)
print("Size :", reshaped.size)
print("Data Type :", reshaped.dtype)

# ----------------------------------------------------
# 9. Mini Exercise
# ----------------------------------------------------

print("\n9. Mini Exercise")

marks = np.array([45, 78, 91, 88, 65])

print("Marks:", marks)

print("Highest:", marks.max())
print("Lowest:", marks.min())
print("Average:", marks.mean())

print("Students scoring above 70:")

for mark in marks:
    if mark > 70:
        print(mark)

# ----------------------------------------------------
# 10. Challenge
# ----------------------------------------------------

print("\n10. Challenge")

data = np.arange(1, 17)

square = data.reshape(4, 4)

print(square)

print("\nDiagonal Elements:")

for i in range(4):
    print(square[i][i])

print("\nProgram Finished Successfully!")