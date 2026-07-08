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


# -----------------------------------
# ARRAY CREATION
# -----------------------------------

def array_creation():

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


# -----------------------------------
# ARRAY ATTRIBUTES
# -----------------------------------

def array_attributes():

    print("\n" + "=" * 50)
    print("2. ARRAY ATTRIBUTES")
    print("=" * 50)

    arr = np.array([[1, 2, 3],
                    [4, 5, 6]])

    print(arr)

    print("\nShape :", arr.shape)
    print("Dimensions :", arr.ndim)
    print("Size :", arr.size)
    print("Data Type :", arr.dtype)


# -----------------------------------
# INDEXING
# -----------------------------------

def indexing():

    print("\n" + "=" * 50)
    print("3. INDEXING & SLICING")
    print("=" * 50)

    arr = np.array([5, 10, 15, 20, 25, 30])

    print("Array:", arr)

    print("First:", arr[0])

    print("Last:", arr[-1])

    print("Slice:", arr[1:5])


# -----------------------------------
# MATH OPERATIONS
# -----------------------------------

def math_operations():

    print("\n" + "=" * 50)
    print("4. MATHEMATICAL OPERATIONS")
    print("=" * 50)

    a = np.array([10, 20, 30])

    b = np.array([2, 4, 5])

    print("Addition :", a + b)

    print("Subtraction :", a - b)

    print("Multiplication :", a * b)

    print("Division :", a / b)


# -----------------------------------
# STATISTICS
# -----------------------------------

def statistics():

    print("\n" + "=" * 50)
    print("5. STATISTICAL FUNCTIONS")
    print("=" * 50)

    marks = np.array([78, 82, 91, 65, 88, 95])

    print("Marks:", marks)

    print("Mean :", np.mean(marks))

    print("Median :", np.median(marks))

    print("Maximum :", np.max(marks))

    print("Minimum :", np.min(marks))

    print("Standard Deviation :", np.std(marks))


# -----------------------------------
# RESHAPE
# -----------------------------------

def reshape_demo():

    print("\n" + "=" * 50)
    print("6. RESHAPING ARRAYS")
    print("=" * 50)

    arr = np.arange(1, 13)

    matrix = arr.reshape(3, 4)

    print(matrix)


# -----------------------------------
# RANDOM
# -----------------------------------

def random_demo():

    print("\n" + "=" * 50)
    print("7. RANDOM NUMBERS")
    print("=" * 50)

    random_marks = np.random.randint(40, 101, 10)

    print(random_marks)


# -----------------------------------
# MINI PROJECT
# -----------------------------------

def student_analysis():

    print("\n" + "=" * 50)
    print("8. MINI PROJECT - STUDENT MARKS ANALYSIS")
    print("=" * 50)

    students = np.array([
        [78, 80, 75],
        [92, 88, 95],
        [65, 70, 60],
        [85, 90, 84],
        [72, 75, 70]
    ])

    average = np.mean(students, axis=1)

    topper = np.argmax(average)

    print("\nStudent Marks\n")

    print(students)

    print("\nAverage Marks")

    print(average)

    print("\nTopper Student Index:", topper)

    print("Highest Average:", average[topper])

    print("\nSubject Wise Average")

    print(np.mean(students, axis=0))


# -----------------------------------
# MAIN
# -----------------------------------

def main():

    array_creation()

    array_attributes()

    indexing()

    math_operations()

    statistics()

    reshape_demo()

    random_demo()

    student_analysis()

    print("\nDay 11 Completed Successfully!")


if __name__ == "__main__":
    main()