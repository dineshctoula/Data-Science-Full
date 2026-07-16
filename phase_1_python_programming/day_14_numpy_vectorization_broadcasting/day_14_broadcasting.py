#!/usr/bin/env python3
"""
====================================================
100 Days of Data Science - Day 14
Topic: NumPy Vectorized Operations & Broadcasting

Topics Covered
--------------
1. Vectorized Operations (Element-wise arithmetic)
2. Broadcasting Rules (Operations on different shapes)
3. Universal Functions (ufuncs)
4. Practical Exercise (Matrix + Vector Broadcasting & Exp)

Author: Dinesh Sitoula
====================================================
"""

import numpy as np
import time

# -----------------------------------
# 1. VECTORIZED OPERATIONS
# -----------------------------------
def vectorized_operations():
    print("=" * 60)
    print("1. VECTORIZED OPERATIONS VS PYTHON LOOPS")
    print("=" * 60)

    # Creating large arrays to measure speed difference
    size = 1000000
    a = np.arange(size)
    b = np.arange(size)

    # 1. Python Loop approach
    start_time = time.time()
    c_loop = [a[i] + b[i] for i in range(size)]
    loop_duration = time.time() - start_time
    print(f"Python Loop Time: {loop_duration:.6f} seconds")

    # 2. NumPy Vectorized approach
    start_time = time.time()
    c_vectorized = a + b
    vectorized_duration = time.time() - start_time
    print(f"NumPy Vectorized Time: {vectorized_duration:.6f} seconds")
    print(f"Speedup: {loop_duration / vectorized_duration:.1f}x faster!")

    # Verify results are equal
    assert np.allclose(c_loop, c_vectorized), "Results must match!"

    # Element-wise operations
    x = np.array([1, 2, 3, 4])
    y = np.array([10, 20, 30, 40])
    print("\nElement-wise Arithmetic:")
    print("Addition (x + y):", x + y)
    print("Subtraction (y - x):", y - x)
    print("Multiplication (x * y):", x * y)
    print("Division (y / x):", y / x)
    print("Exponentiation (x ** 2):", x ** 2)


# -----------------------------------
# 2. BROADCASTING RULES
# -----------------------------------
def broadcasting_rules():
    print("\n" + "=" * 60)
    print("2. BROADCASTING RULES")
    print("=" * 60)

    # Rule 1: Array + Scalar
    a = np.array([1, 2, 3])
    print("1D Array:", a)
    print("Array + Scalar (5):", a + 5)

    # Rule 2: 2D Array + 1D Array
    # Shape of matrix is (2, 3), shape of row_vector is (3,)
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    row_vector = np.array([10, 20, 30])
    
    print("\nMatrix (2x3):\n", matrix)
    print("Row Vector (1x3):", row_vector)
    print("Broadcasting result (Matrix + Row Vector):\n", matrix + row_vector)

    # Rule 3: Stretching in both dimensions (Column + Row)
    # col_vec (3x1) + row_vec (1x3)
    col_vec = np.array([[1], [2], [3]])  # Shape (3, 1)
    row_vec = np.array([10, 20, 30])      # Shape (3,) -> Broadcasts to (1, 3)
    
    print("\nColumn Vector (3x1):\n", col_vec)
    print("Row Vector (1x3):", row_vec)
    print("Broadcasting result (Col + Row):\n", col_vec + row_vec)


# -----------------------------------
# 3. UNIVERSAL FUNCTIONS (UFUNCS)
# -----------------------------------
def universal_functions():
    print("\n" + "=" * 60)
    print("3. UNIVERSAL FUNCTIONS (UFUNCS)")
    print("=" * 60)

    arr = np.array([0, np.pi/2, np.pi])
    print("Angles (radians):", arr)
    print("Sine values:", np.round(np.sin(arr), 4))
    
    nums = np.array([1, 4, 9, 16])
    print("\nNumbers:", nums)
    print("Square Roots:", np.sqrt(nums))
    print("Natural Logarithm (log):", np.round(np.log(nums), 4))
    print("Exponential (exp):", np.round(np.exp(nums), 4))


# -----------------------------------
# 4. PRACTICAL EXERCISE
# -----------------------------------
def practical_exercise():
    print("\n" + "=" * 60)
    print("4. PRACTICAL EXERCISE: BROADCASTING & EXPONENTIAL")
    print("=" * 60)
    print("Task: Create a 4x3 matrix and a 1x3 vector.")
    print("      Add them together to demonstrate broadcasting.")
    print("      Compute the exponential of the resulting matrix.")

    # 1. Create a 4x3 matrix (values from 1 to 12)
    matrix = np.arange(1, 13).reshape(4, 3)
    
    # 2. Create a 1x3 vector
    vector = np.array([100, 200, 300])

    print("\nCreated 4x3 Matrix:")
    print(matrix)
    print("Shape of matrix:", matrix.shape)

    print("\nCreated 1x3 Vector:")
    print(vector)
    print("Shape of vector:", vector.shape)

    # 3. Add matrix and vector (Broadcasting)
    broadcasted_sum = matrix + vector
    print("\nResult of Matrix + Vector (Broadcasting):")
    print(broadcasted_sum)
    print("Shape of result:", broadcasted_sum.shape)

    # 4. Compute exponential of resulting matrix
    exponential_result = np.exp(broadcasted_sum / 100) # Dividing by 100 to avoid overflow values display
    print("\nExponential of (Matrix + Vector) / 100:")
    print(np.round(exponential_result, 4))


def main():
    print("=== Day 14: NumPy Vectorized Operations & Broadcasting ===")
    vectorized_operations()
    broadcasting_rules()
    universal_functions()
    practical_exercise()
    print("\nDay 14 Completed Successfully!")

if __name__ == "__main__":
    main()
