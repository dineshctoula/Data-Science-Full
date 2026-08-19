#!/usr/bin/env python3
"""
====================================================
100 Days of Data Science - Day 16
Topic: NumPy Matrix Operations & Linear Algebra

Topics Covered
--------------
1. Matrix Creation, Multiplication, & Transpose
2. Determinants & Matrix Inverses
3. Solving Systems of Linear Equations
4. Practical Exercise

Author: Dinesh Sitoula
====================================================
"""

import numpy as np


# --------------------------------------------------
# 1. MATRIX MULTIPLICATION & TRANSPOSE
# --------------------------------------------------
def matrix_multiplication_transpose():
    """Demonstrate matrix creation, multiplication, and transpose."""

    print("=" * 60)
    print("1. MATRIX MULTIPLICATION & TRANSPOSE")
    print("=" * 60)

    # Create matrices
    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    B = np.array([
        [7, 8],
        [9, 10],
        [11, 12]
    ])

    print("Matrix A (2x3):")
    print(A)

    print("\nMatrix B (3x2):")
    print(B)

    # Transpose of matrix A
    print("\nTranspose of A:")
    print(A.T)

    # Matrix multiplication using @ operator
    result_at = A @ B

    print("\nMatrix Multiplication (A @ B):")
    print(result_at)

    # Matrix multiplication using np.dot()
    result_dot = np.dot(A, B)

    print("\nMatrix Multiplication (np.dot(A, B)):")
    print(result_dot)

    # Verify both methods produce the same result
    assert np.array_equal(result_at, result_dot)

    print("\nMatrix multiplication verified successfully.")


# --------------------------------------------------
# 2. DETERMINANT & MATRIX INVERSE
# --------------------------------------------------
def determinant_and_inverse():
    """Calculate determinant and inverse of a square matrix."""

    print("\n" + "=" * 60)
    print("2. DETERMINANT & MATRIX INVERSE")
    print("=" * 60)

    # Create a square matrix
    A = np.array([
        [4, 7],
        [2, 6]
    ])

    print("Matrix A:")
    print(A)

    # Calculate determinant
    determinant = np.linalg.det(A)

    print(f"\nDeterminant of A: {determinant:.2f}")

    # A matrix can be inverted only if determinant is not zero
    if np.isclose(determinant, 0):
        print("\nMatrix is singular and cannot be inverted.")
        return

    # Calculate inverse
    inverse = np.linalg.inv(A)

    print("\nInverse of A:")
    print(inverse)

    # Verify A × A^-1 = Identity Matrix
    identity = A @ inverse

    print("\nA @ A^-1:")
    print(np.round(identity, 2))

    # Verify result
    expected_identity = np.eye(A.shape[0])

    assert np.allclose(identity, expected_identity)

    print("\nInverse verified successfully.")


# --------------------------------------------------
# 3. SOLVING SYSTEMS OF LINEAR EQUATIONS
# --------------------------------------------------
def solving_linear_systems():
    """Solve a system of linear equations using np.linalg.solve()."""

    print("\n" + "=" * 60)
    print("3. SOLVING SYSTEMS OF LINEAR EQUATIONS")
    print("=" * 60)

    print("Example system:")
    print("  3x + 2y = 8")
    print("   x - 2y = 0")

    # Matrix form:
    # A @ X = B
    #
    # A = coefficient matrix
    # X = unknowns [x, y]
    # B = constants vector

    A = np.array([
        [3, 2],
        [1, -2]
    ])

    B = np.array([8, 0])

    print("\nCoefficient Matrix A:")
    print(A)

    print("\nConstants Vector B:")
    print(B)

    # Solve the system
    solution = np.linalg.solve(A, B)

    x, y = solution

    print("\nSolution:")
    print(f"  x = {x:.2f}")
    print(f"  y = {y:.2f}")

    # Verify the solution
    verification = A @ solution

    print("\nVerification:")
    print("A @ solution =", verification)

    assert np.allclose(verification, B)

    print("Solution verified successfully.")


# --------------------------------------------------
# 4. PRACTICAL EXERCISE
# --------------------------------------------------
def practical_exercise():
    """Solve and verify a system of two linear equations."""

    print("\n" + "=" * 60)
    print("4. PRACTICAL EXERCISE")
    print("=" * 60)

    print("Task: Solve the following equations:")
    print("  2x + y = 5")
    print("  x - 3y = -1")

    # Convert equations into matrix form:
    #
    # 2x + y  = 5
    # x - 3y  = -1
    #
    # A @ X = B

    A = np.array([
        [2, 1],
        [1, -3]
    ])

    B = np.array([5, -1])

    print("\nCoefficient Matrix A:")
    print(A)

    print("\nConstants Vector B:")
    print(B)

    # Solve for x and y
    solution = np.linalg.solve(A, B)

    x, y = solution

    print("\nSolutions:")
    print(f"  x = {x:.4f}")
    print(f"  y = {y:.4f}")

    # Verify the solution using the original equations
    equation_1 = 2 * x + y
    equation_2 = x - 3 * y

    print("\nVerification:")
    print(f"  2x + y = {equation_1:.2f}  (Expected: 5.00)")
    print(f"  x - 3y = {equation_2:.2f}  (Expected: -1.00)")

    # Confirm both equations are satisfied
    assert np.isclose(equation_1, 5)
    assert np.isclose(equation_2, -1)

    print("\nPractical exercise verified successfully.")


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------
def main():
    """Run all Day 16 demonstrations."""

    print("=" * 60)
    print("DAY 16: NUMPY MATRIX OPERATIONS & LINEAR ALGEBRA")
    print("=" * 60)

    matrix_multiplication_transpose()
    determinant_and_inverse()
    solving_linear_systems()
    practical_exercise()

    print("\n" + "=" * 60)
    print("Day 16 Completed Successfully!")
    print("=" * 60)


# --------------------------------------------------
# PROGRAM ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()

