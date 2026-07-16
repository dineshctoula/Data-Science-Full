#!/usr/bin/env python3
"""
====================================================
100 Days of Data Science - Day 16
Topic: NumPy Matrix Operations & Linear Algebra

Topics Covered
--------------
1. Matrix Creation, Multiplication, & Transpose
2. Determinants & Inverses (np.linalg.det, np.linalg.inv)
3. Solving Systems of Linear Equations (np.linalg.solve)
4. Practical Exercise (Solving 2x + y = 5, x - 3y = -1)

Author: Dinesh Sitoula
====================================================
"""

import numpy as np

# -----------------------------------
# 1. MATRIX MULTIPLICATION & TRANSPOSE
# -----------------------------------
def matrix_multiplication_transpose():
    print("=" * 60)
    print("1. MATRIX MULTIPLICATION & TRANSPOSE")
    print("=" * 60)

    # Creating matrices A (2x3) and B (3x2)
    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    B = np.array([
        [7, 8],
        [9, 10],
        [11, 12]
    ])

    print("Matrix A (2x3):\n", A)
    print("Matrix B (3x2):\n", B)

    # Transpose of A
    print("\nTranspose of A (A.T):\n", A.T)

    # Matrix Multiplication (A @ B)
    # Result is a (2x2) matrix
    C1 = A @ B
    C2 = np.dot(A, B)

    print("\nMatrix Multiplication (A @ B):\n", C1)
    print("Matrix Multiplication (np.dot(A, B)):\n", C2)

    # Verify both methods produce the same result
    assert np.array_equal(C1, C2), "Matrix multiplication methods must yield identical results!"


# -----------------------------------
# 2. DETERMINANTS & INVERSES
# -----------------------------------
def determinant_and_inverse():
    print("\n" + "=" * 60)
    print("2. DETERMINANTS & MATRIX INVERSES")
    print("=" * 60)

    # Define a square 2x2 matrix
    A = np.array([
        [4, 7],
        [2, 6]
    ])
    print("Matrix A:\n", A)

    # Determinant: det(A) = ad - bc = (4*6) - (7*2) = 24 - 14 = 10
    det_A = np.linalg.det(A)
    print(f"\nDeterminant of A: {det_A:.2f}")

    # Check if invertible (determinant != 0)
    if not np.isclose(det_A, 0.0):
        # Inverse: A^-1
        A_inv = np.linalg.inv(A)
        print("\nInverse of A (A^-1):\n", A_inv)

        # Verification: A @ A_inv should equal Identity Matrix (I)
        identity = A @ A_inv
        print("\nVerification A @ A^-1 (should be Identity Matrix):\n", np.round(identity, 2))
    else:
        print("\nMatrix is singular and cannot be inverted!")


# -----------------------------------
# 3. SOLVING SYSTEMS OF LINEAR EQUATIONS
# -----------------------------------
def solving_linear_systems():
    print("\n" + "=" * 60)
    print("3. SOLVING SYSTEMS OF LINEAR EQUATIONS")
    print("=" * 60)
    print("Example system:")
    print("  3x + 2y = 8")
    print("   x - 2y = 0")

    # In matrix form Ax = B
    # A is coefficient matrix, B is ordinate vector
    A = np.array([
        [3, 2],
        [1, -2]
    ])
    B = np.array([8, 0])

    print("\nCoefficient Matrix (A):\n", A)
    print("Ordinate Vector (B):", B)

    # Solve for x
    x = np.linalg.solve(A, B)
    print("\nSolution Vector (x, y):", x)
    print(f"Verified: x = {x[0]:.2f}, y = {x[1]:.2f}")


# -----------------------------------
# 4. PRACTICAL EXERCISE
# -----------------------------------
def practical_exercise():
    print("\n" + "=" * 60)
    print("4. PRACTICAL EXERCISE: SOLVING LINEAR EQUATIONS")
    print("=" * 60)
    print("Task: Solve the system of equations:")
    print("      2x + y = 5")
    print("      x - 3y = -1")
    print("      using np.linalg.solve.")

    # 1. Define coefficient matrix A and constants vector B
    A = np.array([
        [2, 1],
        [1, -3]
    ])
    B = np.array([5, -1])

    print("\nCoefficient Matrix (A):\n", A)
    print("Constants Vector (B):", B)

    # 2. Solve for solutions (x, y)
    solutions = np.linalg.solve(A, B)
    
    print("\nSolutions:")
    print(f"  x = {solutions[0]:.4f}")
    print(f"  y = {solutions[1]:.4f}")

    # 3. Verify correctness by plugging back into equations
    eq1_check = 2 * solutions[0] + solutions[1]
    eq2_check = solutions[0] - 3 * solutions[1]
    
    print("\nVerification:")
    print(f"  Equation 1 (2x + y) value: {eq1_check:.2f} (Expected: 5.00)")
    print(f"  Equation 2 (x - 3y) value: {eq2_check:.2f} (Expected: -1.00)")


def main():
    print("=== Day 16: NumPy Matrix Operations & Linear Algebra ===")
    matrix_multiplication_transpose()
    determinant_and_inverse()
    solving_linear_systems()
    practical_exercise()
    print("\nDay 16 Completed Successfully!")

if __name__ == "__main__":
    main()
