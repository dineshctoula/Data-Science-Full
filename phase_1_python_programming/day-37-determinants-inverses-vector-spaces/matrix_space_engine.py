"""Core linear-algebra utilities for Day 37.

The implementations intentionally expose the mathematical building blocks behind
determinants, inverses, and the four fundamental subspaces of a matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SubspaceAnalysis:
    """A numerically stable description of a matrix's vector spaces."""

    rank: int
    column_space_basis: np.ndarray
    row_space_basis: np.ndarray
    null_space_basis: np.ndarray
    left_null_space_basis: np.ndarray


class MatrixSpaceEngine:
    """Analyze square matrices and their associated vector spaces."""

    @staticmethod
    def _as_matrix(matrix: np.ndarray) -> np.ndarray:
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2:
            raise ValueError("Expected a two-dimensional matrix.")
        if 0 in array.shape:
            raise ValueError("Expected a matrix with at least one row and column.")
        return array

    @staticmethod
    def _square(matrix: np.ndarray) -> np.ndarray:
        array = MatrixSpaceEngine._as_matrix(matrix)
        if array.shape[0] != array.shape[1]:
            raise ValueError("This operation requires a square matrix.")
        return array

    @staticmethod
    def minor(matrix: np.ndarray, row: int, column: int) -> np.ndarray:
        """Return the submatrix made by removing one row and one column."""
        array = MatrixSpaceEngine._square(matrix)
        if not (0 <= row < array.shape[0] and 0 <= column < array.shape[1]):
            raise IndexError("Minor indices are outside the matrix dimensions.")
        return np.delete(np.delete(array, row, axis=0), column, axis=1)

    @staticmethod
    def determinant_by_cofactor(matrix: np.ndarray) -> float:
        """Compute a determinant recursively using first-row cofactor expansion."""
        array = MatrixSpaceEngine._square(matrix)
        n = array.shape[0]
        if n == 0:
            return 1.0
        if n == 1:
            return float(array[0, 0])
        if n == 2:
            return float(array[0, 0] * array[1, 1] - array[0, 1] * array[1, 0])

        return float(sum(
            array[0, col] * (-1) ** col
            * MatrixSpaceEngine.determinant_by_cofactor(MatrixSpaceEngine.minor(array, 0, col))
            for col in range(n)
        ))

    @staticmethod
    def cofactor_matrix(matrix: np.ndarray) -> np.ndarray:
        """Build the signed-minor matrix C where C[i, j] is a cofactor."""
        array = MatrixSpaceEngine._square(matrix)
        return np.array([
            [(-1) ** (row + col) * MatrixSpaceEngine.determinant_by_cofactor(
                MatrixSpaceEngine.minor(array, row, col)
            ) for col in range(array.shape[1])]
            for row in range(array.shape[0])
        ])

    @staticmethod
    def inverse_by_adjugate(matrix: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
        """Compute A⁻¹ = adj(A) / det(A), rejecting singular matrices."""
        array = MatrixSpaceEngine._square(matrix)
        determinant = MatrixSpaceEngine.determinant_by_cofactor(array)
        if abs(determinant) <= tolerance:
            raise ValueError("A singular matrix has no inverse.")
        return MatrixSpaceEngine.cofactor_matrix(array).T / determinant

    @staticmethod
    def verify_inverse(matrix: np.ndarray, inverse: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Return whether both products are identity matrices within tolerance."""
        array = MatrixSpaceEngine._square(matrix)
        inverse_array = MatrixSpaceEngine._square(inverse)
        if array.shape != inverse_array.shape:
            return False

        identity = np.eye(array.shape[0])
        # Checking both multiplication orders catches an incorrectly ordered adjugate.
        return bool(
            np.allclose(array @ inverse_array, identity, atol=tolerance)
            and np.allclose(inverse_array @ array, identity, atol=tolerance)
        )

    @staticmethod
    def invertibility_report(matrix: np.ndarray, tolerance: float = 1e-12) -> dict[str, Any]:
        """Return numerical diagnostics for the invertible matrix theorem."""
        array = MatrixSpaceEngine._square(matrix)
        determinant = float(np.linalg.det(array))
        rank = int(np.linalg.matrix_rank(array, tol=tolerance))
        condition_number = float(np.linalg.cond(array))
        return {
            "shape": array.shape,
            "determinant": determinant,
            "rank": rank,
            "full_rank": rank == array.shape[0],
            "is_invertible": bool(abs(determinant) > tolerance and rank == array.shape[0]),
            "condition_number": condition_number,
        }

    @staticmethod
    def analyze_subspaces(matrix: np.ndarray, tolerance: float = 1e-10) -> SubspaceAnalysis:
        """Find orthonormal bases using the compact singular value decomposition.

        The column and row spaces are spanned by singular vectors associated with
        non-zero singular values. Remaining singular vectors span null spaces.
        """
        array = MatrixSpaceEngine._as_matrix(matrix)
        u, singular_values, vt = np.linalg.svd(array, full_matrices=True)
        if tolerance <= 0:
            raise ValueError("Tolerance must be positive.")
        # A relative threshold remains meaningful when a matrix is rescaled.
        cutoff = tolerance * singular_values[0] if singular_values.size else tolerance
        rank = int(np.sum(singular_values > cutoff))
        return SubspaceAnalysis(
            rank=rank,
            column_space_basis=u[:, :rank],
            row_space_basis=vt[:rank, :].T,
            null_space_basis=vt[rank:, :].T,
            left_null_space_basis=u[:, rank:],
        )

    @staticmethod
    def solve_or_least_squares(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the least-squares solution and its residual norm."""
        array = MatrixSpaceEngine._as_matrix(matrix)
        target_array = np.asarray(target, dtype=float).reshape(-1)
        if target_array.shape[0] != array.shape[0]:
            raise ValueError("Target length must match the matrix row count.")
        solution, _, _, _ = np.linalg.lstsq(array, target_array, rcond=None)
        residual_norm = float(np.linalg.norm(array @ solution - target_array))
        return solution, residual_norm


if __name__ == "__main__":
    A = np.array([[4.0, 7.0], [2.0, 6.0]])
    inverse = MatrixSpaceEngine.inverse_by_adjugate(A)
    assert np.allclose(A @ inverse, np.eye(2))

    dependent = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    spaces = MatrixSpaceEngine.analyze_subspaces(dependent)
    assert spaces.rank == 1
    assert np.allclose(dependent @ spaces.null_space_basis, 0.0)
    print("MatrixSpaceEngine validation tests passed successfully!")
