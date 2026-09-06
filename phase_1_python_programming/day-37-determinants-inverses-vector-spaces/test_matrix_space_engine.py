"""Automated regression tests for the Day 37 linear-algebra utilities."""

import unittest

import numpy as np

from matrix_space_engine import MatrixSpaceEngine


class MatrixSpaceEngineTests(unittest.TestCase):
    def test_cofactor_determinant_matches_numpy_for_three_by_three_matrix(self) -> None:
        matrix = np.array([[3.0, 2.0, 1.0], [1.0, 0.0, 2.0], [4.0, 1.0, 1.0]])
        self.assertAlmostEqual(
            MatrixSpaceEngine.determinant_by_cofactor(matrix), np.linalg.det(matrix)
        )

    def test_adjugate_inverse_satisfies_both_identity_products(self) -> None:
        matrix = np.array([[4.0, 7.0], [2.0, 6.0]])
        inverse = MatrixSpaceEngine.inverse_by_adjugate(matrix)
        self.assertTrue(MatrixSpaceEngine.verify_inverse(matrix, inverse))

    def test_singular_matrix_cannot_be_inverted(self) -> None:
        with self.assertRaisesRegex(ValueError, "singular"):
            MatrixSpaceEngine.inverse_by_adjugate(np.array([[1.0, 2.0], [2.0, 4.0]]))

    def test_subspace_dimensions_obey_rank_nullity(self) -> None:
        matrix = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])
        spaces = MatrixSpaceEngine.analyze_subspaces(matrix)
        self.assertEqual(spaces.rank, 2)
        self.assertEqual(spaces.rank + spaces.null_space_basis.shape[1], matrix.shape[1])
        self.assertTrue(np.allclose(matrix @ spaces.null_space_basis, 0.0))
        self.assertTrue(np.allclose(matrix.T @ spaces.left_null_space_basis, 0.0))

    def test_least_squares_rejects_target_with_wrong_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "Target length"):
            MatrixSpaceEngine.solve_or_least_squares(np.eye(2), np.array([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
