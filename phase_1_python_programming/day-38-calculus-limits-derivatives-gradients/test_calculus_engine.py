"""Regression tests for Day 38 limit and derivative utilities."""

import unittest

import numpy as np

from calculus_engine import CalculusEngine


class CalculusEngineTests(unittest.TestCase):
    def test_one_sided_limits_converge_for_continuous_function(self) -> None:
        left, right = CalculusEngine.one_sided_limit(lambda x: x**2 + 2 * x, 3.0)
        self.assertAlmostEqual(left, 15.0, places=3)
        self.assertAlmostEqual(right, 15.0, places=3)

    def test_symmetric_derivative_matches_quadratic_slope(self) -> None:
        self.assertAlmostEqual(
            CalculusEngine.symmetric_derivative(lambda x: x**2 + 3 * x, 2.0), 7.0, places=5
        )

    def test_exact_polynomial_derivative_matches_finite_difference(self) -> None:
        coefficients = np.array([2.0, -3.0, 4.0, -1.0])
        points = np.array([-2.0, -0.5, 0.0, 1.5])
        exact = CalculusEngine.polynomial_derivative(coefficients, points)
        numerical = CalculusEngine.derivative_curve(
            lambda x: CalculusEngine.polynomial_value(coefficients, x), points
        )
        np.testing.assert_allclose(numerical, exact, atol=1e-6)

    def test_invalid_step_and_polynomial_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive finite"):
            CalculusEngine.symmetric_derivative(lambda x: x, 0.0, step=0)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            CalculusEngine.polynomial_derivative(np.array([]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
