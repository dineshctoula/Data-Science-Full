"""Regression tests for the Day 44 correlation and simple-regression engine."""

import unittest

import numpy as np

from regression_engine import RegressionEngine


class RegressionEngineTests(unittest.TestCase):
    """Check textbook identities, perfect-line recovery, and input safeguards."""

    def test_pearson_is_one_for_a_perfect_positive_line(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x
        result = RegressionEngine.pearson_correlation(x, y)
        self.assertAlmostEqual(result.coefficient, 1.0, places=12)
        self.assertEqual(result.strength, "strong")

    def test_pearson_is_minus_one_for_a_perfect_negative_line(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = 10.0 - 3.0 * x
        result = RegressionEngine.pearson_correlation(x, y)
        self.assertAlmostEqual(result.coefficient, -1.0, places=12)

    def test_simple_regression_recovers_known_intercept_and_slope(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 5.0 + 1.5 * x
        fit = RegressionEngine.fit_simple_regression(x, y)
        self.assertAlmostEqual(fit.intercept, 5.0, places=10)
        self.assertAlmostEqual(fit.slope, 1.5, places=10)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=12)
        self.assertAlmostEqual(fit.rmse, 0.0, places=12)
        np.testing.assert_allclose(fit.predict(np.array([6.0])), np.array([14.0]), atol=1e-10)

    def test_r_squared_equals_pearson_squared_for_simple_regression(self) -> None:
        hours, scores = RegressionEngine.generate_study_hours_scores(n=40, seed=44)
        fit = RegressionEngine.fit_simple_regression(hours, scores)
        pearson = RegressionEngine.pearson_correlation(hours, scores)
        # For one predictor, R² = r² is an exact algebraic identity.
        self.assertAlmostEqual(fit.r_squared, pearson.coefficient**2, places=10)

    def test_spearman_exceeds_pearson_on_curved_monotone_data(self) -> None:
        x, y = RegressionEngine.generate_nonlinear_pair(n=40, seed=44)
        pearson = RegressionEngine.pearson_correlation(x, y)
        spearman = RegressionEngine.spearman_correlation(x, y)
        self.assertGreater(spearman.coefficient, pearson.coefficient)
        self.assertGreater(spearman.coefficient, 0.9)

    def test_residuals_sum_near_zero_for_ols_fit(self) -> None:
        hours, scores = RegressionEngine.generate_study_hours_scores(n=25, seed=7)
        fit = RegressionEngine.fit_simple_regression(hours, scores)
        # OLS residuals are orthogonal to the constant column, so they sum to ~0.
        self.assertAlmostEqual(float(fit.residuals.sum()), 0.0, places=8)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal-length"):
            RegressionEngine.pearson_correlation(np.array([1.0, 2.0]), np.array([1.0]))
        with self.assertRaisesRegex(ValueError, "zero variance"):
            RegressionEngine.pearson_correlation(np.array([1.0, 1.0, 1.0]), np.array([2.0, 3.0, 4.0]))
        with self.assertRaisesRegex(ValueError, "non-constant"):
            RegressionEngine.fit_simple_regression(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "finite"):
            RegressionEngine.covariance(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
