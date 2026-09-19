"""Regression tests for the Day 45 multiple-linear-regression engine."""

import unittest

import numpy as np

from multiple_regression_engine import MultipleRegressionEngine


class MultipleRegressionEngineTests(unittest.TestCase):
    """Check coefficient recovery, diagnostics, VIF, and input safeguards."""

    def test_fit_recovers_known_multi_predictor_surface(self) -> None:
        features = np.array(
            [
                [1.0, 2.0],
                [2.0, 1.0],
                [3.0, 4.0],
                [4.0, 2.0],
                [5.0, 3.0],
                [6.0, 1.0],
                [7.0, 5.0],
                [8.0, 2.0],
            ]
        )
        target = 10.0 + 2.0 * features[:, 0] - 0.5 * features[:, 1]
        fit = MultipleRegressionEngine.fit(features, target, ("a", "b"))
        np.testing.assert_allclose(fit.coefficients, np.array([10.0, 2.0, -0.5]), atol=1e-9)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=12)
        self.assertAlmostEqual(fit.rmse, 0.0, places=12)

    def test_adjusted_r_squared_is_below_r_squared_with_extra_noise(self) -> None:
        features, target, names = MultipleRegressionEngine.generate_housing_sample(n=80, seed=45)
        fit = MultipleRegressionEngine.fit(features, target, names)
        # With noise and multiple predictors, Adj. R² should not exceed R².
        self.assertLessEqual(fit.adjusted_r_squared, fit.r_squared + 1e-12)
        self.assertGreater(fit.r_squared, 0.7)

    def test_housing_coefficients_have_expected_signs(self) -> None:
        features, target, names = MultipleRegressionEngine.generate_housing_sample(n=120, seed=45)
        fit = MultipleRegressionEngine.fit(features, target, names)
        slopes = dict(zip(names, fit.slopes))
        # Larger homes and more bedrooms raise price; farther from center lowers it.
        self.assertGreater(slopes["sqft"], 0.0)
        self.assertGreater(slopes["bedrooms"], 0.0)
        self.assertLess(slopes["distance_km"], 0.0)

    def test_vif_flags_near_duplicate_predictors(self) -> None:
        features, _, names = MultipleRegressionEngine.generate_collinear_sample(n=60, seed=45)
        vifs = MultipleRegressionEngine.variance_inflation_factors(features, names)
        # x1 and x2 are almost linear copies, so their VIFs explode.
        self.assertGreater(vifs["x1"], 50.0)
        self.assertGreater(vifs["x2"], 50.0)
        self.assertLess(vifs["x3"], 5.0)

    def test_predict_matches_in_sample_fitted_values(self) -> None:
        features, target, names = MultipleRegressionEngine.generate_housing_sample(n=40, seed=7)
        fit = MultipleRegressionEngine.fit(features, target, names)
        np.testing.assert_allclose(fit.predict(features), fit.predictions, atol=1e-9)

    def test_residuals_sum_near_zero(self) -> None:
        features, target, names = MultipleRegressionEngine.generate_housing_sample(n=50, seed=3)
        fit = MultipleRegressionEngine.fit(features, target, names)
        # The intercept column forces OLS residuals to sum to approximately zero.
        self.assertAlmostEqual(float(fit.residuals.sum()), 0.0, places=6)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            MultipleRegressionEngine.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "more observations"):
            MultipleRegressionEngine.fit(
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                np.array([1.0, 2.0, 3.0]),
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            MultipleRegressionEngine.variance_inflation_factors(np.array([[1.0], [2.0], [3.0], [4.0]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            MultipleRegressionEngine.fit(
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
                np.array([1.0, 2.0, np.nan, 4.0]),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
