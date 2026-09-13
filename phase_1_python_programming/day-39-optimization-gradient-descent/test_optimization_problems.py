"""Regression tests for Day 39 quadratic and regression problem helpers."""

import unittest

import numpy as np

from optimization_problems import OptimizationProblems


class OptimizationProblemsTests(unittest.TestCase):
    """Check known-minimum recovery, path recording, and input safeguards."""

    def test_quadratic_bowl_reaches_the_known_minimum(self) -> None:
        result = OptimizationProblems.quadratic_bowl(initial_value=-5.0, target=3.0)
        self.assertAlmostEqual(result.parameters[0], 3.0, places=6)
        np.testing.assert_allclose(result.target, np.array([3.0]))
        self.assertLess(result.final_loss, result.initial_loss)
        self.assertEqual(result.steps, 80)
        self.assertEqual(result.parameter_history.shape, (81, 1))
        self.assertGreater(result.loss_reduction_ratio, 0.99)

    def test_already_optimal_bowl_has_zero_reduction_ratio(self) -> None:
        result = OptimizationProblems.quadratic_bowl(initial_value=3.0, target=3.0, iterations=3)
        # A perfect initial guess has no loss to reduce, but should still
        # produce a well-defined progress summary.
        self.assertEqual(result.loss_reduction, 0.0)
        self.assertEqual(result.loss_reduction_ratio, 0.0)

    def test_elongated_bowl_recovers_the_two_dimensional_minimum(self) -> None:
        result = OptimizationProblems.elongated_bowl()
        np.testing.assert_allclose(result.parameters, np.array([2.0, -1.0]), atol=1e-4)
        self.assertEqual(result.parameter_history.shape[1], 2)
        self.assertLess(result.final_loss, result.initial_loss)
        np.testing.assert_array_equal(result.parameter_history[0], np.array([-3.0, 3.0]))

    def test_learning_rate_comparison_keeps_each_run_separate(self) -> None:
        comparisons = OptimizationProblems.compare_quadratic_learning_rates(
            initial_value=-5.0, target=3.0, learning_rates=np.array([0.05, 0.2]), iterations=20
        )
        self.assertEqual(set(comparisons), {0.05, 0.2})
        self.assertEqual(comparisons[0.05].steps, 20)
        self.assertLess(comparisons[0.2].final_loss, comparisons[0.05].final_loss)

    def test_linear_regression_recovers_known_relationships(self) -> None:
        features = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        target = 4.0 + 2.5 * features[:, 0]
        result = OptimizationProblems.linear_regression(features, target, learning_rate=0.1, iterations=500)
        np.testing.assert_allclose(result.parameters, np.array([4.0, 2.5]), atol=1e-5)
        np.testing.assert_allclose(OptimizationProblems.predict(features, result.parameters), target, atol=1e-5)
        self.assertLess(result.final_loss, result.initial_loss)

        two_features = np.array([[1.0, 3.0], [2.0, 1.0], [3.0, 4.0], [4.0, 2.0]])
        two_target = 1.5 + 2.0 * two_features[:, 0] - 0.5 * two_features[:, 1]
        two_result = OptimizationProblems.linear_regression(
            two_features, two_target, learning_rate=0.1, iterations=500
        )
        np.testing.assert_allclose(two_result.parameters, np.array([1.5, 2.0, -0.5]), atol=1e-5)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            OptimizationProblems.quadratic_bowl(initial_value=float("nan"))
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            OptimizationProblems.elongated_bowl(initial_parameters=np.array([1.0]))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            OptimizationProblems.compare_quadratic_learning_rates(learning_rates=np.array([]))
        with self.assertRaisesRegex(ValueError, "constant"):
            OptimizationProblems.linear_regression(np.ones((3, 1)), np.arange(3.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            OptimizationProblems.linear_regression(np.array([[1.0], [np.inf]]), np.array([1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "finite"):
            OptimizationProblems.predict(np.array([[1.0]]), np.array([np.nan, 2.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
