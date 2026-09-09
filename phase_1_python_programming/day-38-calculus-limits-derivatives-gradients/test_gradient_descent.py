"""Regression tests for Day 38 gradient-descent examples."""

import unittest

import numpy as np

from gradient_descent import GradientDescent


class GradientDescentTests(unittest.TestCase):
    def test_quadratic_descent_reaches_target_and_reduces_loss(self) -> None:
        result = GradientDescent.quadratic(initial_value=-5.0, target=3.0)
        self.assertAlmostEqual(result.parameters[0], 3.0, places=3)
        self.assertLess(result.loss_history[-1], result.loss_history[0])
        self.assertEqual(result.steps, 50)
        self.assertEqual(result.initial_loss, result.loss_history[0])
        self.assertEqual(result.final_loss, result.loss_history[-1])

    def test_linear_regression_recovers_known_relationship(self) -> None:
        features = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        target = 4.0 + 2.5 * features[:, 0]
        result = GradientDescent.linear_regression(features, target, learning_rate=0.1, iterations=500)
        np.testing.assert_allclose(result.parameters, np.array([4.0, 2.5]), atol=1e-5)
        np.testing.assert_allclose(GradientDescent.predict(features, result.parameters), target, atol=1e-5)
        self.assertLess(result.loss_history[-1], result.loss_history[0])

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive finite"):
            GradientDescent.quadratic(0.0, 1.0, learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            GradientDescent.quadratic(float("nan"), 1.0)
        with self.assertRaisesRegex(ValueError, "constant"):
            GradientDescent.linear_regression(np.ones((3, 1)), np.arange(3.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            GradientDescent.linear_regression(np.array([[1.0], [np.inf]]), np.array([1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "finite"):
            GradientDescent.predict(np.array([[1.0]]), np.array([np.nan, 2.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
