"""Regression tests for the Day 39 batch gradient-descent engine."""

import unittest

import numpy as np

from optimization_engine import BatchGradientDescent


class BatchGradientDescentTests(unittest.TestCase):
    """Check convergence as well as the safeguards around the update loop."""

    def test_minimize_reaches_the_known_quadratic_minimum(self) -> None:
        # f(w) = (w - 3)^2 has its unique minimum at w = 3.
        trace = BatchGradientDescent.minimize(
            objective=lambda values: (values[0] - 3.0) ** 2,
            gradient=lambda values: np.array([2.0 * (values[0] - 3.0)]),
            initial_parameters=np.array([-5.0]),
            learning_rate=0.1,
            iterations=80,
        )
        self.assertAlmostEqual(trace.parameters[0], 3.0, places=6)
        self.assertLess(trace.final_loss, trace.initial_loss)
        self.assertEqual(trace.loss_history.size, 81)
        self.assertEqual(trace.gradient_norm_history.size, 80)

    def test_minimize_does_not_mutate_the_initial_array(self) -> None:
        initial = np.array([8.0])
        BatchGradientDescent.minimize(
            lambda values: values[0] ** 2,
            lambda values: 2.0 * values,
            initial,
            iterations=2,
        )
        # A caller can safely reuse an initial guess for another experiment.
        np.testing.assert_array_equal(initial, np.array([8.0]))

    def test_invalid_hyperparameters_are_rejected(self) -> None:
        objective = lambda values: float(values @ values)
        gradient = lambda values: 2.0 * values
        with self.assertRaisesRegex(ValueError, "positive finite"):
            BatchGradientDescent.minimize(objective, gradient, np.array([1.0]), learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            BatchGradientDescent.minimize(objective, gradient, np.array([1.0]), iterations=0)

    def test_invalid_gradient_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            BatchGradientDescent.minimize(
                lambda values: float(values @ values),
                lambda values: np.array([1.0, 2.0]),
                np.array([1.0]),
            )

    def test_non_finite_objective_is_rejected(self) -> None:
        with self.assertRaisesRegex(FloatingPointError, "finite loss"):
            BatchGradientDescent.minimize(
                lambda values: float("inf"),
                lambda values: 2.0 * values,
                np.array([1.0]),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
