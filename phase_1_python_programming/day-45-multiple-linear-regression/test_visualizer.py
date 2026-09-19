"""Tests for Day 45 multiple-regression visualization output and validation."""

import tempfile
import unittest
from pathlib import Path

from multiple_regression_engine import MultipleRegressionEngine
from visualizer import MultipleRegressionVisualizer


class MultipleRegressionVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = MultipleRegressionVisualizer(temporary_directory)
            features, target, names = MultipleRegressionEngine.generate_housing_sample(n=40, seed=2)
            fit = MultipleRegressionEngine.fit(features, target, names)
            collinear_x, _, collinear_names = MultipleRegressionEngine.generate_collinear_sample(
                n=40, seed=2
            )
            vifs = MultipleRegressionEngine.variance_inflation_factors(collinear_x, collinear_names)

            actual_path = visualizer.plot_actual_vs_predicted(fit)
            residual_path = visualizer.plot_residuals(fit)
            coef_path = visualizer.plot_coefficients(fit)
            vif_path = visualizer.plot_vif(vifs)
            self.assertTrue(Path(actual_path).is_file())
            self.assertTrue(Path(residual_path).is_file())
            self.assertTrue(Path(coef_path).is_file())
            self.assertTrue(Path(vif_path).is_file())

    def test_invalid_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = MultipleRegressionVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "MultipleRegressionResult"):
                visualizer.plot_actual_vs_predicted("bad")
            with self.assertRaisesRegex(ValueError, "At least one"):
                visualizer.plot_vif({})


if __name__ == "__main__":
    unittest.main(verbosity=2)
