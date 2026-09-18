"""Tests for Day 44 regression visualization output and validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from regression_engine import RegressionEngine
from visualizer import RegressionVisualizer


class RegressionVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = RegressionVisualizer(temporary_directory)
            hours, scores = RegressionEngine.generate_study_hours_scores(n=25, seed=3)
            fit = RegressionEngine.fit_simple_regression(hours, scores)
            curved_x, curved_y = RegressionEngine.generate_nonlinear_pair(n=25, seed=3)
            pearson = RegressionEngine.pearson_correlation(curved_x, curved_y)
            spearman = RegressionEngine.spearman_correlation(curved_x, curved_y)

            scatter_path = visualizer.plot_scatter_with_fit(fit)
            residual_path = visualizer.plot_residuals(fit)
            comparison_path = visualizer.plot_correlation_comparison(
                pearson, spearman, curved_x, curved_y
            )
            self.assertTrue(Path(scatter_path).is_file())
            self.assertTrue(Path(residual_path).is_file())
            self.assertTrue(Path(comparison_path).is_file())

    def test_invalid_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = RegressionVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "RegressionResult"):
                visualizer.plot_scatter_with_fit("not a result")
            with self.assertRaisesRegex(ValueError, "RegressionResult"):
                visualizer.plot_residuals("not a result")
            with self.assertRaisesRegex(ValueError, "CorrelationResult"):
                visualizer.plot_correlation_comparison(
                    "bad", "bad", np.array([1.0, 2.0]), np.array([1.0, 2.0])
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
