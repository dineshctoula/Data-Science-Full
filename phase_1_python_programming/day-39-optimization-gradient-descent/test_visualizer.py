"""Tests for Day 39 optimization visualization output and validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from visualizer import OptimizationVisualizer


class OptimizationVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = OptimizationVisualizer(temporary_directory)
            loss_path = visualizer.plot_loss_history(np.array([10.0, 1.0, 0.1]))
            contour_path = visualizer.plot_contour_path(
                np.array([[-2.0, 2.0], [0.0, 0.5], [1.0, 0.0]]),
                objective=lambda values: float(values[0] ** 2 + values[1] ** 2),
                target=np.array([0.0, 0.0]),
            )
            comparison_path = visualizer.plot_learning_rate_comparison(
                {0.05: np.array([10.0, 5.0, 2.5]), 0.2: np.array([10.0, 0.4, 0.01])}
            )
            self.assertTrue(Path(loss_path).is_file())
            self.assertTrue(Path(contour_path).is_file())
            self.assertTrue(Path(comparison_path).is_file())

    def test_invalid_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = OptimizationVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "cannot be negative"):
                visualizer.plot_loss_history(np.array([1.0, -1.0]))
            with self.assertRaisesRegex(ValueError, r"\(steps, 2\)"):
                visualizer.plot_contour_path(
                    np.array([[1.0], [2.0]]),
                    objective=lambda values: float(values[0] ** 2),
                )
            with self.assertRaisesRegex(ValueError, "At least one"):
                visualizer.plot_learning_rate_comparison({})


if __name__ == "__main__":
    unittest.main(verbosity=2)
