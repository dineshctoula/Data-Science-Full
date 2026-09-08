"""Tests for Day 38 calculus visualization output and validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from visualizer import CalculusVisualizer


class CalculusVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = CalculusVisualizer(temporary_directory)
            x = np.linspace(-2, 2, 40)
            tangent_path = visualizer.plot_tangent_line(x, x**2, point=1.0, slope=2.0)
            loss_path = visualizer.plot_loss_history(np.array([10.0, 1.0, 0.1]))
            self.assertTrue(Path(tangent_path).is_file())
            self.assertTrue(Path(loss_path).is_file())

    def test_invalid_curves_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = CalculusVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "equal shapes"):
                visualizer.plot_tangent_line(
                    np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]), 0.0, 1.0
                )
            with self.assertRaisesRegex(ValueError, "cannot be negative"):
                visualizer.plot_loss_history(np.array([1.0, -1.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
