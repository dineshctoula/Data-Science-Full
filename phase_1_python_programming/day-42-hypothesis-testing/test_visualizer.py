"""Tests for Day 42 hypothesis-testing visualization output and validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hypothesis_engine import HypothesisEngine
from visualizer import HypothesisVisualizer


class HypothesisVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = HypothesisVisualizer(temporary_directory)
            t_result = HypothesisEngine.one_sample_t_test(np.array([2.0, 3.0, 4.0, 5.0]), 0.0)
            null_path = visualizer.plot_null_distribution(t_result)
            conversion_path = visualizer.plot_conversion_rates(96, 1200, 109, 1150)
            chi_path = visualizer.plot_chi_square_contributions(
                np.array([50.0, 45.0, 55.0]), np.array([50.0, 50.0, 50.0])
            )
            self.assertTrue(Path(null_path).is_file())
            self.assertTrue(Path(conversion_path).is_file())
            self.assertTrue(Path(chi_path).is_file())

    def test_invalid_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = HypothesisVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "HypothesisResult"):
                visualizer.plot_null_distribution("not a result")
            with self.assertRaisesRegex(ValueError, "exceed"):
                visualizer.plot_conversion_rates(10, 5, 1, 10)
            with self.assertRaisesRegex(ValueError, "at least two"):
                visualizer.plot_chi_square_contributions(np.array([1.0]), np.array([1.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
