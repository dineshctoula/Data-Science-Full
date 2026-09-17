"""Tests for Day 43 confidence-interval visualization output and validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from confidence_engine import ConfidenceEngine
from visualizer import ConfidenceVisualizer


class ConfidenceVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = ConfidenceVisualizer(temporary_directory)
            interval = ConfidenceEngine.mean_t_interval(np.array([68.0, 70.0, 72.0, 71.0, 69.0]))
            interval_path = visualizer.plot_interval(interval, true_value=70.0)
            coverage_path = visualizer.plot_coverage_simulation(trials=15, seed=7)
            width_path = visualizer.plot_width_vs_confidence(np.array([68.0, 70.0, 72.0, 71.0, 69.0, 73.0]))
            self.assertTrue(Path(interval_path).is_file())
            self.assertTrue(Path(coverage_path).is_file())
            self.assertTrue(Path(width_path).is_file())

    def test_invalid_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = ConfidenceVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "ConfidenceInterval"):
                visualizer.plot_interval("not an interval")
            with self.assertRaisesRegex(ValueError, "at least 2"):
                visualizer.plot_coverage_simulation(trials=1)
            with self.assertRaisesRegex(ValueError, "at least two"):
                visualizer.plot_width_vs_confidence(np.array([1.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
