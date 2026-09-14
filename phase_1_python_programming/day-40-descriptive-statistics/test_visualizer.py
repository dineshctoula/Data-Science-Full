"""Tests for Day 40 descriptive-statistics visualization output and validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stats_engine import DescriptiveStats, FiveNumberSummary
from visualizer import StatsVisualizer


class StatsVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 12.0])
        report = DescriptiveStats.summarize(values)
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = StatsVisualizer(temporary_directory)
            histogram_path = visualizer.plot_histogram(report.values, report.mean, report.median)
            boxplot_path = visualizer.plot_boxplot(report.values, report.five_number)
            zscore_path = visualizer.plot_zscore_strip(report)
            self.assertTrue(Path(histogram_path).is_file())
            self.assertTrue(Path(boxplot_path).is_file())
            self.assertTrue(Path(zscore_path).is_file())

    def test_invalid_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = StatsVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "finite"):
                visualizer.plot_histogram(np.array([1.0, np.inf]), 1.0, 1.0)
            with self.assertRaisesRegex(ValueError, "FiveNumberSummary"):
                visualizer.plot_boxplot(np.array([1.0, 2.0, 3.0]), "summary")
            with self.assertRaisesRegex(ValueError, "DescriptiveReport"):
                visualizer.plot_zscore_strip(FiveNumberSummary(1.0, 2.0, 3.0, 4.0, 5.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
