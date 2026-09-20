"""Tests for Day 46 logistic visualization output and validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from logistic_engine import LogisticEngine
from visualizer import LogisticVisualizer


class LogisticVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = LogisticVisualizer(temporary_directory)
            features, labels = LogisticEngine.generate_one_feature_pass_data(n=50, seed=3)
            result = LogisticEngine.fit(features, labels, ("hours",), learning_rate=0.3, iterations=1_200)
            matrix = LogisticEngine.confusion_matrix(labels, result.predictions)

            sigmoid_path = visualizer.plot_sigmoid()
            curve_path = visualizer.plot_probability_curve(features, labels, result)
            loss_path = visualizer.plot_loss_history(result)
            confusion_path = visualizer.plot_confusion_matrix(matrix)
            self.assertTrue(Path(sigmoid_path).is_file())
            self.assertTrue(Path(curve_path).is_file())
            self.assertTrue(Path(loss_path).is_file())
            self.assertTrue(Path(confusion_path).is_file())

    def test_invalid_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = LogisticVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "LogisticResult"):
                visualizer.plot_loss_history("bad")
            with self.assertRaisesRegex(ValueError, r"\(2, 2\)"):
                visualizer.plot_confusion_matrix(np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
