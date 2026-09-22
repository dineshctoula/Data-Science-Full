"""Tests for Day 48 Naive Bayes visualization output and validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from naive_bayes_engine import NaiveBayesEngine
from visualizer import NaiveBayesVisualizer


class NaiveBayesVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = NaiveBayesVisualizer(temporary_directory)
            features, labels, names = NaiveBayesEngine.generate_iris_like_sample(
                n_per_class=20, seed=2
            )
            result = NaiveBayesEngine.fit(features, labels, names)
            matrix = NaiveBayesEngine.confusion_matrix(labels, result.predictions, result.classes)

            priors_path = visualizer.plot_class_priors(result)
            density_path = visualizer.plot_feature_likelihoods(result, feature_index=0)
            region_path = visualizer.plot_decision_regions(features, labels, result)
            confusion_path = visualizer.plot_confusion_matrix(matrix, result.classes)
            self.assertTrue(Path(priors_path).is_file())
            self.assertTrue(Path(density_path).is_file())
            self.assertTrue(Path(region_path).is_file())
            self.assertTrue(Path(confusion_path).is_file())

    def test_invalid_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = NaiveBayesVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "NaiveBayesResult"):
                visualizer.plot_class_priors("bad")
            with self.assertRaisesRegex(ValueError, "square"):
                visualizer.plot_confusion_matrix(np.array([[1.0, 2.0]]), np.array([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
