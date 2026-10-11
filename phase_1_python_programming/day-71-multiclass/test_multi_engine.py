"""Tests for Day 71 — one-vs-rest should beat a majority guess on separated blobs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from multi_engine import OneVsRest, confusion_matrix, evaluate, make_blobs, split_indices


class MultiTests(unittest.TestCase):
    def test_split_does_not_overlap(self):
        train, test = split_indices(60, test_frac=0.3, seed=3)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(train) + len(test), 60)

    def test_confusion_counts_the_off_diagonal(self):
        matrix = confusion_matrix(np.array([0, 1, 1]), np.array([0, 0, 1]), n_classes=2)
        self.assertEqual(int(matrix[0, 0]), 1)
        self.assertEqual(int(matrix[1, 0]), 1)
        self.assertEqual(int(matrix[1, 1]), 1)

    def test_predict_before_fit_is_rejected(self):
        with self.assertRaises(RuntimeError):
            OneVsRest().predict(np.zeros((2, 2)))

    def test_holdout_beats_majority(self):
        X, y = make_blobs(seed=71)
        result = evaluate(X, y, seed=71)
        self.assertGreater(result.accuracy, result.majority_accuracy + 0.3)
        self.assertEqual(result.confusion.shape, (3, 3))
        # class 2 is the weak one on this seed, but it is not a total miss
        self.assertGreater(result.recall(2), 0.5)
        self.assertLess(result.recall(2), result.recall(1))


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import MultiPlots

        with tempfile.TemporaryDirectory() as tmp:
            plots = MultiPlots(tmp)
            paths = [
                plots.confusion_heatmap(np.array([[3, 1], [0, 4]])),
                plots.recall_bars([0.8, 0.6, 0.7]),
            ]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
