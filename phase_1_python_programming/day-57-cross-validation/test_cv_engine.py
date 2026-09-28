"""Tests for Day 57 folds — sizes, balance, and the obvious score bounds."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cv_engine import (
    NearestCentroid,
    cross_val_accuracy,
    kfold_indices,
    make_two_class_cloud,
    stratified_kfold_indices,
)


class SplitTests(unittest.TestCase):
    def test_kfold_covers_every_row_once(self):
        n, k = 23, 5
        seen = []
        for train, test in kfold_indices(n, k=k, seed=1):
            self.assertEqual(len(train) + len(test), n)
            self.assertEqual(len(np.intersect1d(train, test)), 0)
            seen.append(test)
        all_test = np.concatenate(seen)
        self.assertEqual(sorted(all_test.tolist()), list(range(n)))

    def test_stratified_keeps_class_mix(self):
        y = np.array([0] * 30 + [1] * 15)
        for _, test in stratified_kfold_indices(y, k=5, seed=2):
            # each fold should see both classes
            self.assertGreaterEqual(np.unique(y[test]).size, 2)
            # not wildly lopsided vs the 2:1 base rate
            frac = np.mean(y[test] == 0)
            self.assertGreater(frac, 0.4)
            self.assertLess(frac, 0.9)

    def test_bad_k(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            list(kfold_indices(10, k=1))
        with self.assertRaisesRegex(ValueError, "smaller"):
            list(kfold_indices(3, k=5))


class ScoreTests(unittest.TestCase):
    def test_easy_cloud_scores_high(self):
        X, y = make_two_class_cloud(n_per=40, sep=2.4, seed=3)
        result = cross_val_accuracy(X, y, k=5, stratified=True, seed=3)
        self.assertEqual(len(result.scores), 5)
        self.assertGreater(result.mean, 0.8)

    def test_centroid_fit_predict(self):
        X, y = make_two_class_cloud(n_per=20, sep=3.0, seed=4)
        model = NearestCentroid().fit(X, y)
        acc = float(np.mean(model.predict(X) == y))
        self.assertGreater(acc, 0.85)

    def test_plain_and_stratified_both_run(self):
        X, y = make_two_class_cloud(n_per=25, seed=5)
        a = cross_val_accuracy(X, y, k=4, stratified=False, seed=5)
        b = cross_val_accuracy(X, y, k=4, stratified=True, seed=5)
        self.assertEqual(a.k, 4)
        self.assertTrue(b.stratified)
        for s in list(a.scores) + list(b.scores):
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import CVPlots

        with tempfile.TemporaryDirectory() as tmp:
            plots = CVPlots(tmp)
            paths = [
                plots.fold_bars([0.8, 0.75, 0.9, 0.85, 0.82]),
                plots.holdout_vs_cv([0.7, 0.9, 0.6, 0.85], [0.8, 0.82, 0.78, 0.81, 0.79]),
                plots.k_sweep([2, 5, 10], [0.7, 0.8, 0.78], [0.1, 0.05, 0.04]),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())


if __name__ == "__main__":
    unittest.main()
