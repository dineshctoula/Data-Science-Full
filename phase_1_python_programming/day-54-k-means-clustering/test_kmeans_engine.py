"""Tests for Day 54 k-means — mostly 'does it find the obvious piles?'"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kmeans_engine import KMeans, elbow_scores, inertia, make_blobs, make_uneven_blobs


class HelperTests(unittest.TestCase):
    def test_inertia_zero_when_on_centers(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        labels = np.array([0, 1])
        centers = X.copy()
        self.assertAlmostEqual(inertia(X, labels, centers), 0.0)


class KMeansTests(unittest.TestCase):
    def test_recovers_three_blobs(self):
        X, y = make_blobs(k=3, n_per=35, seed=1)
        km = KMeans(k=3, n_init=8, seed=1).fit(X)
        # label ids won't match y, so check pairwise agreement roughly via inertia
        self.assertLess(km.inertia_, 80.0)
        self.assertEqual(len(np.unique(km.labels_)), 3)

    def test_predict_matches_fit_labels(self):
        X, _ = make_blobs(k=2, n_per=25, seed=2)
        km = KMeans(k=2, seed=2).fit(X)
        self.assertTrue(np.array_equal(km.predict(X), km.labels_))

    def test_elbow_decreases_overall(self):
        X, _ = make_blobs(k=3, n_per=30, seed=3)
        scores = elbow_scores(X, ks=range(1, 6), seed=3)
        # more clusters → lower (or equal) inertia in theory; allow tiny noise
        self.assertLess(scores[5], scores[1])

    def test_uneven_still_runs(self):
        X, _ = make_uneven_blobs(seed=4)
        km = KMeans(k=3, n_init=6, seed=4).fit(X)
        self.assertEqual(km.centers_.shape, (3, 2))
        self.assertGreater(km.inertia_, 0.0)

    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "k should"):
            KMeans(k=0)
        with self.assertRaisesRegex(ValueError, "n_init"):
            KMeans(n_init=0)
        km = KMeans(k=2)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            km.predict([[1.0, 2.0]])
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "bigger than"):
            KMeans(k=5).fit(X)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import KMeansPlots

        X, y = make_blobs(k=3, n_per=20, seed=5)
        km = KMeans(k=3, seed=5).fit(X)
        with tempfile.TemporaryDirectory() as tmp:
            plots = KMeansPlots(tmp)
            paths = [
                plots.clusters(km, X),
                plots.elbow({1: 100, 2: 40, 3: 15}),
                plots.inertia_curve(km.inertia_curve_),
                plots.truth_vs_pred(X, y, km.labels_),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())


if __name__ == "__main__":
    unittest.main()
