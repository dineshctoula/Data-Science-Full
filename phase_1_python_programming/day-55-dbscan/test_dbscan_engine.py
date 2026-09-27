"""Tests for Day 55 DBSCAN — moons should split, junk should stay noise."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dbscan_engine import NOISE, DBSCAN, make_blobs_with_outliers, make_two_moons


class DBSCANTests(unittest.TestCase):
    def test_moons_find_two_clusters(self):
        X, y = make_two_moons(n_per=50, noise=0.05, seed=1)
        model = DBSCAN(eps=0.25, min_samples=4).fit(X)
        res = model.result()
        self.assertEqual(res.n_clusters, 2)
        self.assertLess(res.n_noise, 8)

    def test_outliers_mostly_marked_noise(self):
        X, y = make_blobs_with_outliers(n_per=35, n_out=10, seed=2)
        model = DBSCAN(eps=0.45, min_samples=5).fit(X)
        # true junk rows
        junk = y == NOISE
        pred_junk = model.labels_[junk]
        # most of the planted outliers should not join a dense blob
        self.assertGreater(np.mean(pred_junk == NOISE), 0.5)
        self.assertGreaterEqual(model.result().n_clusters, 2)

    def test_tiny_eps_is_all_noise(self):
        X, _ = make_two_moons(n_per=30, seed=3)
        model = DBSCAN(eps=0.01, min_samples=5).fit(X)
        self.assertEqual(model.result().n_clusters, 0)
        self.assertEqual(model.result().n_noise, len(X))

    def test_labels_length(self):
        X, _ = make_two_moons(n_per=20, seed=4)
        labels = DBSCAN(eps=0.3, min_samples=3).fit_predict(X)
        self.assertEqual(len(labels), len(X))

    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "eps"):
            DBSCAN(eps=0)
        with self.assertRaisesRegex(ValueError, "min_samples"):
            DBSCAN(min_samples=0)
        model = DBSCAN()
        with self.assertRaisesRegex(RuntimeError, "fit"):
            model.result()


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import DBSCANPlots

        X, _ = make_two_moons(n_per=25, seed=5)
        model = DBSCAN(eps=0.3, min_samples=4).fit(X)
        with tempfile.TemporaryDirectory() as tmp:
            plots = DBSCANPlots(tmp)
            paths = [
                plots.clusters(model, X),
                plots.eps_sweep([(0.1, 0, 50), (0.3, 2, 3), (0.8, 1, 0)]),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())


if __name__ == "__main__":
    unittest.main()
