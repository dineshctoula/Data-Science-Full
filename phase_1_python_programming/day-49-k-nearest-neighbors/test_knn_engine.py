"""Tests for the Day 49 KNN stuff — nothing fancy, just the gotchas I hit while writing it."""

import unittest

import numpy as np

from knn_engine import (
    KNearestNeighbors,
    compare_k_values,
    leave_one_out_accuracy,
    make_blob_classes,
    make_two_moons_ish,
)


class KNNTests(unittest.TestCase):
    def test_perfect_fit_on_training_with_k1(self):
        # with k=1 every training point's nearest neighbor is itself
        X, y = make_blob_classes(n_per_class=20, seed=1)
        model = KNearestNeighbors(k=1).fit(X, y)
        result = model.score(X, y)
        self.assertEqual(result.accuracy, 1.0)

    def test_manhattan_and_euclidean_both_run(self):
        X, y = make_blob_classes(n_per_class=15, seed=2)
        eu = KNearestNeighbors(k=3, distance="euclidean").fit(X, y).score(X, y)
        man = KNearestNeighbors(k=3, distance="manhattan").fit(X, y).score(X, y)
        self.assertGreater(eu.accuracy, 0.8)
        self.assertGreater(man.accuracy, 0.8)

    def test_loo_rejects_k_equal_n(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        y = np.array([0.0, 1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "k must be < n"):
            leave_one_out_accuracy(X, y, k=3)

    def test_compare_k_returns_sorted_keys_we_asked_for(self):
        X, y = make_blob_classes(n_per_class=20, seed=3)
        scores = compare_k_values(X, y, ks=[1, 5, 9])
        self.assertEqual(set(scores), {1, 5, 9})
        for acc in scores.values():
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_moons_are_learnable(self):
        X, y = make_two_moons_ish(n_per_class=50, seed=49)
        acc = leave_one_out_accuracy(X, y, k=5)
        # moons are wiggly but still should clear ~90% with a decent k
        self.assertGreater(acc, 0.85)

    def test_predict_shape_matches_queries(self):
        X, y = make_blob_classes(n_per_class=12, seed=4)
        model = KNearestNeighbors(k=3).fit(X, y)
        q = np.array([[0.1, 0.2], [3.0, 2.0], [0.4, 3.5]])
        preds = model.predict(q)
        self.assertEqual(preds.shape, (3,))

    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            KNearestNeighbors(k=0)
        with self.assertRaisesRegex(ValueError, "euclidean or manhattan"):
            KNearestNeighbors(distance="cosine")
        model = KNearestNeighbors(k=2)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            model.predict([[1.0, 2.0]])


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        import tempfile
        from pathlib import Path

        from visualizer import KNNPlots

        X, y = make_blob_classes(n_per_class=18, seed=5)
        model = KNearestNeighbors(k=3).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            plots = KNNPlots(tmp)
            paths = [
                plots.decision_map(model, X, y),
                plots.k_sweep_bars({1: 0.9, 3: 0.95, 5: 0.93}),
                plots.neighbor_example(model, X, y, X[0]),
                plots.confusion_simple(y, model.predict(X)),
            ]
            for p in paths:
                self.assertTrue(Path(p).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
