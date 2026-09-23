"""Tests for Day 50 trees — mostly the edge cases that bit me while coding."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tree_engine import (
    DecisionTreeClassifier,
    entropy,
    gini,
    make_axis_aligned_blobs,
    make_xor_like,
    majority_label,
    train_test_split,
)


class ImpurityTests(unittest.TestCase):
    def test_pure_bucket_is_zero(self):
        self.assertEqual(gini([1, 1, 1, 1]), 0.0)
        self.assertEqual(entropy([0, 0, 0]), 0.0)

    def test_fifty_fifty_gini_is_half(self):
        # two classes equal → gini = 1 - 2*(0.5^2) = 0.5
        self.assertAlmostEqual(gini([0, 0, 1, 1]), 0.5)

    def test_majority_picks_the_winner(self):
        self.assertEqual(majority_label([0, 1, 1, 1, 0]), 1)


class TreeFitTests(unittest.TestCase):
    def test_blobs_almost_perfect_with_shallow_tree(self):
        X, y = make_axis_aligned_blobs(n_per_class=35, seed=7)
        model = DecisionTreeClassifier(max_depth=3, criterion="gini").fit(X, y)
        result = model.score(X, y)
        self.assertGreater(result.accuracy, 0.95)
        self.assertGreaterEqual(result.n_leaves, 2)

    def test_xor_needs_more_than_depth_one(self):
        X, y = make_xor_like(n_per_class=40, seed=8)
        shallow = DecisionTreeClassifier(max_depth=1).fit(X, y)
        # axis-aligned trees need a few cuts to carve out XOR corners
        deeper = DecisionTreeClassifier(max_depth=6).fit(X, y)
        self.assertLess(shallow.score(X, y).accuracy, 0.75)
        self.assertGreater(deeper.score(X, y).accuracy, 0.95)

    def test_entropy_criterion_also_runs(self):
        X, y = make_axis_aligned_blobs(n_per_class=20, seed=9)
        model = DecisionTreeClassifier(max_depth=4, criterion="entropy").fit(X, y)
        self.assertGreater(model.score(X, y).accuracy, 0.9)

    def test_predict_shape(self):
        X, y = make_axis_aligned_blobs(n_per_class=15, seed=10)
        model = DecisionTreeClassifier(max_depth=3).fit(X, y)
        q = np.array([[1.0, 1.0], [3.0, 3.0], [2.0, 2.0]])
        preds = model.predict(q)
        self.assertEqual(preds.shape, (3,))

    def test_train_test_split_sizes(self):
        X, y = make_axis_aligned_blobs(n_per_class=20, seed=11)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.25, seed=11)
        self.assertEqual(len(Xtr) + len(Xte), len(X))
        self.assertEqual(len(ytr), len(Xtr))
        self.assertEqual(len(yte), len(Xte))

    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "max_depth"):
            DecisionTreeClassifier(max_depth=0)
        with self.assertRaisesRegex(ValueError, "gini or entropy"):
            DecisionTreeClassifier(criterion="mse")
        model = DecisionTreeClassifier()
        with self.assertRaisesRegex(RuntimeError, "fit"):
            model.predict([[1.0, 2.0]])


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import TreePlots

        X, y = make_axis_aligned_blobs(n_per_class=18, seed=12)
        model = DecisionTreeClassifier(max_depth=3).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            plots = TreePlots(tmp)
            paths = [
                plots.decision_map(model, X, y),
                plots.depth_sweep([1, 2, 3], [0.7, 0.9, 1.0], [0.65, 0.85, 0.8]),
                plots.impurity_bars(y, y[:10], y[10:]),
                plots.confusion_simple(y, model.predict(X)),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())
                self.assertGreater(Path(p).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
