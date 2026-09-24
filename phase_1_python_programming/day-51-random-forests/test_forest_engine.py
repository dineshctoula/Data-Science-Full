"""Tests for Day 51 forests — bagging quirks and the obvious regressions."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from forest_engine import (
    RandomForestClassifier,
    compare_n_trees,
    gini,
    make_noisy_moons,
    make_wide_features,
    train_test_split,
)


class HelperTests(unittest.TestCase):
    def test_gini_pure_and_mixed(self):
        self.assertEqual(gini([1, 1, 1]), 0.0)
        self.assertAlmostEqual(gini([0, 1, 0, 1]), 0.5)


class ForestTests(unittest.TestCase):
    def test_forest_beats_single_tree_on_noisy_moons(self):
        X, y = make_noisy_moons(n_per_class=70, noise=0.3, seed=3)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.3, seed=3)
        one = RandomForestClassifier(n_trees=1, max_depth=6, seed=3).fit(Xtr, ytr)
        many = RandomForestClassifier(n_trees=30, max_depth=6, seed=3).fit(Xtr, ytr)
        # not guaranteed every seed, but with this noise the ensemble should help
        self.assertGreaterEqual(many.score(Xte, yte).accuracy, one.score(Xte, yte).accuracy - 0.05)
        self.assertGreater(many.score(Xte, yte).accuracy, 0.7)

    def test_oob_is_in_unit_interval(self):
        X, y = make_noisy_moons(n_per_class=40, seed=4)
        forest = RandomForestClassifier(n_trees=15, seed=4).fit(X, y)
        oob = forest.oob_score(X, y)
        self.assertGreaterEqual(oob, 0.0)
        self.assertLessEqual(oob, 1.0)

    def test_wide_features_forest_learns_something(self):
        X, y = make_wide_features(n=120, p=6, seed=5)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.25, seed=5)
        forest = RandomForestClassifier(n_trees=40, max_depth=6, max_features="sqrt", seed=5).fit(Xtr, ytr)
        # soft-XOR on x0/x1 — should clear coin-flip by a decent margin
        self.assertGreater(forest.score(Xte, yte).accuracy, 0.7)

    def test_predict_shape(self):
        X, y = make_noisy_moons(n_per_class=25, seed=6)
        forest = RandomForestClassifier(n_trees=5, seed=6).fit(X, y)
        q = np.array([[0.0, 0.5], [1.0, 0.0]])
        self.assertEqual(forest.predict(q).shape, (2,))

    def test_compare_n_trees_keys(self):
        X, y = make_noisy_moons(n_per_class=35, seed=7)
        scores = compare_n_trees(X, y, n_list=(1, 5, 10), seed=7)
        self.assertEqual(set(scores), {1, 5, 10})

    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "n_trees"):
            RandomForestClassifier(n_trees=0)
        with self.assertRaisesRegex(ValueError, "max_depth"):
            RandomForestClassifier(max_depth=0)
        forest = RandomForestClassifier(n_trees=3)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            forest.predict([[1.0, 2.0]])


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import ForestPlots

        X, y = make_noisy_moons(n_per_class=30, seed=8)
        forest = RandomForestClassifier(n_trees=8, seed=8).fit(X, y)
        one = RandomForestClassifier(n_trees=1, seed=8).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            plots = ForestPlots(tmp)
            paths = [
                plots.decision_map(forest, X, y),
                plots.n_trees_curve({1: 0.6, 5: 0.75, 10: 0.8}),
                plots.single_vs_forest(X, y, one.predict(X), forest.predict(X)),
                plots.confusion_simple(y, forest.predict(X)),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())
                self.assertGreater(Path(p).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
