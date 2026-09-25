"""Tests for Day 52 boosting — residuals should shrink, clf should separate blobs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from boost_engine import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RegressionTree,
    make_regression_wave,
    make_two_blobs_clf,
    train_test_split,
)


class TreeTests(unittest.TestCase):
    def test_stump_fits_step(self):
        # left side ~0, right side ~1
        X = np.array([[0.0], [0.5], [1.5], [2.0]])
        r = np.array([0.0, 0.1, 0.9, 1.0])
        tree = RegressionTree(max_depth=1).fit(X, r)
        preds = tree.predict(X)
        self.assertLess(preds[0], 0.4)
        self.assertGreater(preds[-1], 0.6)


class RegressorTests(unittest.TestCase):
    def test_loss_drops_over_rounds(self):
        X, y = make_regression_wave(n=100, noise=0.2, seed=1)
        model = GradientBoostingRegressor(n_estimators=30, learning_rate=0.15, max_depth=2).fit(X, y)
        self.assertLess(model.train_loss_[-1], model.train_loss_[0])
        self.assertLess(model.score(X, y).mse, 0.5)

    def test_predict_shape(self):
        X, y = make_regression_wave(n=40, seed=2)
        model = GradientBoostingRegressor(n_estimators=10).fit(X, y)
        self.assertEqual(model.predict(X[:5]).shape, (5,))

    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "n_estimators"):
            GradientBoostingRegressor(n_estimators=0)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            GradientBoostingRegressor(learning_rate=0)
        model = GradientBoostingRegressor()
        with self.assertRaisesRegex(RuntimeError, "fit"):
            model.predict([[1.0]])


class ClassifierTests(unittest.TestCase):
    def test_blobs_mostly_correct(self):
        X, y = make_two_blobs_clf(n_per=40, seed=3)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.3, seed=3)
        clf = GradientBoostingClassifier(n_estimators=40, learning_rate=0.2, max_depth=2).fit(Xtr, ytr)
        self.assertGreater(clf.score(Xte, yte).accuracy, 0.85)

    def test_rejects_multiclass(self):
        X = np.random.default_rng(0).normal(size=(30, 2))
        y = np.array([0, 1, 2] * 10)
        with self.assertRaisesRegex(ValueError, "two classes"):
            GradientBoostingClassifier().fit(X, y)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import BoostPlots

        X, y = make_regression_wave(n=50, seed=4)
        reg = GradientBoostingRegressor(n_estimators=15).fit(X, y)
        Xc, yc = make_two_blobs_clf(n_per=25, seed=4)
        clf = GradientBoostingClassifier(n_estimators=15).fit(Xc, yc)
        with tempfile.TemporaryDirectory() as tmp:
            plots = BoostPlots(tmp)
            paths = [
                plots.loss_curve(reg.train_loss_),
                plots.regression_fit(reg, X, y),
                plots.decision_map(clf, Xc, yc),
                plots.lr_compare({"0.05": [1, 0.8, 0.6], "0.2": [1, 0.5, 0.3]}),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())


if __name__ == "__main__":
    unittest.main()
