"""Tests for Day 53 linear SVM."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from svm_engine import LinearSVM, hinge, make_linearly_separable, make_soft_overlap, train_test_split


class HingeTests(unittest.TestCase):
    def test_hinge_zero_when_past_margin(self):
        self.assertEqual(hinge(1.0, 2.0), 0.0)
        self.assertEqual(hinge(-1.0, -2.0), 0.0)

    def test_hinge_positive_when_inside(self):
        self.assertAlmostEqual(hinge(1.0, 0.0), 1.0)
        self.assertAlmostEqual(hinge(1.0, 0.5), 0.5)


class SVMTests(unittest.TestCase):
    def test_separable_almost_perfect(self):
        X, y = make_linearly_separable(n_per=35, seed=1)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.3, seed=1)
        model = LinearSVM(C=1.0, lr=0.05, n_epochs=50, seed=1).fit(Xtr, ytr)
        self.assertGreater(model.score(Xte, yte).accuracy, 0.9)

    def test_soft_overlap_beats_chance(self):
        X, y = make_soft_overlap(n_per=45, seed=2)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.3, seed=2)
        model = LinearSVM(C=2.0, lr=0.05, n_epochs=60, seed=2).fit(Xtr, ytr)
        self.assertGreater(model.score(Xte, yte).accuracy, 0.7)

    def test_loss_tends_to_drop(self):
        X, y = make_linearly_separable(n_per=30, seed=3)
        model = LinearSVM(C=1.0, lr=0.05, n_epochs=40, seed=3).fit(X, y)
        # not monotonic, but last half should be better than the start on average
        early = np.mean(model.loss_curve_[:5])
        late = np.mean(model.loss_curve_[-5:])
        self.assertLess(late, early)

    def test_margin_mask_length(self):
        X, y = make_linearly_separable(n_per=20, seed=4)
        model = LinearSVM(n_epochs=30, seed=4).fit(X, y)
        mask = model.margin_mask()
        self.assertEqual(len(mask), len(X))

    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "C"):
            LinearSVM(C=0)
        with self.assertRaisesRegex(ValueError, "lr"):
            LinearSVM(lr=-0.1)
        model = LinearSVM()
        with self.assertRaisesRegex(RuntimeError, "fit"):
            model.predict([[1.0, 2.0]])


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import SVMPlots

        X, y = make_linearly_separable(n_per=22, seed=5)
        model = LinearSVM(n_epochs=25, seed=5).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            plots = SVMPlots(tmp)
            paths = [
                plots.decision_boundary(model, X, y),
                plots.loss_curve(model.loss_curve_),
                plots.c_sweep_bars({0.1: 0.8, 1.0: 0.9, 10: 0.85}),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())


if __name__ == "__main__":
    unittest.main()
