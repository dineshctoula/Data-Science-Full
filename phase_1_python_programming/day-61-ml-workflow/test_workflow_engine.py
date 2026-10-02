"""Tests for Day 61 — the floor model, and that validation is a real split."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflow_engine import (
    LogisticGD,
    MajorityClass,
    accuracy,
    compare_models,
    make_signup_data,
    pick_by_validation,
    train_val_split,
)


class SplitTests(unittest.TestCase):
    def test_split_uses_every_row_once(self):
        X, y, _ = make_signup_data(n=50, seed=1)
        Xtr, Xva, ytr, yva = train_val_split(X, y, val_frac=0.3, seed=1)
        self.assertEqual(len(Xtr) + len(Xva), len(X))
        self.assertEqual(len(ytr), len(Xtr))
        self.assertEqual(len(yva), len(Xva))

    def test_bad_fraction(self):
        X, y, _ = make_signup_data(n=20, seed=2)
        with self.assertRaisesRegex(ValueError, "val_frac"):
            train_val_split(X, y, val_frac=0)


class ModelTests(unittest.TestCase):
    def test_majority_is_constant(self):
        X, y, _ = make_signup_data(n=80, seed=3)
        model = MajorityClass().fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(np.unique(preds)), 1)
        self.assertGreater(accuracy(y, preds), 0.4)

    def test_logistic_beats_majority_on_train(self):
        X, y, _ = make_signup_data(n=300, seed=4)
        maj = MajorityClass().fit(X, y)
        log = LogisticGD(n_steps=400).fit(X, y)
        self.assertGreater(accuracy(y, log.predict(X)), accuracy(y, maj.predict(X)))

    def test_compare_returns_three_and_picks_a_real_one(self):
        X, y, _ = make_signup_data(n=220, seed=5)
        scores = compare_models(X, y, val_frac=0.3, seed=5)
        self.assertEqual([s.name for s in scores], ["majority", "nearest centroid", "logistic"])
        winner = pick_by_validation(scores)
        self.assertIn(winner.name, {s.name for s in scores})
        for s in scores:
            self.assertGreaterEqual(s.val_acc, 0.0)
            self.assertLessEqual(s.val_acc, 1.0)

    def test_bad_logistic(self):
        with self.assertRaisesRegex(ValueError, "lr"):
            LogisticGD(lr=0)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import WorkflowPlots

        X, y, names = make_signup_data(n=60, seed=6)
        scores = compare_models(X, y, seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            plots = WorkflowPlots(tmp)
            paths = [
                plots.accuracy_bars(scores),
                plots.visits_scatter(X, y, names),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())
                self.assertGreater(Path(p).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
