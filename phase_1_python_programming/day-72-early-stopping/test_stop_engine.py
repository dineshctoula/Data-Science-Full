"""Tests for Day 72 — the saved epoch should beat the last one on the holdout."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stop_engine import fit_until, log_loss, make_noisy_classes, three_way


class StopTests(unittest.TestCase):
    def test_split_does_not_overlap(self):
        train, val, test = three_way(100, n_train=40, n_val=30, seed=2)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(val) & set(test)), 0)
        self.assertEqual(len(train) + len(val) + len(test), 100)

    def test_log_loss_is_zero_on_a_sure_call(self):
        # clip keeps a perfect 0/1 from becoming exactly zero, but it should be tiny
        loss = log_loss(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        self.assertLess(loss, 1e-6)

    def test_best_epoch_is_the_val_minimum(self):
        X, y = make_noisy_classes(seed=72)
        result = fit_until(X, y, epochs=200, seed=72)
        self.assertEqual(result.epochs, 200)
        valley = int(np.argmin(result.val_loss)) + 1
        self.assertEqual(result.best_epoch, valley)
        self.assertLess(result.best_epoch, result.epochs)
        self.assertLess(result.val_loss[result.best_epoch - 1], result.val_loss[-1])
        self.assertLess(result.test_loss_best, result.test_loss_final)

    def test_bad_learning_rate_is_rejected(self):
        X, y = make_noisy_classes(n=50, n_noise=2, seed=1)
        with self.assertRaises(ValueError):
            fit_until(X, y, n_train=20, n_val=15, lr=0, epochs=2, seed=1)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import StopPlots

        with tempfile.TemporaryDirectory() as tmp:
            plots = StopPlots(tmp)
            paths = [
                plots.loss_curves([0.7, 0.4, 0.3], [0.7, 0.5, 0.8], best_epoch=2),
                plots.test_bars(0.55, 0.7),
            ]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
