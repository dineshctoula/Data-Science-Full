"""Tests for Day 73 — a small step lags, a huge step runs away."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rate_engine import fit_rate, make_line, mse, split_indices


class RateTests(unittest.TestCase):
    def test_split_does_not_overlap(self):
        train, test = split_indices(50, test_frac=0.3, seed=2)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(train) + len(test), 50)

    def test_zero_rate_is_rejected(self):
        X, y = make_line(n=40, seed=1)
        with self.assertRaises(ValueError):
            fit_rate(X, y, lr=0, epochs=2, seed=1)

    def test_slow_lags_and_large_diverges(self):
        X, y = make_line(seed=73)
        slow = fit_rate(X, y, lr=0.02, epochs=40, seed=73)
        good = fit_rate(X, y, lr=0.2, epochs=40, seed=73)
        wild = fit_rate(X, y, lr=1.9, epochs=40, seed=73)
        self.assertEqual(slow.epochs, 40)
        # still above the settled fit
        self.assertGreater(slow.train_mse[-1], good.train_mse[-1] * 2)
        self.assertLess(good.test_mse[-1], 1.0)
        # the big step is worse at the end than after its first update
        self.assertGreater(wild.train_mse[-1], wild.train_mse[0] * 100)
        self.assertGreater(wild.test_mse[-1], good.test_mse[-1])

    def test_mse_matches_a_hand_check(self):
        self.assertAlmostEqual(mse(np.array([1.0, 3.0]), np.array([1.0, 1.0])), 2.0)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from rate_engine import RateRun
        from visualizer import RatePlots

        run = RateRun(0.2, np.array([2.0, 1.0]), np.array([2.2, 1.1]), np.zeros(3))
        with tempfile.TemporaryDirectory() as tmp:
            plots = RatePlots(tmp)
            paths = [plots.train_curves([run]), plots.final_bars([run])]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
