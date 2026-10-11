"""Tests for Day 70 — shuffling a real column should hurt, noise should not."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from perm_engine import fit_ols, make_table, permutation_rises, predict_ols, split_indices


class PermTests(unittest.TestCase):
    def test_line_fit_recovers_a_slope(self):
        x = np.linspace(-1, 1, 25)
        X = x.reshape(-1, 1)
        y = 0.5 + 3.0 * x
        coef = fit_ols(X, y)
        self.assertAlmostEqual(coef[1], 3.0, places=6)
        pred = predict_ols(X, coef)
        np.testing.assert_allclose(pred, y, atol=1e-8)

    def test_split_does_not_overlap(self):
        train, test = split_indices(50, test_frac=0.3, seed=2)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(train) + len(test), 50)

    def test_signal_columns_rise_more_than_noise(self):
        X, y = make_table(n=240, seed=70)
        train, val = split_indices(len(X), test_frac=0.3, seed=70)
        base, coef, rises = permutation_rises(X[train], y[train], X[val], y[val], repeats=20, seed=70)
        self.assertGreater(base, 0)
        by_index = {item.index: item.mean_rise for item in rises}
        self.assertGreater(by_index[0], 2.0)
        self.assertGreater(by_index[1], 1.0)
        self.assertLess(by_index[2], 0.15)
        self.assertLess(by_index[3], 0.15)
        # the fit itself is the training fit; slopes should keep their signs
        self.assertGreater(coef[1], 0)
        self.assertLess(coef[2], 0)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from perm_engine import ColumnRise
        from visualizer import PermPlots

        rises = [ColumnRise(0, 4.0, 0.4), ColumnRise(1, 0.02, 0.01)]
        with tempfile.TemporaryDirectory() as tmp:
            plots = PermPlots(tmp)
            paths = [plots.rise_bars(rises), plots.coef_bars(np.array([0.1, 2.0, -1.0]))]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
