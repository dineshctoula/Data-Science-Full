"""Tests for Day 58 — shrinkage, zeros, and the obvious input errors."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from reg_engine import (
    coefficient_path,
    fit_lasso,
    fit_ridge,
    make_sparse_regression,
    soft_threshold,
    train_test_split,
)


class SoftThresholdTests(unittest.TestCase):
    def test_kills_small_values(self):
        self.assertEqual(soft_threshold(0.2, 0.5), 0.0)
        self.assertEqual(soft_threshold(-0.2, 0.5), 0.0)

    def test_shrinks_large_values(self):
        self.assertAlmostEqual(soft_threshold(1.5, 0.5), 1.0)
        self.assertAlmostEqual(soft_threshold(-1.5, 0.5), -1.0)


class FitTests(unittest.TestCase):
    def test_ols_recovers_slope(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(80, 1))
        y = 2.5 * x[:, 0] + 0.4
        fit = fit_ridge(x, y, lam=0.0)
        self.assertAlmostEqual(fit.coef[0], 2.5, places=1)
        self.assertAlmostEqual(fit.intercept, 0.4, places=1)

    def test_ridge_shrinks_compared_with_ols(self):
        X, y, _ = make_sparse_regression(n=100, seed=2)
        ols = fit_ridge(X, y, lam=0.0)
        ridge = fit_ridge(X, y, lam=20.0)
        self.assertLess(np.sum(ridge.coef ** 2), np.sum(ols.coef ** 2))

    def test_lasso_zeros_noise_columns(self):
        X, y, _ = make_sparse_regression(n=160, seed=3)
        fit = fit_lasso(X, y, lam=8.0)
        # last three columns are pure noise — should be gone at this lambda
        self.assertTrue(np.all(np.abs(fit.coef[3:]) < 1e-6))
        # the real signals should still be there
        self.assertGreater(abs(fit.coef[0]) + abs(fit.coef[1]), 1.0)

    def test_path_shape(self):
        X, y, _ = make_sparse_regression(n=60, seed=4)
        lams = [0.1, 1.0, 10.0]
        path = coefficient_path(X, y, lams, kind="ridge")
        self.assertEqual(path.shape, (3, X.shape[1]))

    def test_bad_inputs(self):
        X = np.ones((10, 2))
        y = np.arange(10, dtype=float)
        with self.assertRaisesRegex(ValueError, "lambda"):
            fit_ridge(X, y, lam=-1)
        with self.assertRaisesRegex(ValueError, "lambda"):
            fit_lasso(X, y, lam=-0.2)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            fit_ridge(X, y[:4], lam=1)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import RegPlots

        X, y, names = make_sparse_regression(n=40, seed=5)
        lams = [0.5, 2, 8]
        path = coefficient_path(X, y, lams, kind="lasso")
        Xtr, Xte, ytr, yte = train_test_split(X, y, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            plots = RegPlots(tmp)
            paths = [
                plots.coef_path(lams, path, names),
                plots.mse_vs_lambda(lams, [1.2, 0.9, 1.1], [1.3, 0.8, 1.4]),
                plots.coef_bars(names, fit_ridge(Xtr, ytr, 0).coef, fit_ridge(Xtr, ytr, 5).coef, fit_lasso(Xtr, ytr, 5).coef),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())
                self.assertGreater(Path(p).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
