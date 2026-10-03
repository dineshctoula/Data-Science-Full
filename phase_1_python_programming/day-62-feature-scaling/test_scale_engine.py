"""Tests for Day 62 — scalers, and that raw kNN gets fooled by income."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scale_engine import (
    MinMaxScaler,
    StandardScaler,
    compare_scalers,
    make_age_income,
)


class ScalerTests(unittest.TestCase):
    def test_standard_centers_and_spreads(self):
        rng = np.random.default_rng(1)
        X = np.column_stack([rng.normal(10, 2, 80), rng.normal(500, 40, 80)])
        Z = StandardScaler().fit_transform(X)
        self.assertTrue(np.allclose(Z.mean(axis=0), 0, atol=1e-8))
        self.assertTrue(np.allclose(Z.std(axis=0), 1, atol=1e-8))

    def test_minmax_sits_in_unit_interval(self):
        X = np.array([[0.0, 10.0], [5.0, 30.0], [2.0, 20.0]])
        Z = MinMaxScaler().fit_transform(X)
        self.assertTrue(np.all(Z >= -1e-9))
        self.assertTrue(np.all(Z <= 1 + 1e-9))
        self.assertAlmostEqual(Z[:, 0].min(), 0.0)
        self.assertAlmostEqual(Z[:, 0].max(), 1.0)

    def test_inverse_roundtrip(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 3)) * np.array([1, 50, 0.01])
        scaler = StandardScaler().fit(X)
        back = scaler.inverse_transform(scaler.transform(X))
        self.assertTrue(np.allclose(back, X))

    def test_constant_column_does_not_explode(self):
        X = np.column_stack([np.ones(12), np.arange(12, dtype=float)])
        Z = StandardScaler().fit_transform(X)
        self.assertTrue(np.isfinite(Z).all())
        self.assertTrue(np.allclose(Z[:, 0], 0))


class KNNTests(unittest.TestCase):
    def test_scaled_knn_beats_raw(self):
        X, y, _ = make_age_income(n=240, seed=3)
        rows = {r.name: r.val_acc for r in compare_scalers(X, y, k=5, seed=3)}
        self.assertGreater(rows["standard"], rows["raw"])
        self.assertGreater(rows["standard"], 0.6)

    def test_compare_names(self):
        X, y, _ = make_age_income(n=80, seed=4)
        names = [r.name for r in compare_scalers(X, y, seed=4)]
        self.assertEqual(names, ["raw", "standard", "minmax"])


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import ScalePlots

        X, y, names = make_age_income(n=60, seed=5)
        Z = StandardScaler().fit_transform(X)
        rows = compare_scalers(X, y, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            plots = ScalePlots(tmp)
            paths = [plots.before_after(X, Z, names), plots.accuracy_bars(rows)]
            for p in paths:
                self.assertTrue(Path(p).exists())
                self.assertGreater(Path(p).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
