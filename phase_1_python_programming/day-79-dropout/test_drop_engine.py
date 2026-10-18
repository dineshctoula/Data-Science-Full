"""Tests for Day 79 — dropout should loosen a memorized fit, not wreck it."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from drop_engine import DropTrial, TinyNet, make_curve, mse, pick_by_val, sweep


class NetTests(unittest.TestCase):
    def test_bad_rate_is_rejected(self):
        for drop in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                TinyNet(drop=drop)

    def test_predict_before_fit_is_rejected(self):
        with self.assertRaises(RuntimeError):
            TinyNet().predict([0.0])

    def test_same_seed_same_net(self):
        x = np.linspace(-1, 1, 12)
        y = np.sin(x)
        a = TinyNet(drop=0.2, epochs=300, seed=3).fit(x, y).predict(x)
        b = TinyNet(drop=0.2, epochs=300, seed=3).fit(x, y).predict(x)
        np.testing.assert_allclose(a, b)

    def test_runaway_step_raises(self):
        x = np.linspace(-2, 2, 16)
        with self.assertRaises(FloatingPointError):
            with np.errstate(all="ignore"):
                TinyNet(lr=5.0, epochs=300).fit(x, np.sin(x) * 10)

    def test_more_dropout_fits_the_training_rows_worse(self):
        data = make_curve(seed=79)
        (x_tr, y_tr), _, _ = data
        plain = TinyNet(drop=0.0, seed=0).fit(x_tr, y_tr)
        heavy = TinyNet(drop=0.5, seed=0).fit(x_tr, y_tr)
        self.assertLess(mse(y_tr, plain.predict(x_tr)), mse(y_tr, heavy.predict(x_tr)))


class SweepTests(unittest.TestCase):
    def test_light_dropout_wins_with_a_big_validation_slice(self):
        data = make_curve(n_val=200, seed=79)
        trials = sweep(data, [0.0, 0.1], n_seeds=5)
        plain, light = trials
        self.assertEqual(pick_by_val(trials).drop, 0.1)
        self.assertLess(light.test_mse, plain.test_mse)

    def test_pick_ignores_the_test_column(self):
        trials = [DropTrial(0.0, 0.1, 0.30, 0.01, 0.02), DropTrial(0.1, 0.2, 0.20, 0.90, 0.95)]
        self.assertEqual(pick_by_val(trials).drop, 0.1)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import DropPlots

        x = np.linspace(-2, 2, 5)
        trials = [DropTrial(0.0, 0.1, 0.2, 0.22, 0.25), DropTrial(0.1, 0.15, 0.17, 0.18, 0.2)]
        with tempfile.TemporaryDirectory() as tmp:
            plots = DropPlots(tmp)
            paths = [plots.fits(x, np.sin(x), [("a", x, np.sin(x))]), plots.mse_bars(trials)]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
