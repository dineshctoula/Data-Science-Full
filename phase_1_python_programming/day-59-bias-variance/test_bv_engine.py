"""Tests for Day 59 — the stiff model should be biased, the wiggly one noisy."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bv_engine import decompose_degrees, draw_sample, fit_polynomial, predict_polynomial, true_curve


class FitTests(unittest.TestCase):
    def test_line_fit_on_a_line(self):
        rng = np.random.default_rng(1)
        x = np.linspace(-2, 2, 20)
        y = 1.5 * x - 0.3
        coef = fit_polynomial(x, y, degree=1)
        pred = predict_polynomial(coef, x)
        self.assertLess(np.max(np.abs(pred - y)), 1e-6)

    def test_rejects_degree_past_n(self):
        x, y = draw_sample(5, noise=0.1, rng=np.random.default_rng(2))
        with self.assertRaisesRegex(ValueError, "more rows"):
            fit_polynomial(x, y, degree=5)

    def test_truth_is_not_flat(self):
        xs = np.array([-2.0, 0.0, 2.0])
        ys = true_curve(xs)
        self.assertGreater(np.ptp(ys), 0.5)


class DecompositionTests(unittest.TestCase):
    def test_stiff_model_has_more_bias(self):
        pts = decompose_degrees([1, 8], n_train=30, n_runs=20, noise=0.4, seed=3)
        by_deg = {p.degree: p for p in pts}
        self.assertGreater(by_deg[1].bias2, by_deg[8].bias2)

    def test_wiggly_model_has_more_variance(self):
        pts = decompose_degrees([1, 8], n_train=28, n_runs=20, noise=0.4, seed=4)
        by_deg = {p.degree: p for p in pts}
        self.assertGreater(by_deg[8].variance, by_deg[1].variance)

    def test_mse_is_the_sum(self):
        pts = decompose_degrees([3], n_train=30, n_runs=12, noise=0.5, seed=5)
        p = pts[0]
        self.assertAlmostEqual(p.mse, p.bias2 + p.variance + p.noise, places=6)

    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "2 runs"):
            decompose_degrees([1], n_runs=1)
        with self.assertRaisesRegex(ValueError, "noise"):
            decompose_degrees([1], noise=-0.1)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from bv_engine import sample_fit_curves
        from visualizer import BiasVariancePlots

        pts = decompose_degrees([1, 3], n_train=24, n_runs=8, noise=0.3, seed=6)
        grid, truth, curves = sample_fit_curves(degree=1, n_curves=4, n_train=24, noise=0.3, seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            plots = BiasVariancePlots(tmp)
            paths = [
                plots.tradeoff(pts),
                plots.spaghetti(grid, truth, curves),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())
                self.assertGreater(Path(p).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
