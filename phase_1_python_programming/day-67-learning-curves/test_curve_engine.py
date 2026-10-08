"""Tests for Day 67 — more rows close a flexible model's gap, not a stiff one's error."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from curve_engine import fit_poly, learning_curve, make_sine, poly_design, predict_poly


class PolyTests(unittest.TestCase):
    def test_design_starts_with_ones(self):
        design = poly_design(np.array([2.0, 3.0]), degree=1)
        self.assertEqual(design.shape, (2, 2))
        np.testing.assert_allclose(design[:, 0], [1.0, 1.0])
        np.testing.assert_allclose(design[:, 1], [2.0, 3.0])

    def test_line_fit_recovers_the_slope(self):
        x = np.linspace(-1, 1, 30)
        y = 1.5 + 2.0 * x
        beta = fit_poly(x, y, degree=1)
        pred = predict_poly(x, beta)
        np.testing.assert_allclose(pred, y, atol=1e-8)
        self.assertAlmostEqual(beta[1], 2.0, places=6)

    def test_too_small_a_size_is_rejected(self):
        x, y = make_sine(n=80, seed=1)
        with self.assertRaises(ValueError):
            learning_curve(x, y, degree=4, sizes=[3], n_test=20, repeats=2, seed=1)


class CurveTests(unittest.TestCase):
    def test_flexible_gap_shrinks_and_beats_the_line(self):
        x, y = make_sine(n=500, noise=0.4, seed=67)
        sizes = [40, 80, 160, 320]
        line = learning_curve(x, y, degree=1, sizes=sizes, n_test=100, repeats=16, seed=67)
        bend = learning_curve(x, y, degree=4, sizes=sizes, n_test=100, repeats=16, seed=67)
        self.assertEqual(len(line.sizes), 4)
        # degree 4 starts optimistic and the test number catches up
        self.assertGreater(bend.gap()[0], bend.gap()[-1])
        self.assertGreater(line.test_mse[-1], bend.test_mse[-1])


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import CurvePlots
        from curve_engine import LearningCurve

        sizes = np.array([20, 40])
        stiff = LearningCurve(1, sizes, np.array([0.3, 0.28]), np.array([0.31, 0.29]))
        flex = LearningCurve(4, sizes, np.array([0.1, 0.14]), np.array([0.22, 0.16]))
        with tempfile.TemporaryDirectory() as tmp:
            plots = CurvePlots(tmp)
            paths = [plots.two_curves(stiff, flex), plots.gap_lines([stiff, flex])]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
