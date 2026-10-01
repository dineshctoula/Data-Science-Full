"""Tests for the Day 60 wrap-up. Mostly 'did the obvious signal survive?'"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from wrap_engine import (
    bootstrap_slope,
    column_summary,
    fit_ols,
    make_exam_scores,
    mse,
    r2_score,
    train_test_split,
)


class SummaryTests(unittest.TestCase):
    def test_hours_correlates_more_than_sleep(self):
        X, y, names = make_exam_scores(n=200, seed=1, noise=4)
        rows = {r["name"]: r for r in column_summary(X, y, names)}
        self.assertGreater(rows["hours"]["corr_with_y"], rows["sleep"]["corr_with_y"])
        self.assertGreater(rows["hours"]["corr_with_y"], 0.5)


class FitTests(unittest.TestCase):
    def test_hours_slope_near_truth(self):
        # low noise so the 4.2 coefficient isn't a coin flip
        X, y, names = make_exam_scores(n=400, seed=2, noise=1.5)
        fit = fit_ols(X, y, names)
        self.assertAlmostEqual(fit.coef[0], 4.2, delta=0.4)
        self.assertGreater(r2_score(y, fit.predict(X)), 0.9)

    def test_holdout_beats_predicting_the_mean(self):
        X, y, names = make_exam_scores(n=180, seed=3)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.25, seed=3)
        fit = fit_ols(Xtr, ytr, names)
        model_mse = mse(yte, fit.predict(Xte))
        mean_mse = mse(yte, np.full_like(yte, ytr.mean()))
        self.assertLess(model_mse, mean_mse)

    def test_bootstrap_interval_contains_the_point_estimate(self):
        X, y, names = make_exam_scores(n=120, seed=4)
        fit = fit_ols(X, y, names)
        boot = bootstrap_slope(X, y, feature=0, n_boot=80, seed=4)
        self.assertLess(boot["lo"], fit.coef[0])
        self.assertLess(fit.coef[0], boot["hi"])

    def test_bad_bootstrap(self):
        X, y, _ = make_exam_scores(n=30, seed=5)
        with self.assertRaisesRegex(ValueError, "n_boot"):
            bootstrap_slope(X, y, n_boot=5)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import WrapPlots

        X, y, names = make_exam_scores(n=80, seed=6)
        Xtr, Xte, ytr, yte = train_test_split(X, y, seed=6)
        fit = fit_ols(Xtr, ytr, names)
        boot = bootstrap_slope(X, y, n_boot=40, seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            plots = WrapPlots(tmp)
            paths = [
                plots.hours_vs_score(X, y, names),
                plots.residuals(yte, fit.predict(Xte)),
                plots.slope_hist(boot["samples"], boot["lo"], boot["hi"]),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())
                self.assertGreater(Path(p).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
