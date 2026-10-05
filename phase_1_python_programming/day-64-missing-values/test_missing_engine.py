"""Tests for Day 64 — fill values, indicators, and not using the holdout to fill."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from missing_engine import (
    SimpleImputer,
    compare_impute,
    drop_incomplete,
    make_study_table,
    missing_report,
)


class ImputerTests(unittest.TestCase):
    def test_mean_fill_uses_observed_only(self):
        X = np.array([[1.0, 10.0], [np.nan, 30.0], [3.0, np.nan]])
        filled = SimpleImputer(strategy="mean").fit_transform(X)
        self.assertAlmostEqual(filled[1, 0], 2.0)  # mean of 1 and 3
        self.assertAlmostEqual(filled[2, 1], 20.0)  # mean of 10 and 30
        self.assertAlmostEqual(filled[0, 0], 1.0)

    def test_median_ignores_the_outlier_a_bit(self):
        X = np.array([[1.0], [np.nan], [3.0], [100.0]])
        filled = SimpleImputer(strategy="median").fit_transform(X)
        self.assertAlmostEqual(filled[1, 0], 3.0)

    def test_indicator_marks_the_blank(self):
        X = np.array([[1.0, 5.0], [np.nan, 5.0]])
        filled = SimpleImputer(strategy="mean", add_indicator=True).fit_transform(X)
        self.assertEqual(filled.shape, (2, 4))
        self.assertEqual(filled[1, 2], 1.0)  # hours was missing
        self.assertEqual(filled[1, 3], 0.0)  # sleep was not

    def test_drop_removes_any_blank_row(self):
        X = np.array([[1.0, 2.0], [np.nan, 2.0], [3.0, 4.0]])
        y = np.array([10.0, 20.0, 30.0])
        X2, y2 = drop_incomplete(X, y)
        self.assertEqual(len(y2), 2)
        self.assertTrue(np.array_equal(y2, [10.0, 30.0]))

    def test_bad_strategy(self):
        with self.assertRaisesRegex(ValueError, "mean or median"):
            SimpleImputer(strategy="mode")


class ReportTests(unittest.TestCase):
    def test_hours_are_the_blank_column(self):
        X, y, names = make_study_table(n=180, seed=2)
        report = {r["name"]: r for r in missing_report(X, names)}
        self.assertGreater(report["hours"]["n_missing"], 10)
        self.assertEqual(report["sleep"]["n_missing"], 0)
        rows = compare_impute(X, y, seed=2)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r.mse > 0 for r in rows))


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import MissingPlots

        X, y, names = make_study_table(n=40, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            plots = MissingPlots(tmp)
            paths = [
                plots.missing_bars(missing_report(X, names)),
                plots.mse_bars(compare_impute(X, y, seed=3)),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())
                self.assertGreater(Path(p).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
