"""Tests for Day 69 — sharpening should not change the 0.5 calls."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cal_engine import (
    brier,
    expected_calibration_error,
    holdout_compare,
    make_labels,
    sharpen,
    split_indices,
)


class CalTests(unittest.TestCase):
    def test_sharpen_keeps_the_half_calls(self):
        prob = np.array([0.1, 0.4, 0.5, 0.62, 0.9])
        sharp = sharpen(prob, temperature=0.5)
        np.testing.assert_array_equal(prob >= 0.5, sharp >= 0.5)
        self.assertLess(sharp[0], prob[0])
        self.assertGreater(sharp[-1], prob[-1])

    def test_overconfident_bins_have_higher_ece(self):
        y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
        honest = np.full(len(y), 0.2)
        boast = np.full(len(y), 0.9)
        self.assertGreater(expected_calibration_error(y, boast, n_bins=5), expected_calibration_error(y, honest, n_bins=5))
        self.assertAlmostEqual(brier(y, y.astype(float)), 0.0)

    def test_split_does_not_overlap(self):
        train, val, test = split_indices(80, seed=3)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(val) & set(test)), 0)
        self.assertEqual(len(train) + len(val) + len(test), 80)

    def test_platt_repairs_the_sharpened_brier(self):
        X, y = make_labels(n=600, seed=69)
        report = holdout_compare(X, y, temperature=0.5, seed=69)
        self.assertAlmostEqual(report["raw"].accuracy, report["sharp"].accuracy, places=6)
        self.assertGreater(report["sharp"].brier, report["raw"].brier)
        self.assertGreater(report["sharp"].ece, report["platt"].ece)
        self.assertLess(report["platt"].brier, report["sharp"].brier)
        # slope should undo a temperature of 0.5, roughly
        slope = float(report["platt_coef"][1])
        self.assertGreater(slope, 0.35)
        self.assertLess(slope, 0.7)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from cal_engine import Bin, CalScore
        from visualizer import CalPlots

        score = CalScore("raw", 0.8, 0.16, 0.04, [Bin(0.2, 0.25, 10), Bin(0.8, 0.75, 12)])
        with tempfile.TemporaryDirectory() as tmp:
            plots = CalPlots(tmp)
            paths = [plots.reliability([score]), plots.brier_bars(["raw", "sharp"], [0.16, 0.19])]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
