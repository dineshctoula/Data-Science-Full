"""Tests for Day 66 — accuracy can look great while the rare class is missed."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from balance_engine import (
    WeightedLogit,
    holdout_compare,
    make_rare_events,
    oversample_minority,
    score_predictions,
    split_indices,
)


class ScoreTests(unittest.TestCase):
    def test_all_negative_has_zero_recall(self):
        y = np.array([0, 0, 0, 1, 0, 0, 1])
        pred = np.zeros(len(y), dtype=int)
        score = score_predictions(y, pred)
        self.assertEqual(score.recall, 0.0)
        self.assertEqual(score.tp, 0)
        self.assertAlmostEqual(score.accuracy, 5 / 7)

    def test_perfect_calls_score_one(self):
        y = np.array([0, 1, 0, 1, 1])
        score = score_predictions(y, y.copy())
        self.assertAlmostEqual(score.precision, 1.0)
        self.assertAlmostEqual(score.recall, 1.0)
        self.assertAlmostEqual(score.f1, 1.0)

    def test_split_does_not_overlap(self):
        train, test = split_indices(80, test_frac=0.25, seed=4)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(train) + len(test), 80)


class BalanceTests(unittest.TestCase):
    def test_oversample_matches_the_bigger_class(self):
        rng = np.random.default_rng(3)
        y = np.array([0] * 40 + [1] * 5)
        X = rng.normal(size=(len(y), 2))
        Xb, yb = oversample_minority(X, y, seed=3)
        self.assertEqual(int(np.sum(yb == 1)), int(np.sum(yb == 0)))
        self.assertEqual(int(np.sum(yb == 0)), 40)

    def test_plain_misses_what_oversample_catches(self):
        X, y = make_rare_events(seed=66)
        result = holdout_compare(X, y, seed=66, pos_weight=12.0)
        # on this seed the quiet model never flags anyone
        self.assertEqual(result["plain"].recall, 0.0)
        self.assertGreater(result["oversample"].recall, result["plain"].recall)
        self.assertGreater(result["majority"].accuracy, result["oversample"].accuracy)
        self.assertGreater(result["weighted"].recall, result["plain"].recall)

    def test_bad_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            WeightedLogit(pos_weight=0)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import BalancePlots

        names = ["majority", "plain", "weighted"]
        with tempfile.TemporaryDirectory() as tmp:
            plots = BalancePlots(tmp)
            paths = [
                plots.metric_bars(names, [0.9, 0.9, 0.7], [0.0, 0.1, 0.6]),
                plots.catch_bars(names, [0, 1, 8], [0, 0, 20]),
            ]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
