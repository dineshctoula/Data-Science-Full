"""Tests for Day 65 — the leak should make the test number look too good, on average."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from leak_engine import (
    abs_correlation,
    compare_selection,
    make_wide_table,
    repeat_gap,
    split_indices,
    top_features,
)


class SelectionTests(unittest.TestCase):
    def test_perfect_column_wins(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=40)
        X = np.column_stack([rng.normal(size=40), y.copy(), rng.normal(size=40)])
        scores = abs_correlation(X, y)
        self.assertAlmostEqual(scores[1], 1.0, places=6)
        self.assertEqual(int(top_features(X, y, k=1)[0]), 1)

    def test_split_does_not_overlap(self):
        train, test = split_indices(50, test_frac=0.3, seed=2)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(train) + len(test), 50)

    def test_constant_column_scores_zero(self):
        X = np.column_stack([np.ones(20), np.arange(20, dtype=float)])
        y = np.arange(20, dtype=float)
        scores = abs_correlation(X, y)
        self.assertEqual(scores[0], 0.0)
        self.assertGreater(scores[1], 0.9)


class LeakTests(unittest.TestCase):
    def test_average_gap_is_positive(self):
        gaps = repeat_gap(lambda s: make_wide_table(n=70, n_noise=35, seed=s), n_seeds=16, k=3, base_seed=10)
        # not every seed leaks the same way, but the mean should favor the cheat
        self.assertGreater(float(gaps.mean()), 0.0)

    def test_compare_returns_k_columns(self):
        X, y = make_wide_table(n=60, n_noise=15, seed=3)
        result = compare_selection(X, y, k=3, seed=3)
        self.assertEqual(len(result.honest_cols), 3)
        self.assertEqual(len(result.leaky_cols), 3)
        self.assertGreater(result.honest_mse, 0)
        self.assertGreater(result.leaky_mse, 0)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import LeakPlots

        gaps = np.array([0.2, 0.5, -0.1, 0.4])
        with tempfile.TemporaryDirectory() as tmp:
            plots = LeakPlots(tmp)
            paths = [plots.gap_hist(gaps), plots.one_seed_bars(1.2, 0.8)]
            for p in paths:
                self.assertTrue(Path(p).exists())
                self.assertGreater(Path(p).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
