"""Tests for Day 68 — the training column will vote for k = 1."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from search_engine import Trial, choose_by_val, knn_predict, make_curve, run_search, three_way


class SearchTests(unittest.TestCase):
    def test_split_does_not_overlap(self):
        train, val, test = three_way(80, seed=4)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(val) & set(test)), 0)
        self.assertEqual(len(train) + len(val) + len(test), 80)

    def test_k1_finds_the_point_itself(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([4.0, 5.0, 9.0])
        pred = knn_predict(x, y, np.array([1.0]), k=1)
        self.assertAlmostEqual(float(pred[0]), 5.0)

    def test_choice_ignores_the_test_column(self):
        trials = [
            Trial(k=1, train_mse=0.0, val_mse=0.5, test_mse=0.01),
            Trial(k=9, train_mse=0.2, val_mse=0.15, test_mse=0.40),
        ]
        chosen = choose_by_val(trials)
        self.assertEqual(chosen.k, 9)

    def test_train_pick_loses_on_the_holdout(self):
        x, y = make_curve(n=280, noise=0.35, seed=68)
        trials = run_search(x, y, ks=[1, 3, 5, 9, 15, 25, 41], seed=68)
        by_train = min(trials, key=lambda trial: (trial.train_mse, trial.k))
        by_val = choose_by_val(trials)
        self.assertEqual(by_train.k, 1)
        self.assertAlmostEqual(by_train.train_mse, 0.0, places=6)
        self.assertNotEqual(by_val.k, 1)
        self.assertLess(by_val.test_mse, by_train.test_mse)

    def test_k_too_large_is_rejected(self):
        with self.assertRaises(ValueError):
            knn_predict(np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.array([0.2]), k=5)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import SearchPlots

        trials = [
            Trial(1, 0.0, 0.4, 0.5),
            Trial(5, 0.1, 0.2, 0.25),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            plots = SearchPlots(tmp)
            paths = [
                plots.mse_lines(trials),
                plots.pick_bars(["train pick", "val pick"], [0.5, 0.25]),
            ]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
