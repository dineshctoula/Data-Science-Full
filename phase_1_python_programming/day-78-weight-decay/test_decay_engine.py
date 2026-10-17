"""Tests for Day 78 — some decay helps a wide fit, too much flattens it."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from decay_engine import DecayTrial, fit_decay, make_wide, mse_of, pick_by_val, sweep, three_way


class DecayTests(unittest.TestCase):
    def test_split_does_not_overlap(self):
        train, val, test = three_way(80, seed=4)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(val) & set(test)), 0)
        self.assertEqual(len(train) + len(val) + len(test), 80)

    def test_zero_decay_finds_a_clean_line(self):
        x = np.linspace(-1, 1, 40).reshape(-1, 1)
        y = 0.5 + 3.0 * x[:, 0]
        coef = fit_decay(x, y, decay=0.0, lr=0.5, epochs=2000)
        self.assertAlmostEqual(coef[1], 3.0, places=3)
        self.assertLess(mse_of(x, y, coef), 1e-6)

    def test_intercept_is_not_pulled(self):
        # constant target, no signal: a huge decay should still keep the mean
        X = np.random.default_rng(1).normal(size=(50, 3))
        y = np.full(50, 7.0)
        coef = fit_decay(X, y, decay=50.0, lr=0.01, epochs=3000)
        self.assertAlmostEqual(coef[0], 7.0, places=2)

    def test_negative_decay_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_decay(np.zeros((5, 1)), np.zeros(5), decay=-0.1)

    def test_middle_decay_wins_the_holdout(self):
        X, y = make_wide(seed=78)
        trials, _ = sweep(X, y, [0.0, 0.03, 0.1, 0.3, 1.0, 3.0], seed=78)
        plain, heavy = trials[0], trials[-1]
        chosen = pick_by_val(trials)
        self.assertLess(plain.train_mse, 0.05)
        self.assertNotIn(chosen.decay, (0.0, 3.0))
        self.assertLess(chosen.test_mse, plain.test_mse)
        self.assertLess(chosen.test_mse, heavy.test_mse)
        # decay shrinks the junk weights as it goes up
        norms = [t.noise_norm for t in trials]
        self.assertEqual(norms, sorted(norms, reverse=True))

    def test_pick_ignores_the_test_column(self):
        trials = [
            DecayTrial(0.0, 0.0, 4.0, 0.1, 2.0, 1.0, -1.0),
            DecayTrial(0.1, 0.2, 3.0, 9.0, 1.0, 1.0, -1.0),
        ]
        self.assertEqual(pick_by_val(trials).decay, 0.1)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import DecayPlots

        trials = [
            DecayTrial(0.0, 0.01, 4.8, 4.7, 2.3, 1.3, -0.8),
            DecayTrial(0.1, 0.14, 2.9, 3.1, 1.4, 0.9, -0.7),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            plots = DecayPlots(tmp)
            for path in (plots.mse_lines(trials), plots.weight_shrink(trials)):
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
