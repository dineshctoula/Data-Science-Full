"""Tests for Day 77 — a stuck high rate should lose to a schedule that cools."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sched_engine import (
    constant_lr,
    cosine_lr,
    fit_schedule,
    make_line,
    split_indices,
    step_decay_lr,
)


class SchedTests(unittest.TestCase):
    def test_split_does_not_overlap(self):
        train, test = split_indices(50, test_frac=0.3, seed=2)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(train) + len(test), 50)

    def test_step_decay_halves_on_schedule(self):
        self.assertAlmostEqual(step_decay_lr(1, 1.8, 60, every=15), 1.8)
        self.assertAlmostEqual(step_decay_lr(15, 1.8, 60, every=15), 1.8)
        self.assertAlmostEqual(step_decay_lr(16, 1.8, 60, every=15), 0.9)
        self.assertAlmostEqual(step_decay_lr(31, 1.8, 60, every=15), 0.45)

    def test_cosine_ends_near_zero(self):
        self.assertAlmostEqual(cosine_lr(1, 1.8, 60), 1.8, places=6)
        self.assertLess(cosine_lr(60, 1.8, 60), 1e-9)

    def test_schedules_recover_when_constant_stalls(self):
        X, y = make_line(seed=77)
        stuck = fit_schedule(X, y, constant_lr, "constant", base_lr=1.8, epochs=60, seed=77)
        stepped = fit_schedule(X, y, step_decay_lr, "step", base_lr=1.8, epochs=60, seed=77)
        cooled = fit_schedule(X, y, cosine_lr, "cosine", base_lr=1.8, epochs=60, seed=77)
        self.assertGreater(stuck.test_mse[-1], 1.5)
        self.assertLess(stepped.test_mse[-1], 1.0)
        self.assertLess(cooled.test_mse[-1], 1.0)
        self.assertLess(stepped.test_mse[-1], stuck.test_mse[-1])


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from sched_engine import SchedRun
        from visualizer import SchedPlots

        run = SchedRun("constant", np.array([1.8, 1.8]), np.array([2.0, 1.0]), np.array([2.2, 1.1]))
        with tempfile.TemporaryDirectory() as tmp:
            plots = SchedPlots(tmp)
            paths = [plots.lr_curves([run]), plots.test_curves([run])]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
