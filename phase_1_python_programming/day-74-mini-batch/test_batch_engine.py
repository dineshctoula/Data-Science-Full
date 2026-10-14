"""Tests for Day 74 — smaller batches take more steps per epoch."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from batch_engine import fit_batches, make_line, split_indices


class BatchTests(unittest.TestCase):
    def test_split_does_not_overlap(self):
        train, test = split_indices(50, test_frac=0.3, seed=2)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(train) + len(test), 50)

    def test_empty_batch_is_rejected(self):
        X, y = make_line(n=40, seed=1)
        with self.assertRaises(ValueError):
            fit_batches(X, y, batch_size=0, epochs=2, seed=1)

    def test_smaller_batches_move_faster_then_bounce(self):
        X, y = make_line(seed=74)
        full = fit_batches(X, y, batch_size=10**6, epochs=20, seed=74)
        mid = fit_batches(X, y, batch_size=32, epochs=20, seed=74)
        one = fit_batches(X, y, batch_size=1, epochs=20, seed=74)
        self.assertEqual(full.updates_per_epoch, 1)
        self.assertGreater(mid.updates_per_epoch, 1)
        self.assertEqual(one.updates_per_epoch, 168)
        # one epoch of small batches has already stepped more than full batch
        self.assertLess(mid.test_mse[0], full.test_mse[0])
        self.assertLess(one.test_mse[0], full.test_mse[0])
        # full batch is still descending; size 1 gets worse after a good first pass
        self.assertLess(full.test_mse[-1], full.test_mse[0])
        self.assertGreater(one.test_mse[4], one.test_mse[0])


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from batch_engine import BatchRun
        from visualizer import BatchPlots

        run = BatchRun(32, 6, np.array([2.0, 1.0]), np.array([2.2, 1.1]))
        with tempfile.TemporaryDirectory() as tmp:
            plots = BatchPlots(tmp)
            paths = [plots.test_curves([run]), plots.end_bars([run])]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
