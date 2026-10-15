"""Tests for Day 75 — a moderate beta should beat both no momentum and too much."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mom_engine import bowl_loss, descend


class MomTests(unittest.TestCase):
    def test_loss_of_the_origin_is_zero(self):
        self.assertEqual(bowl_loss([0.0, 0.0], [1.0, 18.0]), 0.0)

    def test_beta_zero_is_plain_gradient_descent(self):
        # one step from (2, 1): grad is (2, 18), step 0.05, so (1.9, 0.1)
        run = descend(beta=0.0, steps=1)
        np.testing.assert_allclose(run.weights[0], [1.9, 0.1])
        self.assertAlmostEqual(run.losses[0], bowl_loss([1.9, 0.1], [1.0, 18.0]))

    def test_beta_of_one_is_rejected(self):
        with self.assertRaises(ValueError):
            descend(beta=1.0, steps=2)

    def test_moderate_beta_wins_and_high_beta_does_not(self):
        plain = descend(beta=0.0, steps=80)
        mid = descend(beta=0.85, steps=80)
        high = descend(beta=0.9, steps=80)
        self.assertLess(mid.losses[-1], plain.losses[-1])
        # 0.9 is still ringing at step 80 and finishes above plain GD
        self.assertGreater(high.losses[-1], plain.losses[-1])
        # plain GD has already flattened the steep axis by step 10
        self.assertAlmostEqual(plain.weights[9, 1], 0.0, places=6)
        self.assertGreater(abs(plain.weights[9, 0]), 1.0)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from mom_engine import MomRun
        from visualizer import MomPlots

        run = MomRun(0.0, np.array([1.0, 0.5]), np.array([[1.5, 0.2], [1.2, 0.0]]))
        with tempfile.TemporaryDirectory() as tmp:
            plots = MomPlots(tmp)
            paths = [plots.loss_curves([run]), plots.flat_axis([run])]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
