"""Tests for Day 76 — Adam's first step should not copy the raw gradient scale."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from adam_engine import adam, bowl_loss, gradient_descent, momentum


class AdamTests(unittest.TestCase):
    def test_origin_has_zero_loss(self):
        self.assertEqual(bowl_loss([0.0, 0.0], [1.0, 18.0]), 0.0)

    def test_gd_first_step_follows_the_gradient(self):
        run = gradient_descent(steps=1, lr=0.05)
        # grad at (2,1) is (2, 18); step is 0.05 * that
        np.testing.assert_allclose(run.first_step, [0.1, 0.9])

    def test_adam_first_step_is_nearly_square(self):
        run = adam(steps=1, lr=0.1)
        # bias-corrected Adam with zero state takes lr on both axes
        np.testing.assert_allclose(run.first_step, [0.1, 0.1], atol=1e-8)
        ratio = abs(run.first_step[1] / run.first_step[0])
        self.assertLess(ratio, 1.05)

    def test_momentum_still_beats_adam_on_this_bowl(self):
        gd = gradient_descent(steps=80)
        mom = momentum(steps=80)
        ada = adam(steps=80, lr=0.1)
        self.assertLess(mom.losses[-1], gd.losses[-1])
        self.assertLess(mom.losses[-1], ada.losses[-1])
        self.assertLess(ada.losses[-1], gd.losses[-1] * 2)

    def test_bad_beta_is_rejected(self):
        with self.assertRaises(ValueError):
            adam(beta1=1.0, steps=2)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from adam_engine import OptRun
        from visualizer import AdamPlots

        run = OptRun("gd", np.array([1.0, 0.5]), np.zeros((2, 2)), np.array([0.1, 0.9]))
        with tempfile.TemporaryDirectory() as tmp:
            plots = AdamPlots(tmp)
            paths = [plots.loss_curves([run]), plots.first_step_bars([run])]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
