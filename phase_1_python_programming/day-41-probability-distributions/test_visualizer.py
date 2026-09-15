"""Tests for Day 41 distribution visualization output and validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from probability_engine import ProbabilityEngine
from visualizer import DistributionVisualizer


class DistributionVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = DistributionVisualizer(temporary_directory)
            binomial_path = visualizer.plot_binomial_pmf(8, 0.35)
            poisson_path = visualizer.plot_poisson_pmf(2.5, k_max=12)
            normal_path = visualizer.plot_normal_pdf_with_sample(
                ProbabilityEngine.sample_normal(0.0, 1.0, size=400, seed=7),
                mu=0.0,
                sigma=1.0,
            )
            exponential_path = visualizer.plot_exponential_pdf_cdf(1.2)
            # Each helper must produce a real PNG the pipeline can open.
            self.assertTrue(Path(binomial_path).is_file())
            self.assertTrue(Path(poisson_path).is_file())
            self.assertTrue(Path(normal_path).is_file())
            self.assertTrue(Path(exponential_path).is_file())

    def test_invalid_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = DistributionVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                visualizer.plot_binomial_pmf(-1, 0.5)
            with self.assertRaisesRegex(ValueError, "at least two"):
                visualizer.plot_normal_pdf_with_sample(np.array([1.0]))
            with self.assertRaisesRegex(ValueError, "positive"):
                visualizer.plot_exponential_pdf_cdf(1.0, x_max=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
