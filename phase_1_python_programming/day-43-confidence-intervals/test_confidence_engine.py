"""Regression tests for the Day 43 confidence-interval engine."""

import unittest

import numpy as np

from confidence_engine import ConfidenceEngine


class ConfidenceEngineTests(unittest.TestCase):
    """Check critical values, interval geometry, and input safeguards."""

    def test_z_critical_matches_classic_95_percent_value(self) -> None:
        # The familiar Normal critical value for 95% confidence is about 1.96.
        self.assertAlmostEqual(ConfidenceEngine.z_critical(0.95), 1.96, places=2)

    def test_t_critical_exceeds_z_for_small_samples(self) -> None:
        # With few degrees of freedom the t tails are heavier, so t* > z*.
        z_star = ConfidenceEngine.z_critical(0.95)
        t_star = ConfidenceEngine.t_critical(0.95, df=9)
        self.assertGreater(t_star, z_star)

    def test_mean_z_interval_is_symmetric_around_the_sample_mean(self) -> None:
        sample = np.array([71.0, 72.0, 68.0, 70.0, 69.0, 73.0, 71.0, 70.0, 72.0, 69.0])
        interval = ConfidenceEngine.mean_z_interval(sample, sigma=5.0, confidence=0.95)
        self.assertAlmostEqual(interval.estimate, float(sample.mean()), places=12)
        self.assertAlmostEqual(interval.estimate - interval.lower, interval.upper - interval.estimate, places=12)
        self.assertAlmostEqual(interval.margin_of_error, interval.critical_value * interval.standard_error, places=12)

    def test_mean_t_interval_contains_the_sample_mean(self) -> None:
        sample = ConfidenceEngine.generate_exam_sample(n=40, seed=43)
        interval = ConfidenceEngine.mean_t_interval(sample, confidence=0.95)
        self.assertTrue(interval.contains(interval.estimate))
        self.assertEqual(interval.degrees_of_freedom, 39.0)
        self.assertGreater(interval.width, 0.0)

    def test_proportion_interval_matches_hand_calculation(self) -> None:
        # For 96/800 = 0.12, SE = sqrt(0.12 * 0.88 / 800) and z* ≈ 1.96.
        interval = ConfidenceEngine.proportion_interval(96, 800, confidence=0.95)
        se = np.sqrt(0.12 * 0.88 / 800)
        z_star = ConfidenceEngine.z_critical(0.95)
        self.assertAlmostEqual(interval.estimate, 0.12, places=12)
        self.assertAlmostEqual(interval.standard_error, se, places=12)
        self.assertAlmostEqual(interval.lower, 0.12 - z_star * se, places=8)
        self.assertAlmostEqual(interval.upper, 0.12 + z_star * se, places=8)

    def test_mean_difference_interval_centers_on_mean_gap(self) -> None:
        a = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
        b = np.array([14.0, 15.0, 16.0, 14.0, 15.0])
        interval = ConfidenceEngine.mean_difference_interval(a, b, equal_var=False)
        self.assertAlmostEqual(interval.estimate, float(a.mean() - b.mean()), places=12)
        self.assertTrue(interval.lower < interval.estimate < interval.upper)

    def test_coverage_simulation_is_near_the_nominal_level(self) -> None:
        coverage = ConfidenceEngine.simulate_mean_coverage(
            true_mean=50.0, sigma=10.0, n=30, confidence=0.95, trials=250, seed=43
        )
        # Monte Carlo noise is expected; stay within a few percentage points of 95%.
        self.assertGreater(coverage, 0.90)
        self.assertLess(coverage, 0.99)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            ConfidenceEngine.z_critical(1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            ConfidenceEngine.mean_z_interval(np.array([1.0, 2.0]), sigma=0.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            ConfidenceEngine.mean_t_interval(np.array([3.0, 3.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "successes"):
            ConfidenceEngine.proportion_interval(5, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
