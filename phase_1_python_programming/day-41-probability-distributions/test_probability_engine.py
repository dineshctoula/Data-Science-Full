"""Regression tests for the Day 41 probability-distributions engine."""

import unittest

import numpy as np

from probability_engine import ProbabilityEngine


class ProbabilityEngineTests(unittest.TestCase):
    """Check textbook identities, CDF totals, and input safeguards."""

    def test_bernoulli_pmf_and_mean_variance(self) -> None:
        # A fair coin has equal probability on the two support points.
        self.assertEqual(ProbabilityEngine.bernoulli_pmf(1, 0.5), 0.5)
        self.assertEqual(ProbabilityEngine.bernoulli_pmf(0, 0.5), 0.5)
        self.assertEqual(ProbabilityEngine.bernoulli_pmf(2, 0.5), 0.0)
        summary = ProbabilityEngine.bernoulli_summary(0.3)
        self.assertAlmostEqual(summary.mean, 0.3)
        self.assertAlmostEqual(summary.variance, 0.3 * 0.7)

    def test_binomial_pmf_sums_to_one_and_matches_hand_values(self) -> None:
        # For Binomial(5, 0.5), P(X = 2) = C(5, 2) / 32 = 10 / 32 = 0.3125.
        self.assertAlmostEqual(ProbabilityEngine.binomial_pmf(2, 5, 0.5), 0.3125)
        total = sum(ProbabilityEngine.binomial_pmf(k, 5, 0.5) for k in range(6))
        self.assertAlmostEqual(total, 1.0, places=12)
        # The median of a symmetric Binomial(5, 0.5) sits at 2.5, so P(X ≤ 2) = 0.5.
        self.assertAlmostEqual(ProbabilityEngine.binomial_cdf(2, 5, 0.5), 0.5)
        summary = ProbabilityEngine.binomial_summary(10, 0.4)
        self.assertAlmostEqual(summary.mean, 4.0)
        self.assertAlmostEqual(summary.variance, 10 * 0.4 * 0.6)

    def test_poisson_mean_equals_variance_and_pmf_is_normalized(self) -> None:
        summary = ProbabilityEngine.poisson_summary(2.5)
        self.assertEqual(summary.mean, summary.variance)
        # Truncating the infinite support at a large k still recovers nearly 1.
        total = sum(ProbabilityEngine.poisson_pmf(k, 2.5) for k in range(40))
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_normal_pdf_peak_and_standard_cdf_values(self) -> None:
        # The standard Normal peaks at x = 0 with height 1 / sqrt(2π).
        self.assertAlmostEqual(ProbabilityEngine.normal_pdf(0.0), 1.0 / np.sqrt(2.0 * np.pi), places=12)
        self.assertAlmostEqual(ProbabilityEngine.normal_cdf(0.0), 0.5, places=12)
        # The classic 95% two-sided critical value is approximately ±1.96.
        self.assertAlmostEqual(ProbabilityEngine.normal_cdf(1.96), 0.975, places=3)
        summary = ProbabilityEngine.normal_summary(10.0, 2.0)
        self.assertEqual(summary.mean, 10.0)
        self.assertEqual(summary.variance, 4.0)

    def test_exponential_pdf_cdf_and_memoryless_mean(self) -> None:
        # At x = 0 the Exponential(λ) density equals the rate λ.
        self.assertAlmostEqual(ProbabilityEngine.exponential_pdf(0.0, 2.0), 2.0)
        self.assertEqual(ProbabilityEngine.exponential_pdf(-1.0, 2.0), 0.0)
        self.assertAlmostEqual(ProbabilityEngine.exponential_cdf(0.5, 2.0), 1.0 - np.exp(-1.0))
        summary = ProbabilityEngine.exponential_summary(2.0)
        self.assertAlmostEqual(summary.mean, 0.5)
        self.assertAlmostEqual(summary.variance, 0.25)

    def test_monte_carlo_binomial_probability_is_close_to_cdf(self) -> None:
        samples = ProbabilityEngine.sample_binomial(n=10, p=0.5, size=8_000, seed=41)
        empirical = ProbabilityEngine.empirical_probability(samples, "le", 5)
        exact = ProbabilityEngine.binomial_cdf(5, 10, 0.5)
        # A few thousand draws should land within a couple of percentage points.
        self.assertAlmostEqual(empirical, exact, delta=0.03)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            ProbabilityEngine.bernoulli_pmf(1, 1.5)
        with self.assertRaisesRegex(ValueError, "positive"):
            ProbabilityEngine.poisson_pmf(1, 0.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            ProbabilityEngine.normal_pdf(0.0, sigma=-1.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ProbabilityEngine.binomial_pmf(-1, 5, 0.5)
        with self.assertRaisesRegex(ValueError, "le, ge, eq"):
            ProbabilityEngine.empirical_probability(np.array([1.0, 2.0]), "lt", 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
