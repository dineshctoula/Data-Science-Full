"""Regression tests for the Day 42 hypothesis-testing engine."""

import unittest

import numpy as np

from hypothesis_engine import HypothesisEngine


class HypothesisEngineTests(unittest.TestCase):
    """Check textbook decisions, tail directions, and input safeguards."""

    def test_one_sample_z_test_fails_to_reject_near_null(self) -> None:
        sample = np.array([71.0, 72.0, 68.0, 70.0, 69.0, 73.0, 71.0, 70.0, 72.0, 69.0])
        result = HypothesisEngine.one_sample_z_test(sample, mu0=70.0, sigma=5.0)
        # The sample mean is only 0.5 above 70 with a generous known σ, so H₀ should stand.
        self.assertGreater(result.p_value, 0.05)
        self.assertFalse(result.reject_null)

    def test_one_sample_t_test_rejects_when_mean_is_far_from_null(self) -> None:
        sample = np.array([12.0, 13.0, 14.0, 15.0, 16.0])
        result = HypothesisEngine.one_sample_t_test(sample, mu0=10.0, alternative="greater")
        self.assertGreater(result.statistic, 0.0)
        self.assertLess(result.p_value, 0.05)
        self.assertTrue(result.reject_null)
        self.assertEqual(result.degrees_of_freedom, 4.0)

    def test_welch_two_sample_test_detects_separated_groups(self) -> None:
        control = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
        variant = np.array([14.0, 15.0, 16.0, 14.0, 15.0])
        result = HypothesisEngine.two_sample_t_test(control, variant, equal_var=False)
        self.assertLess(result.p_value, 0.05)
        self.assertTrue(result.reject_null)

    def test_one_proportion_z_test_uses_null_standard_error(self) -> None:
        # 96/1200 = 8% exactly matches p0, so the z statistic should be ~0.
        result = HypothesisEngine.one_proportion_z_test(96, 1200, p0=0.08)
        self.assertAlmostEqual(result.statistic, 0.0, places=6)
        # At z = 0 the two-sided p-value is 1 because both tails are equally far.
        self.assertAlmostEqual(result.p_value, 1.0, places=6)

    def test_chi_square_goodness_of_fit_is_high_when_counts_match(self) -> None:
        result = HypothesisEngine.chi_square_goodness_of_fit(
            observed=np.array([50.0, 50.0, 50.0]),
            expected=np.array([50.0, 50.0, 50.0]),
        )
        self.assertAlmostEqual(result.statistic, 0.0, places=9)
        self.assertGreater(result.p_value, 0.05)
        self.assertFalse(result.reject_null)

    def test_paired_before_after_dataset_has_expected_size(self) -> None:
        before, after = HypothesisEngine.generate_before_after_scores(seed=42)
        self.assertEqual(before.size, after.size)
        self.assertGreater(after.mean(), before.mean())

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            HypothesisEngine.one_sample_z_test(np.array([1.0, 2.0]), 0.0, 1.0, alpha=1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            HypothesisEngine.one_sample_t_test(np.array([3.0, 3.0, 3.0]), 3.0)
        with self.assertRaisesRegex(ValueError, "alternative"):
            HypothesisEngine.one_sample_t_test(np.array([1.0, 2.0, 3.0]), 0.0, alternative="both")
        with self.assertRaisesRegex(ValueError, "successes"):
            HypothesisEngine.one_proportion_z_test(5, 4, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
