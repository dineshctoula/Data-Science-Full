"""Regression tests for the Day 40 descriptive-statistics engine."""

import unittest

import numpy as np

from stats_engine import DescriptiveStats


class DescriptiveStatsTests(unittest.TestCase):
    """Check textbook identities, outlier fences, and input safeguards."""

    def test_center_and_spread_match_hand_calculations(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(DescriptiveStats.mean(values), 3.0)
        self.assertEqual(DescriptiveStats.median(values), 3.0)
        self.assertEqual(DescriptiveStats.variance(values, sample=False), 2.0)
        self.assertEqual(DescriptiveStats.variance(values, sample=True), 2.5)
        self.assertEqual(DescriptiveStats.percentile(values, 25.0), 2.0)
        self.assertEqual(DescriptiveStats.percentile(values, 75.0), 4.0)
        self.assertAlmostEqual(DescriptiveStats.skewness(values), 0.0, places=12)
        np.testing.assert_array_equal(DescriptiveStats.modes(values), np.array([]))

    def test_even_count_median_averages_the_two_center_values(self) -> None:
        self.assertEqual(DescriptiveStats.median(np.array([1.0, 2.0, 3.0, 4.0])), 2.5)

    def test_modes_return_every_most_frequent_value(self) -> None:
        np.testing.assert_array_equal(DescriptiveStats.modes(np.array([1.0, 2.0, 2.0, 3.0])), np.array([2.0]))
        np.testing.assert_array_equal(
            DescriptiveStats.modes(np.array([1.0, 1.0, 2.0, 2.0, 3.0])),
            np.array([1.0, 2.0]),
        )

    def test_iqr_fences_flag_the_extreme_observation(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        mask = DescriptiveStats.outlier_mask(values)
        np.testing.assert_array_equal(values[mask], np.array([100.0]))
        summary = DescriptiveStats.five_number_summary(values)
        self.assertEqual(summary.iqr, 2.0)
        self.assertEqual(summary.lower_fence, -1.0)
        self.assertEqual(summary.upper_fence, 7.0)

    def test_z_scores_have_zero_mean_and_unit_sample_std(self) -> None:
        values = np.array([2.0, 4.0, 6.0, 8.0])
        z_scores = DescriptiveStats.z_scores(values)
        self.assertAlmostEqual(float(z_scores.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(z_scores.std(ddof=1)), 1.0, places=12)

    def test_summarize_keeps_sample_properties_together(self) -> None:
        scores = DescriptiveStats.generate_exam_scores()
        report = DescriptiveStats.summarize(scores)
        self.assertEqual(report.count, 80)
        self.assertGreater(report.mean, report.median)
        self.assertGreater(report.skewness, 0.0)
        self.assertGreater(report.outlier_count, 0)
        self.assertEqual(report.z_scores.size, report.count)
        np.testing.assert_array_equal(report.outlier_values, report.values[report.outlier_mask])

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            DescriptiveStats.mean(np.array([[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            DescriptiveStats.mean(np.array([1.0, np.nan]))
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            DescriptiveStats.percentile(np.array([1.0, 2.0]), 120.0)
        with self.assertRaisesRegex(ValueError, "zero"):
            DescriptiveStats.z_scores(np.array([3.0, 3.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "the same"):
            DescriptiveStats.skewness(np.array([2.0, 2.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "at least 3"):
            DescriptiveStats.generate_exam_scores(count=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
