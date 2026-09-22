"""Regression tests for the Day 48 Gaussian Naive Bayes engine."""

import unittest

import numpy as np

from naive_bayes_engine import NaiveBayesEngine


class NaiveBayesEngineTests(unittest.TestCase):
    """Check Bayes identities, classification quality, and input safeguards."""

    def test_gaussian_log_pdf_peaks_at_the_mean(self) -> None:
        values = np.array([0.0, 1.0, 2.0])
        dens = np.exp(NaiveBayesEngine.gaussian_log_pdf(values, mean=1.0, variance=1.0))
        self.assertGreater(dens[1], dens[0])
        self.assertGreater(dens[1], dens[2])

    def test_posteriors_sum_to_one_for_each_row(self) -> None:
        features, labels, names = NaiveBayesEngine.generate_binary_exam_sample(n=60, seed=3)
        result = NaiveBayesEngine.fit(features, labels, names)
        row_sums = result.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(row_sums.size), atol=1e-10)

    def test_iris_like_sample_is_nearly_linearly_separable(self) -> None:
        features, labels, names = NaiveBayesEngine.generate_iris_like_sample(n_per_class=40, seed=48)
        result = NaiveBayesEngine.fit(features, labels, names)
        self.assertGreaterEqual(result.accuracy, 0.95)
        self.assertEqual(len(result.class_stats), 3)
        # Balanced classes with Laplace smoothing stay near 1/3.
        for stats in result.class_stats:
            self.assertAlmostEqual(stats.prior, 1.0 / 3.0, places=2)

    def test_binary_exam_model_learns_higher_study_mean_for_passers(self) -> None:
        features, labels, names = NaiveBayesEngine.generate_binary_exam_sample(n=120, seed=48)
        result = NaiveBayesEngine.fit(features, labels, names)
        fail_stats, pass_stats = result.class_stats
        # Class 1 (pass) should have a larger mean study-hours feature.
        self.assertGreater(pass_stats.means[0], fail_stats.means[0])
        self.assertGreater(result.accuracy, 0.9)

    def test_predict_matches_training_argmax_labels(self) -> None:
        features, labels, names = NaiveBayesEngine.generate_iris_like_sample(n_per_class=20, seed=7)
        result = NaiveBayesEngine.fit(features, labels, names)
        predicted = NaiveBayesEngine.predict(features, result.class_stats)
        np.testing.assert_array_equal(predicted, result.predictions)

    def test_confusion_matrix_is_diagonal_for_perfect_fit(self) -> None:
        features, labels, names = NaiveBayesEngine.generate_iris_like_sample(n_per_class=30, seed=48)
        result = NaiveBayesEngine.fit(features, labels, names)
        matrix = NaiveBayesEngine.confusion_matrix(labels, result.predictions, result.classes)
        self.assertEqual(matrix.shape, (3, 3))
        # Off-diagonal mass should stay tiny when classes are well separated.
        off_diagonal = matrix.sum() - np.trace(matrix)
        self.assertLessEqual(off_diagonal, 5.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            NaiveBayesEngine.gaussian_log_pdf(np.array([0.0]), mean=0.0, variance=0.0)
        with self.assertRaisesRegex(ValueError, "two classes"):
            NaiveBayesEngine.fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 1.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            NaiveBayesEngine.fit(
                np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 2.0], [4.0, 1.0]]),
                np.array([0.0, 1.0, 0.0, 1.0]),
                var_smoothing=-1.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
