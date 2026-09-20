"""Regression tests for the Day 46 logistic-regression engine."""

import unittest

import numpy as np

from logistic_engine import LogisticEngine


class LogisticEngineTests(unittest.TestCase):
    """Check sigmoid identities, training progress, metrics, and safeguards."""

    def test_sigmoid_maps_zero_to_one_half_and_stays_in_unit_interval(self) -> None:
        values = LogisticEngine.sigmoid(np.array([-100.0, 0.0, 100.0]))
        self.assertAlmostEqual(values[1], 0.5, places=12)
        # Extreme scores saturate at the boundaries under float64.
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))
        self.assertLess(values[0], 1e-10)
        self.assertGreater(values[2], 1.0 - 1e-10)

    def test_logit_is_inverse_of_sigmoid_on_open_unit_interval(self) -> None:
        scores = np.array([-2.0, 0.0, 1.5])
        probs = LogisticEngine.sigmoid(scores)
        recovered = LogisticEngine.logit(probs)
        np.testing.assert_allclose(recovered, scores, atol=1e-10)

    def test_binary_cross_entropy_is_zero_for_perfect_probabilities(self) -> None:
        labels = np.array([0.0, 1.0, 1.0, 0.0])
        probs = np.array([0.0, 1.0, 1.0, 0.0])
        # Clip inside the loss keeps this finite but extremely small.
        loss = LogisticEngine.binary_cross_entropy(labels, np.clip(probs, 1e-12, 1 - 1e-12))
        self.assertLess(loss, 1e-10)

    def test_fit_learns_positive_hours_coefficient_and_reduces_loss(self) -> None:
        features, labels = LogisticEngine.generate_one_feature_pass_data(n=100, seed=46)
        result = LogisticEngine.fit(features, labels, ("hours",), learning_rate=0.3, iterations=2_500)
        self.assertGreater(result.slopes[0], 0.0)
        self.assertLess(result.final_loss, result.loss_history[0])
        self.assertGreater(result.accuracy, 0.7)

    def test_two_feature_model_predicts_with_matching_shapes(self) -> None:
        features, labels, names = LogisticEngine.generate_exam_pass_data(n=80, seed=7)
        result = LogisticEngine.fit(features, labels, names, learning_rate=0.2, iterations=2_000)
        probs = LogisticEngine.predict_proba(features, result.coefficients)
        preds = LogisticEngine.predict(features, result.coefficients)
        self.assertEqual(probs.shape, labels.shape)
        self.assertEqual(preds.shape, labels.shape)
        self.assertTrue(np.all((preds == 0.0) | (preds == 1.0)))

    def test_confusion_matrix_matches_metric_counts(self) -> None:
        y_true = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        y_pred = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        matrix = LogisticEngine.confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(matrix, np.array([[1.0, 1.0], [1.0, 2.0]]))
        metrics = LogisticEngine.classification_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly between 0 and 1"):
            LogisticEngine.logit(np.array([0.0, 0.5]))
        with self.assertRaisesRegex(ValueError, "binary"):
            LogisticEngine.fit(np.array([[1.0], [2.0], [3.0]]), np.array([0.0, 1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "Both classes"):
            LogisticEngine.fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 1.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "positive"):
            LogisticEngine.fit(np.array([[1.0], [2.0], [3.0]]), np.array([0.0, 1.0, 0.0]), learning_rate=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
