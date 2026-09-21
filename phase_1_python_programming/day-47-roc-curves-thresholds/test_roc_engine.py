"""Regression tests for the Day 47 ROC and threshold-tuning engine."""

import unittest

import numpy as np

from roc_engine import RocEngine


class RocEngineTests(unittest.TestCase):
    """Check metric identities, AUC extremes, threshold choice, and safeguards."""

    def test_perfect_ranking_has_auc_near_one(self) -> None:
        # Every positive score sits strictly above every negative score.
        y_true = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        curve = RocEngine.build_curves(y_true, y_score)
        self.assertGreater(curve.auc_roc, 0.99)

    def test_random_scores_have_auc_near_half(self) -> None:
        labels, scores = RocEngine.generate_poor_ranking(n=300, seed=47)
        curve = RocEngine.build_curves(labels, scores)
        self.assertAlmostEqual(curve.auc_roc, 0.5, delta=0.08)

    def test_metrics_at_extreme_thresholds(self) -> None:
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_score = np.array([0.2, 0.8, 0.4, 0.6])
        # Threshold above every score predicts all negatives → TPR = FPR = 0.
        high = RocEngine.metrics_at_threshold(y_true, y_score, threshold=1.1)
        self.assertEqual(high.true_positive_rate, 0.0)
        self.assertEqual(high.false_positive_rate, 0.0)
        # Threshold below every score predicts all positives → TPR = FPR = 1.
        low = RocEngine.metrics_at_threshold(y_true, y_score, threshold=-0.1)
        self.assertEqual(low.true_positive_rate, 1.0)
        self.assertEqual(low.false_positive_rate, 1.0)

    def test_youden_and_f1_selectors_return_evaluated_thresholds(self) -> None:
        labels, scores = RocEngine.generate_scored_labels(n=160, seed=47)
        curve = RocEngine.build_curves(labels, scores)
        youden = curve.best_youden_threshold()
        f1 = curve.best_f1_threshold()
        self.assertIn(youden, curve.metrics_by_threshold)
        self.assertIn(f1, curve.metrics_by_threshold)
        self.assertGreaterEqual(youden.youden_j, curve.at_threshold(0.5).youden_j - 1e-12)

    def test_f1_is_harmonic_mean_of_precision_and_recall(self) -> None:
        metrics = RocEngine.metrics_at_threshold(
            np.array([0.0, 1.0, 1.0, 0.0, 1.0]),
            np.array([0.1, 0.9, 0.8, 0.4, 0.7]),
            threshold=0.5,
        )
        expected = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
        self.assertAlmostEqual(metrics.f1, expected, places=12)

    def test_auc_trapezoid_matches_hand_calculation(self) -> None:
        # Triangle under (0,0)-(0,1)-(1,1) has area 1; under (0,0)-(1,0)-(1,1) has area 0.
        self.assertAlmostEqual(RocEngine.auc_trapezoid([0.0, 0.0, 1.0], [0.0, 1.0, 1.0]), 1.0, places=12)
        self.assertAlmostEqual(RocEngine.auc_trapezoid([0.0, 1.0, 1.0], [0.0, 0.0, 1.0]), 0.0, places=12)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary"):
            RocEngine.metrics_at_threshold(np.array([0.0, 2.0]), np.array([0.1, 0.2]), 0.5)
        with self.assertRaisesRegex(ValueError, "Both classes"):
            RocEngine.build_curves(np.array([1.0, 1.0, 1.0]), np.array([0.2, 0.3, 0.4]))
        with self.assertRaisesRegex(ValueError, "equal-length"):
            RocEngine.auc_trapezoid(np.array([0.0, 1.0]), np.array([0.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
