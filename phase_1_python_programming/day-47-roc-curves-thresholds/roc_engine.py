"""Reusable ROC and threshold-tuning tools for the Day 47 lesson.

Varying the decision threshold trades false positives for false negatives.
ROC/PR curves make that tradeoff visible; AUC summarizes ranking quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdMetrics:
    """Classification metrics at one probability cutoff."""

    threshold: float
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positive: float
    false_positive: float
    true_negative: float
    false_negative: float

    @property
    def youden_j(self) -> float:
        """Return Youden's J = TPR − FPR (higher is better balanced separation)."""
        return self.true_positive_rate - self.false_positive_rate

    def summary(self) -> str:
        """Return a printable one-line description of this threshold."""
        return (
            f"t = {self.threshold:.3f} | TPR = {self.true_positive_rate:.3f}, "
            f"FPR = {self.false_positive_rate:.3f}, precision = {self.precision:.3f}, "
            f"F1 = {self.f1:.3f}, J = {self.youden_j:.3f}"
        )


@dataclass(frozen=True)
class CurveResult:
    """A full threshold sweep with ROC/PR coordinates and AUC."""

    thresholds: np.ndarray
    tpr: np.ndarray
    fpr: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    auc_roc: float
    metrics_by_threshold: tuple[ThresholdMetrics, ...]

    def best_youden_threshold(self) -> ThresholdMetrics:
        """Return the threshold that maximizes Youden's J statistic."""
        return max(self.metrics_by_threshold, key=lambda item: item.youden_j)

    def best_f1_threshold(self) -> ThresholdMetrics:
        """Return the threshold that maximizes the F1 score."""
        return max(self.metrics_by_threshold, key=lambda item: item.f1)

    def at_threshold(self, threshold: float) -> ThresholdMetrics:
        """Return metrics for the nearest evaluated threshold."""
        index = int(np.argmin(np.abs(self.thresholds - float(threshold))))
        return self.metrics_by_threshold[index]


class RocEngine:
    """Build ROC/PR curves and choose classification thresholds."""

    @staticmethod
    def _validate_scores(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate binary labels and finite probability/score arrays."""
        labels = np.asarray(y_true, dtype=float).reshape(-1)
        scores = np.asarray(y_score, dtype=float).reshape(-1)
        if labels.shape != scores.shape or labels.size == 0:
            raise ValueError("y_true and y_score must be equal-length non-empty arrays.")
        if not np.isfinite(labels).all() or not np.isfinite(scores).all():
            raise ValueError("y_true and y_score must contain only finite numbers.")
        if np.any((labels != 0.0) & (labels != 1.0)):
            raise ValueError("y_true must contain only binary 0/1 labels.")
        if np.unique(labels).size < 2:
            raise ValueError("Both classes 0 and 1 must appear in y_true.")
        return labels, scores

    @staticmethod
    def metrics_at_threshold(
        y_true: np.ndarray, y_score: np.ndarray, threshold: float
    ) -> ThresholdMetrics:
        """Compute TPR/FPR/precision/recall/F1 for one cutoff on the scores."""
        labels, scores = RocEngine._validate_scores(y_true, y_score)
        if not np.isfinite(threshold):
            raise ValueError("threshold must be a finite number.")

        # Predict class 1 when the score meets or exceeds the cutoff.
        predicted = (scores >= threshold).astype(float)
        true_positive = float(np.sum((predicted == 1.0) & (labels == 1.0)))
        false_positive = float(np.sum((predicted == 1.0) & (labels == 0.0)))
        true_negative = float(np.sum((predicted == 0.0) & (labels == 0.0)))
        false_negative = float(np.sum((predicted == 0.0) & (labels == 1.0)))

        positives = true_positive + false_negative
        negatives = true_negative + false_positive
        tpr = true_positive / positives if positives else 0.0
        fpr = false_positive / negatives if negatives else 0.0
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0
        )
        recall = tpr
        # F1 is the harmonic mean of precision and recall.
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        accuracy = (true_positive + true_negative) / labels.size
        return ThresholdMetrics(
            threshold=float(threshold),
            true_positive_rate=float(tpr),
            false_positive_rate=float(fpr),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
            true_positive=true_positive,
            false_positive=false_positive,
            true_negative=true_negative,
            false_negative=false_negative,
        )

    @staticmethod
    def build_curves(
        y_true: np.ndarray,
        y_score: np.ndarray,
        thresholds: np.ndarray | None = None,
    ) -> CurveResult:
        """Sweep thresholds to build ROC and precision-recall coordinates."""
        labels, scores = RocEngine._validate_scores(y_true, y_score)
        if thresholds is None:
            # Include scores themselves plus endpoints so the curve spans [0, 1].
            unique_scores = np.unique(scores)
            cutoffs = np.concatenate(([unique_scores.max() + 1e-6], unique_scores[::-1], [-1e-6]))
        else:
            cutoffs = np.asarray(thresholds, dtype=float)
            if cutoffs.ndim != 1 or cutoffs.size == 0:
                raise ValueError("thresholds must be a non-empty one-dimensional array.")
            if not np.isfinite(cutoffs).all():
                raise ValueError("thresholds must contain only finite numbers.")
            # Sort high → low so ROC moves from (0,0)-ish toward (1,1).
            cutoffs = np.sort(cutoffs)[::-1]

        metrics = tuple(
            RocEngine.metrics_at_threshold(labels, scores, float(cutoff)) for cutoff in cutoffs
        )
        tpr = np.array([item.true_positive_rate for item in metrics])
        fpr = np.array([item.false_positive_rate for item in metrics])
        precision = np.array([item.precision for item in metrics])
        recall = np.array([item.recall for item in metrics])
        f1 = np.array([item.f1 for item in metrics])
        return CurveResult(
            thresholds=cutoffs,
            tpr=tpr,
            fpr=fpr,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=RocEngine.auc_trapezoid(fpr, tpr),
            metrics_by_threshold=metrics,
        )

    @staticmethod
    def auc_trapezoid(x_values: np.ndarray, y_values: np.ndarray) -> float:
        """Integrate y against x with the trapezoidal rule after sorting by x."""
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
            raise ValueError("x_values and y_values must be equal-length 1-D arrays with ≥ 2 points.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("AUC inputs must contain only finite numbers.")
        order = np.argsort(x)
        return float(np.trapezoid(y[order], x[order]))

    @staticmethod
    def generate_scored_labels(
        n: int = 200,
        seed: int = 47,
        separation: float = 1.6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate binary labels with class-conditional score distributions.

        Positives are centered higher than negatives by ``separation``, so a
        useful ranking exists without needing a separate classifier module.
        """
        if not isinstance(n, (int, np.integer)) or n < 20:
            raise ValueError("n must be an integer of at least 20.")
        if not np.isfinite(separation) or separation < 0:
            raise ValueError("separation must be a non-negative finite number.")
        rng = np.random.default_rng(seed)
        n_pos = n // 2
        n_neg = n - n_pos
        # Scores play the role of predicted probabilities after a sigmoid squash.
        pos_logits = rng.normal(separation, 1.0, size=n_pos)
        neg_logits = rng.normal(-separation, 1.0, size=n_neg)
        logits = np.concatenate((pos_logits, neg_logits))
        labels = np.concatenate((np.ones(n_pos), np.zeros(n_neg)))
        scores = 1.0 / (1.0 + np.exp(-logits))
        # Shuffle so class order does not leak into later plots.
        order = rng.permutation(n)
        return labels[order], scores[order]

    @staticmethod
    def generate_poor_ranking(n: int = 200, seed: int = 47) -> tuple[np.ndarray, np.ndarray]:
        """Simulate nearly random scores so AUC sits close to 0.5."""
        if not isinstance(n, (int, np.integer)) or n < 20:
            raise ValueError("n must be an integer of at least 20.")
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, size=int(n)).astype(float)
        # Uninformative scores: independent of the true label.
        scores = rng.uniform(0.0, 1.0, size=int(n))
        if np.unique(labels).size < 2:
            labels[0], labels[1] = 0.0, 1.0
        return labels, scores
