"""Reusable logistic-regression tools for the Day 46 lesson.

The sigmoid maps a linear score into a probability, and gradient descent
minimizes binary cross-entropy so the update rule stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogisticResult:
    """Fitted logistic model with probabilities and classification metrics."""

    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    probabilities: np.ndarray
    predictions: np.ndarray
    loss_history: np.ndarray
    accuracy: float
    precision: float
    recall: float
    n_observations: int
    n_features: int

    @property
    def intercept(self) -> float:
        """Return the leading intercept coefficient."""
        return float(self.coefficients[0])

    @property
    def slopes(self) -> np.ndarray:
        """Return the slope coefficients that multiply each feature."""
        return self.coefficients[1:].copy()

    @property
    def final_loss(self) -> float:
        """Return the binary cross-entropy after the last update."""
        return float(self.loss_history[-1])

    @property
    def equation(self) -> str:
        """Return the linear predictor inside the sigmoid."""
        parts = [f"{self.intercept:.4f}"]
        for name, slope in zip(self.feature_names, self.slopes):
            sign = "+" if slope >= 0 else "-"
            parts.append(f"{sign} {abs(slope):.4f}·{name}")
        return "logit = " + " ".join(parts)

    def summary(self) -> str:
        """Return a compact printable summary of the fit."""
        return (
            f"{self.equation} | accuracy = {self.accuracy:.3f}, "
            f"precision = {self.precision:.3f}, recall = {self.recall:.3f}, "
            f"loss = {self.final_loss:.4f}"
        )


class LogisticEngine:
    """Fit binary logistic regression with batch gradient descent."""

    @staticmethod
    def sigmoid(scores: np.ndarray) -> np.ndarray:
        """Map real-valued scores to probabilities in (0, 1).

        A large positive score approaches 1; a large negative score approaches 0.
        The exponential is clipped so overflow does not produce NaNs.
        """
        values = np.asarray(scores, dtype=float)
        clipped = np.clip(values, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def logit(probabilities: np.ndarray) -> np.ndarray:
        """Return log-odds log(p / (1 − p)) for probabilities in (0, 1)."""
        probs = np.asarray(probabilities, dtype=float)
        if np.any(probs <= 0.0) or np.any(probs >= 1.0):
            raise ValueError("Probabilities for logit must lie strictly between 0 and 1.")
        return np.log(probs / (1.0 - probs))

    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
        """Return the average −[y log p + (1 − y) log(1 − p)].

        Tiny epsilons keep log(0) from exploding when a probability hits a boundary.
        """
        labels = np.asarray(y_true, dtype=float).reshape(-1)
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if labels.shape != probs.shape:
            raise ValueError("Labels and probabilities must have the same shape.")
        if not np.isfinite(labels).all() or not np.isfinite(probs).all():
            raise ValueError("Labels and probabilities must contain only finite numbers.")
        if np.any((labels != 0.0) & (labels != 1.0)):
            raise ValueError("Labels must be binary 0/1 values.")
        clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
        return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))

    @staticmethod
    def _validate_binary_design(
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Validate X and binary y, then return copies plus feature labels."""
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float).reshape(-1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Features must be a non-empty array with shape (n,) or (n, p).")
        if y.shape[0] != x.shape[0]:
            raise ValueError("Label length must match the number of feature rows.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Features and labels must contain only finite numbers.")
        if np.any((y != 0.0) & (y != 1.0)):
            raise ValueError("Labels must be binary 0/1 values.")
        if np.unique(y).size < 2:
            raise ValueError("Both classes 0 and 1 must appear in the labels.")

        if feature_names is None:
            names = tuple(f"x{index + 1}" for index in range(x.shape[1]))
        else:
            names = tuple(feature_names)
            if len(names) != x.shape[1]:
                raise ValueError("feature_names length must match the number of columns.")
        return x.copy(), y.copy(), names

    @staticmethod
    def design_matrix(features: np.ndarray) -> np.ndarray:
        """Return [1 | X] so the first coefficient is an intercept."""
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Features must be a non-empty array with shape (n,) or (n, p).")
        return np.column_stack((np.ones(x.shape[0]), x))

    @staticmethod
    def fit(
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
        learning_rate: float = 0.1,
        iterations: int = 2_000,
    ) -> LogisticResult:
        """Fit logistic regression by minimizing binary cross-entropy.

        Each update is ``β ← β − η · Xᵀ (p − y) / n``.  That gradient is the
        average residual between predicted probabilities and true labels.
        """
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive finite number.")
        if not isinstance(iterations, (int, np.integer)) or iterations < 1:
            raise ValueError("Iterations must be a positive integer.")

        x, y, names = LogisticEngine._validate_binary_design(features, labels, feature_names)
        design = LogisticEngine.design_matrix(x)
        coefficients = np.zeros(design.shape[1], dtype=float)
        history: list[float] = []
        n = float(x.shape[0])

        for _ in range(int(iterations)):
            scores = design @ coefficients
            probabilities = LogisticEngine.sigmoid(scores)
            history.append(LogisticEngine.binary_cross_entropy(y, probabilities))
            # Gradient of mean BCE w.r.t. coefficients for the logistic model.
            gradient = design.T @ (probabilities - y) / n
            coefficients -= learning_rate * gradient

        probabilities = LogisticEngine.sigmoid(design @ coefficients)
        history.append(LogisticEngine.binary_cross_entropy(y, probabilities))
        # Default decision rule: predict class 1 when probability ≥ 0.5.
        predicted = (probabilities >= 0.5).astype(float)
        metrics = LogisticEngine.classification_metrics(y, predicted)

        return LogisticResult(
            coefficients=coefficients,
            feature_names=names,
            probabilities=probabilities,
            predictions=predicted,
            loss_history=np.asarray(history),
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            n_observations=x.shape[0],
            n_features=x.shape[1],
        )

    @staticmethod
    def predict_proba(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        """Return class-1 probabilities for new rows using fitted coefficients."""
        design = LogisticEngine.design_matrix(features)
        weights = np.asarray(coefficients, dtype=float).reshape(-1)
        if design.shape[1] != weights.size:
            raise ValueError("Coefficient length must equal features columns + 1.")
        if not np.isfinite(weights).all():
            raise ValueError("Coefficients must contain only finite numbers.")
        return LogisticEngine.sigmoid(design @ weights)

    @staticmethod
    def predict(features: np.ndarray, coefficients: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return hard 0/1 labels from probabilities and a decision threshold."""
        if not np.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("threshold must be a finite number strictly between 0 and 1.")
        probabilities = LogisticEngine.predict_proba(features, coefficients)
        return (probabilities >= threshold).astype(float)

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Return accuracy, precision, and recall for binary predictions."""
        labels = np.asarray(y_true, dtype=float).reshape(-1)
        preds = np.asarray(y_pred, dtype=float).reshape(-1)
        if labels.shape != preds.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        if np.any((labels != 0.0) & (labels != 1.0)) or np.any((preds != 0.0) & (preds != 1.0)):
            raise ValueError("Metrics require binary 0/1 arrays.")

        true_positive = float(np.sum((preds == 1.0) & (labels == 1.0)))
        true_negative = float(np.sum((preds == 0.0) & (labels == 0.0)))
        false_positive = float(np.sum((preds == 1.0) & (labels == 0.0)))
        false_negative = float(np.sum((preds == 0.0) & (labels == 1.0)))
        total = labels.size
        accuracy = (true_positive + true_negative) / total
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Return a 2×2 matrix [[TN, FP], [FN, TP]]."""
        metrics = LogisticEngine.classification_metrics(y_true, y_pred)
        return np.array(
            [
                [metrics["true_negative"], metrics["false_positive"]],
                [metrics["false_negative"], metrics["true_positive"]],
            ],
            dtype=float,
        )

    @staticmethod
    def generate_exam_pass_data(
        n: int = 120, seed: int = 46
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Simulate pass/fail labels from study hours and prior GPA."""
        if not isinstance(n, (int, np.integer)) or n < 20:
            raise ValueError("n must be an integer of at least 20.")
        rng = np.random.default_rng(seed)
        hours = rng.uniform(0.5, 12.0, size=int(n))
        gpa = rng.uniform(1.5, 4.0, size=int(n))
        # True logit: more hours and higher GPA raise the pass probability.
        logits = -4.0 + 0.55 * hours + 1.1 * gpa
        probabilities = LogisticEngine.sigmoid(logits)
        labels = rng.binomial(1, probabilities).astype(float)
        features = np.column_stack((hours, gpa))
        return features, labels, ("study_hours", "gpa")

    @staticmethod
    def generate_one_feature_pass_data(
        n: int = 100, seed: int = 46
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a single-feature pass/fail series for sigmoid plots."""
        if not isinstance(n, (int, np.integer)) or n < 20:
            raise ValueError("n must be an integer of at least 20.")
        rng = np.random.default_rng(seed)
        hours = np.sort(rng.uniform(0.0, 10.0, size=int(n)))
        probabilities = LogisticEngine.sigmoid(-3.0 + 0.7 * hours)
        labels = rng.binomial(1, probabilities).astype(float)
        return hours.reshape(-1, 1), labels
