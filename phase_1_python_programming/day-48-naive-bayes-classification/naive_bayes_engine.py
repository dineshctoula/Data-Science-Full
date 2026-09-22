"""Reusable Naive Bayes tools for the Day 48 lesson.

Bayes' rule combines class priors with feature likelihoods.  The naive
assumption factors the joint likelihood into a product over features so the
posterior stays easy to compute by hand and in code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianClassStats:
    """Per-class mean, variance, and prior used by Gaussian Naive Bayes."""

    label: float
    prior: float
    means: np.ndarray
    variances: np.ndarray

    def summary(self) -> str:
        """Return a short printable description of one class model."""
        return (
            f"class {self.label:g}: prior = {self.prior:.3f}, "
            f"means = {np.round(self.means, 3)}, vars = {np.round(self.variances, 3)}"
        )


@dataclass(frozen=True)
class NaiveBayesResult:
    """Fitted Gaussian Naive Bayes model with predictions and accuracy."""

    class_stats: tuple[GaussianClassStats, ...]
    feature_names: tuple[str, ...]
    predictions: np.ndarray
    probabilities: np.ndarray
    accuracy: float
    n_observations: int
    n_features: int

    @property
    def classes(self) -> np.ndarray:
        """Return the fitted class labels in ascending order."""
        return np.array([stats.label for stats in self.class_stats], dtype=float)

    def summary(self) -> str:
        """Return a compact printable summary of the fit."""
        labels = ", ".join(f"{stats.label:g}" for stats in self.class_stats)
        return (
            f"GaussianNB classes [{labels}] | "
            f"accuracy = {self.accuracy:.3f}, n = {self.n_observations}, p = {self.n_features}"
        )


class NaiveBayesEngine:
    """Fit and apply Gaussian Naive Bayes with Laplace-smoothed priors."""

    @staticmethod
    def _validate_training_data(
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Validate X/y and return copies plus feature labels."""
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
        if np.unique(y).size < 2:
            raise ValueError("At least two classes are required.")
        if feature_names is None:
            names = tuple(f"x{index + 1}" for index in range(x.shape[1]))
        else:
            names = tuple(feature_names)
            if len(names) != x.shape[1]:
                raise ValueError("feature_names length must match the number of columns.")
        return x.copy(), y.copy(), names

    @staticmethod
    def gaussian_log_pdf(values: np.ndarray, mean: float, variance: float) -> np.ndarray:
        """Return log N(x | μ, σ²) for each value.

        Working in log space avoids underflow when many feature likelihoods
        are multiplied together under the naive independence assumption.
        """
        data = np.asarray(values, dtype=float)
        if not np.isfinite(mean) or not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("mean must be finite and variance must be positive.")
        # log density = -½ log(2πσ²) - (x − μ)² / (2σ²)
        return -0.5 * np.log(2.0 * np.pi * variance) - (data - mean) ** 2 / (2.0 * variance)

    @staticmethod
    def fit(
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
        var_smoothing: float = 1e-9,
    ) -> NaiveBayesResult:
        """Estimate class priors and per-feature Gaussian likelihoods.

        Priors use Laplace smoothing ``(count + 1) / (n + k)`` so no class
        receives probability zero.  A tiny ``var_smoothing`` keeps zero-width
        features from breaking the Normal density.
        """
        if not np.isfinite(var_smoothing) or var_smoothing < 0.0:
            raise ValueError("var_smoothing must be a non-negative finite number.")

        x, y, names = NaiveBayesEngine._validate_training_data(features, labels, feature_names)
        classes = np.sort(np.unique(y))
        n = y.size
        k = classes.size
        stats_list: list[GaussianClassStats] = []

        for label in classes:
            mask = y == label
            subset = x[mask]
            count = int(mask.sum())
            # Laplace-smoothed prior keeps empty-looking classes from vanishing
            # if we later evaluate on slightly different class sets.
            prior = (count + 1.0) / (n + k)
            means = subset.mean(axis=0)
            # Population variance within the class, then floor with smoothing.
            variances = subset.var(axis=0) + var_smoothing
            if np.any(variances <= 0.0):
                raise ValueError("All class feature variances must stay positive after smoothing.")
            stats_list.append(
                GaussianClassStats(
                    label=float(label),
                    prior=float(prior),
                    means=means,
                    variances=variances,
                )
            )

        result_stats = tuple(stats_list)
        probabilities = NaiveBayesEngine.predict_proba(x, result_stats)
        predictions = NaiveBayesEngine.predict_from_proba(probabilities, result_stats)
        accuracy = float(np.mean(predictions == y))
        return NaiveBayesResult(
            class_stats=result_stats,
            feature_names=names,
            predictions=predictions,
            probabilities=probabilities,
            accuracy=accuracy,
            n_observations=n,
            n_features=x.shape[1],
        )

    @staticmethod
    def log_joint(features: np.ndarray, class_stats: tuple[GaussianClassStats, ...]) -> np.ndarray:
        """Return log P(x, y) = log P(y) + Σ log P(x_j | y) for each class column."""
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Features must be a non-empty array with shape (n,) or (n, p).")
        if not class_stats:
            raise ValueError("class_stats must contain at least one class.")
        n_features = class_stats[0].means.size
        if x.shape[1] != n_features:
            raise ValueError(f"Features must have {n_features} columns.")

        joints = np.zeros((x.shape[0], len(class_stats)), dtype=float)
        for column, stats in enumerate(class_stats):
            # Start from the class prior, then add each feature's log-likelihood.
            total = np.full(x.shape[0], np.log(stats.prior), dtype=float)
            for feature_index in range(n_features):
                total += NaiveBayesEngine.gaussian_log_pdf(
                    x[:, feature_index],
                    float(stats.means[feature_index]),
                    float(stats.variances[feature_index]),
                )
            joints[:, column] = total
        return joints

    @staticmethod
    def predict_proba(
        features: np.ndarray, class_stats: tuple[GaussianClassStats, ...]
    ) -> np.ndarray:
        """Return posterior class probabilities via normalized log-joints."""
        log_joints = NaiveBayesEngine.log_joint(features, class_stats)
        # Log-sum-exp keeps the normalization stable across large negative logs.
        max_log = log_joints.max(axis=1, keepdims=True)
        exp_shifted = np.exp(log_joints - max_log)
        return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

    @staticmethod
    def predict_from_proba(
        probabilities: np.ndarray, class_stats: tuple[GaussianClassStats, ...]
    ) -> np.ndarray:
        """Choose the MAP class for each row of posterior probabilities."""
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 2 or probs.shape[1] != len(class_stats):
            raise ValueError("probabilities must have one column per class.")
        labels = np.array([stats.label for stats in class_stats], dtype=float)
        return labels[np.argmax(probs, axis=1)]

    @staticmethod
    def predict(features: np.ndarray, class_stats: tuple[GaussianClassStats, ...]) -> np.ndarray:
        """Return MAP class labels for new feature rows."""
        return NaiveBayesEngine.predict_from_proba(
            NaiveBayesEngine.predict_proba(features, class_stats), class_stats
        )

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Return a square confusion matrix ordered by ``classes``."""
        labels = np.asarray(y_true, dtype=float).reshape(-1)
        preds = np.asarray(y_pred, dtype=float).reshape(-1)
        class_list = np.asarray(classes, dtype=float).reshape(-1)
        if labels.shape != preds.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        if class_list.size < 2:
            raise ValueError("classes must contain at least two labels.")
        matrix = np.zeros((class_list.size, class_list.size), dtype=float)
        for row_index, actual in enumerate(class_list):
            for column_index, predicted in enumerate(class_list):
                matrix[row_index, column_index] = np.sum((labels == actual) & (preds == predicted))
        return matrix

    @staticmethod
    def generate_iris_like_sample(
        n_per_class: int = 40, seed: int = 48
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Simulate two measurements for three flower-like Gaussian classes."""
        if not isinstance(n_per_class, (int, np.integer)) or n_per_class < 5:
            raise ValueError("n_per_class must be an integer of at least 5.")
        rng = np.random.default_rng(seed)
        # Class centers are separated enough for Naive Bayes to learn well.
        centers = np.array([[1.5, 1.0], [4.0, 3.5], [6.5, 2.0]])
        scales = np.array([[0.55, 0.45], [0.60, 0.50], [0.55, 0.55]])
        features = []
        labels = []
        for label, (center, scale) in enumerate(zip(centers, scales)):
            sample = rng.normal(center, scale, size=(int(n_per_class), 2))
            features.append(sample)
            labels.append(np.full(int(n_per_class), float(label)))
        x = np.vstack(features)
        y = np.concatenate(labels)
        order = rng.permutation(x.shape[0])
        return x[order], y[order], ("petal_length", "petal_width")

    @staticmethod
    def generate_binary_exam_sample(
        n: int = 120, seed: int = 48
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Simulate pass/fail labels from study hours and sleep hours."""
        if not isinstance(n, (int, np.integer)) or n < 20:
            raise ValueError("n must be an integer of at least 20.")
        rng = np.random.default_rng(seed)
        n_pass = n // 2
        n_fail = n - n_pass
        # Passing students tend to study more and sleep a bit more.
        pass_features = np.column_stack(
            (
                rng.normal(8.0, 1.5, size=n_pass),
                rng.normal(7.5, 0.8, size=n_pass),
            )
        )
        fail_features = np.column_stack(
            (
                rng.normal(3.5, 1.4, size=n_fail),
                rng.normal(5.5, 1.0, size=n_fail),
            )
        )
        features = np.vstack((pass_features, fail_features))
        labels = np.concatenate((np.ones(n_pass), np.zeros(n_fail)))
        order = rng.permutation(features.shape[0])
        return features[order], labels[order], ("study_hours", "sleep_hours")
