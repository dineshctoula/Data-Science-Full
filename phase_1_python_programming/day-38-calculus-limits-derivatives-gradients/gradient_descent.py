"""Gradient-based optimization examples for Day 38."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    """Parameters and loss history emitted by gradient descent."""

    parameters: np.ndarray
    loss_history: np.ndarray


class GradientDescent:
    """Use gradients to minimize simple data-science objective functions."""

    @staticmethod
    def _validate_hyperparameters(learning_rate: float, iterations: int) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive finite number.")
        if iterations < 1:
            raise ValueError("Iterations must be at least one.")

    @staticmethod
    def quadratic(
        initial_value: float, target: float, learning_rate: float = 0.1, iterations: int = 50
    ) -> OptimizationResult:
        """Minimize f(x) = (x - target)² using its gradient 2(x - target)."""
        GradientDescent._validate_hyperparameters(learning_rate, iterations)
        value = float(initial_value)
        history: list[float] = []
        for _ in range(iterations):
            history.append((value - target) ** 2)
            value -= learning_rate * 2 * (value - target)
        history.append((value - target) ** 2)
        return OptimizationResult(np.array([value]), np.asarray(history))

    @staticmethod
    def linear_regression(
        features: np.ndarray,
        target: np.ndarray,
        learning_rate: float = 0.05,
        iterations: int = 1_000,
    ) -> OptimizationResult:
        """Fit intercept and coefficients by batch gradient descent on MSE.

        Features are standardized internally, then coefficients are converted back
        to the original units so callers can predict with raw feature values.
        """
        GradientDescent._validate_hyperparameters(learning_rate, iterations)
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Features must be a non-empty two-dimensional array.")
        if y.shape[0] != x.shape[0]:
            raise ValueError("Target length must match the number of feature rows.")

        means = x.mean(axis=0)
        scales = x.std(axis=0)
        if np.any(np.isclose(scales, 0.0)):
            raise ValueError("Features cannot contain a constant column.")
        standardized = (x - means) / scales
        design = np.column_stack((np.ones(x.shape[0]), standardized))
        weights = np.zeros(design.shape[1])
        history: list[float] = []

        for _ in range(iterations):
            residuals = design @ weights - y
            history.append(float(np.mean(residuals**2)))
            gradient = 2 / x.shape[0] * design.T @ residuals
            weights -= learning_rate * gradient

        residuals = design @ weights - y
        history.append(float(np.mean(residuals**2)))
        raw_coefficients = weights[1:] / scales
        raw_intercept = weights[0] - means @ raw_coefficients
        return OptimizationResult(np.concatenate(([raw_intercept], raw_coefficients)), np.asarray(history))

    @staticmethod
    def predict(features: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        """Predict from raw features using [intercept, coefficient₁, ...]."""
        x = np.asarray(features, dtype=float)
        weights = np.asarray(parameters, dtype=float).reshape(-1)
        if x.ndim != 2 or weights.size != x.shape[1] + 1:
            raise ValueError("Parameters must include one intercept and one value per feature.")
        return weights[0] + x @ weights[1:]
