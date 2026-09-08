"""Numerical and analytical calculus tools for Day 38.

The small functions in this module keep the mathematical ideas visible while
still applying the validation needed for reliable experiments.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


ScalarFunction = Callable[[float], float]


class CalculusEngine:
    """Estimate limits and derivatives with finite-difference methods."""

    @staticmethod
    def _positive_step(step: float) -> float:
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Step size must be a positive finite number.")
        return float(step)

    @staticmethod
    def one_sided_limit(function: ScalarFunction, point: float, step: float = 1e-5) -> tuple[float, float]:
        """Estimate the left and right limits of ``function`` at ``point``."""
        h = CalculusEngine._positive_step(step)
        return float(function(point - h)), float(function(point + h))

    @staticmethod
    def symmetric_derivative(function: ScalarFunction, point: float, step: float = 1e-5) -> float:
        """Estimate f'(x) with the accurate central-difference formula."""
        h = CalculusEngine._positive_step(step)
        return float((function(point + h) - function(point - h)) / (2 * h))

    @staticmethod
    def derivative_curve(
        function: ScalarFunction, points: np.ndarray, step: float = 1e-5
    ) -> np.ndarray:
        """Evaluate central-difference derivatives at an array of points."""
        values = np.asarray(points, dtype=float)
        if values.ndim != 1:
            raise ValueError("Derivative points must be a one-dimensional array.")
        h = CalculusEngine._positive_step(step)
        return (np.asarray(function(values + h)) - np.asarray(function(values - h))) / (2 * h)

    @staticmethod
    def numerical_gradient(
        function: Callable[[np.ndarray], float], point: np.ndarray, step: float = 1e-5
    ) -> np.ndarray:
        """Estimate a multivariable gradient with one central difference per axis."""
        location = np.asarray(point, dtype=float)
        if location.ndim != 1 or location.size == 0:
            raise ValueError("Gradient point must be a non-empty one-dimensional array.")
        h = CalculusEngine._positive_step(step)
        gradient = np.empty_like(location)
        for index in range(location.size):
            offset = np.zeros_like(location)
            offset[index] = h
            gradient[index] = (function(location + offset) - function(location - offset)) / (2 * h)
        return gradient

    @staticmethod
    def polynomial_value(coefficients: np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate coefficients ordered from highest to lowest degree."""
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("Polynomial coefficients must be a non-empty one-dimensional array.")
        return np.polyval(coefficients, x)

    @staticmethod
    def polynomial_derivative(coefficients: np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the exact derivative of a polynomial at ``x``."""
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("Polynomial coefficients must be a non-empty one-dimensional array.")
        if coefficients.size == 1:
            return np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0
        powers = np.arange(coefficients.size - 1, 0, -1)
        return np.polyval(coefficients[:-1] * powers, x)
