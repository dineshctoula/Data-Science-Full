"""Canonical problems solved with the Day 39 batch gradient-descent engine.

Each helper exposes a known minimum so one update rule can be checked on a
1-D bowl, an elongated 2-D bowl, and mean-squared-error linear regression.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optimization_engine import BatchGradientDescent, OptimizationTrace


Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ProblemResult:
    """An optimization run plus the parameter path used by contour plots."""

    parameters: np.ndarray
    loss_history: np.ndarray
    gradient_norm_history: np.ndarray
    parameter_history: np.ndarray
    target: np.ndarray | None = None

    @property
    def steps(self) -> int:
        """Return the number of parameter updates that were applied."""
        return self.loss_history.size - 1

    @property
    def initial_loss(self) -> float:
        """Return the loss before the first update."""
        return float(self.loss_history[0])

    @property
    def final_loss(self) -> float:
        """Return the loss after the final update."""
        return float(self.loss_history[-1])

    @property
    def loss_reduction(self) -> float:
        """Return the absolute improvement between the first and last losses."""
        return self.initial_loss - self.final_loss

    @property
    def loss_reduction_ratio(self) -> float:
        """Return the fraction of the initial loss removed by optimization.

        A zero initial loss is already optimal, so the ratio is defined as
        zero instead of dividing by zero.
        """
        if np.isclose(self.initial_loss, 0.0):
            return 0.0
        return self.loss_reduction / self.initial_loss

    @classmethod
    def from_trace(
        cls,
        trace: OptimizationTrace,
        parameter_history: np.ndarray,
        target: np.ndarray | None = None,
    ) -> ProblemResult:
        """Build a problem result from an engine trace and a recorded path."""
        copied_target = None if target is None else np.asarray(target, dtype=float).copy()
        return cls(
            parameters=trace.parameters,
            loss_history=trace.loss_history,
            gradient_norm_history=trace.gradient_norm_history,
            parameter_history=parameter_history,
            target=copied_target,
        )


class OptimizationProblems:
    """Reusable problem setups that all call ``BatchGradientDescent.minimize``."""

    @staticmethod
    def _minimize_with_history(
        objective: Objective,
        gradient: Gradient,
        initial_parameters: np.ndarray,
        learning_rate: float,
        iterations: int,
        target: np.ndarray | None = None,
    ) -> ProblemResult:
        """Run the shared engine while recording every evaluated parameter vector.

        The engine evaluates the objective before the first update and after
        every update, so the recorded list is the full descent path.
        """
        path: list[np.ndarray] = []

        def recording_objective(parameters: np.ndarray) -> float:
            path.append(np.asarray(parameters, dtype=float).copy())
            return objective(parameters)

        trace = BatchGradientDescent.minimize(
            recording_objective,
            gradient,
            initial_parameters,
            learning_rate=learning_rate,
            iterations=iterations,
        )
        return ProblemResult.from_trace(trace, np.asarray(path), target)

    @staticmethod
    def quadratic_bowl(
        initial_value: float = -5.0,
        target: float = 3.0,
        learning_rate: float = 0.1,
        iterations: int = 80,
    ) -> ProblemResult:
        """Minimize the one-dimensional bowl ``f(w) = (w - target)²``."""
        if not np.isfinite(initial_value) or not np.isfinite(target):
            raise ValueError("Initial value and target must be finite numbers.")

        return OptimizationProblems._minimize_with_history(
            objective=lambda values: float((values[0] - target) ** 2),
            gradient=lambda values: np.array([2.0 * (values[0] - target)]),
            initial_parameters=np.array([initial_value], dtype=float),
            learning_rate=learning_rate,
            iterations=iterations,
            target=np.array([target], dtype=float),
        )

    @staticmethod
    def elongated_bowl(
        initial_parameters: np.ndarray | None = None,
        target: np.ndarray | None = None,
        learning_rate: float = 0.08,
        iterations: int = 80,
    ) -> ProblemResult:
        """Minimize ``f(x, y) = (x - a)² + 4(y - b)²``.

        The y-direction is four times steeper, so the path approaches the
        minimum along a visible curved trajectory instead of a straight line.
        """
        start = (
            np.array([-3.0, 3.0], dtype=float)
            if initial_parameters is None
            else np.asarray(initial_parameters, dtype=float)
        )
        minimum = (
            np.array([2.0, -1.0], dtype=float)
            if target is None
            else np.asarray(target, dtype=float)
        )
        if start.shape != (2,) or minimum.shape != (2,):
            raise ValueError("The elongated bowl is defined on two-dimensional parameters.")
        if not np.isfinite(start).all() or not np.isfinite(minimum).all():
            raise ValueError("Initial parameters and target must contain only finite values.")

        def objective(values: np.ndarray) -> float:
            delta = values - minimum
            return float(delta[0] ** 2 + 4.0 * delta[1] ** 2)

        def gradient(values: np.ndarray) -> np.ndarray:
            delta = values - minimum
            return np.array([2.0 * delta[0], 8.0 * delta[1]])

        return OptimizationProblems._minimize_with_history(
            objective,
            gradient,
            start,
            learning_rate=learning_rate,
            iterations=iterations,
            target=minimum,
        )

    @staticmethod
    def compare_quadratic_learning_rates(
        initial_value: float = -5.0,
        target: float = 3.0,
        learning_rates: np.ndarray | None = None,
        iterations: int = 20,
    ) -> dict[float, ProblemResult]:
        """Run the same 1-D bowl at each candidate learning rate.

        Holding the start point and update count constant makes the final-loss
        differences attributable to the learning rate, not a changed problem.
        """
        rates = (
            np.array([0.05, 0.1, 0.2], dtype=float)
            if learning_rates is None
            else np.asarray(learning_rates, dtype=float)
        )
        if rates.ndim != 1 or rates.size == 0:
            raise ValueError("Learning rates must be a non-empty one-dimensional array.")

        comparisons: dict[float, ProblemResult] = {}
        for rate in rates:
            comparisons[float(rate)] = OptimizationProblems.quadratic_bowl(
                initial_value=initial_value,
                target=target,
                learning_rate=float(rate),
                iterations=iterations,
            )
        return comparisons

    @staticmethod
    def linear_regression(
        features: np.ndarray,
        target: np.ndarray,
        learning_rate: float = 0.1,
        iterations: int = 500,
    ) -> ProblemResult:
        """Fit intercept and coefficients by batch gradient descent on MSE.

        Features are standardized internally, then coefficients are converted
        back to the original units so callers can predict with raw values.
        """
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Features must be a non-empty two-dimensional array.")
        if y.shape[0] != x.shape[0]:
            raise ValueError("Target length must match the number of feature rows.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Features and target must contain only finite values.")

        # Scaling gives each feature roughly equal influence on the update size.
        means = x.mean(axis=0)
        scales = x.std(axis=0)
        if np.any(np.isclose(scales, 0.0)):
            raise ValueError("Features cannot contain a constant column.")
        standardized = (x - means) / scales
        # The leading column of ones lets the first weight represent intercept.
        design = np.column_stack((np.ones(x.shape[0]), standardized))
        n_rows = float(x.shape[0])

        def objective(weights: np.ndarray) -> float:
            residuals = design @ weights - y
            return float(np.mean(residuals**2))

        def gradient(weights: np.ndarray) -> np.ndarray:
            residuals = design @ weights - y
            return (2.0 / n_rows) * design.T @ residuals

        result = OptimizationProblems._minimize_with_history(
            objective,
            gradient,
            np.zeros(design.shape[1], dtype=float),
            learning_rate=learning_rate,
            iterations=iterations,
        )
        # Convert the standardized-space model back to raw feature units.
        raw_coefficient_path = result.parameter_history[:, 1:] / scales
        raw_intercept_path = result.parameter_history[:, 0] - raw_coefficient_path @ means
        raw_path = np.column_stack((raw_intercept_path, raw_coefficient_path))
        return ProblemResult(
            parameters=raw_path[-1],
            loss_history=result.loss_history,
            gradient_norm_history=result.gradient_norm_history,
            parameter_history=raw_path,
        )

    @staticmethod
    def predict(features: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        """Predict from raw features using ``[intercept, coefficient₁, ...]``."""
        x = np.asarray(features, dtype=float)
        weights = np.asarray(parameters, dtype=float).reshape(-1)
        if x.ndim != 2 or weights.size != x.shape[1] + 1:
            raise ValueError("Parameters must include one intercept and one value per feature.")
        if not np.isfinite(x).all() or not np.isfinite(weights).all():
            raise ValueError("Features and parameters must contain only finite values.")
        return weights[0] + x @ weights[1:]
