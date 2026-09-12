"""Reusable batch gradient-descent tools for the Day 39 lesson.

The module deliberately accepts an objective and its gradient separately.  That
keeps the update rule visible and lets one optimizer solve many small problems.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class OptimizationTrace:
    """The parameters and measurements produced by an optimization run."""

    parameters: np.ndarray
    loss_history: np.ndarray
    gradient_norm_history: np.ndarray

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


class BatchGradientDescent:
    """Minimize differentiable functions using a fixed-size gradient step."""

    @staticmethod
    def _validate_inputs(initial_parameters: np.ndarray, learning_rate: float, iterations: int) -> np.ndarray:
        """Validate user inputs before an iteration can mutate parameters."""
        parameters = np.asarray(initial_parameters, dtype=float)
        if parameters.ndim != 1 or parameters.size == 0:
            raise ValueError("Initial parameters must be a non-empty one-dimensional array.")
        if not np.isfinite(parameters).all():
            raise ValueError("Initial parameters must contain only finite values.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive finite number.")
        if not isinstance(iterations, (int, np.integer)) or iterations < 1:
            raise ValueError("Iterations must be a positive integer.")
        # Copy so callers retain ownership of the array passed into the method.
        return parameters.copy()

    @staticmethod
    def minimize(
        objective: Objective,
        gradient: Gradient,
        initial_parameters: np.ndarray,
        learning_rate: float = 0.1,
        iterations: int = 100,
    ) -> OptimizationTrace:
        """Minimize an objective with ``parameters -= learning_rate * gradient``.

        The loss is recorded before the first update and after every update;
        therefore a trace with ``n`` updates always has ``n + 1`` losses.
        """
        parameters = BatchGradientDescent._validate_inputs(initial_parameters, learning_rate, iterations)
        losses = [BatchGradientDescent._evaluate_loss(objective, parameters)]
        gradient_norms: list[float] = []

        for _ in range(iterations):
            direction = BatchGradientDescent._evaluate_gradient(gradient, parameters)
            gradient_norms.append(float(np.linalg.norm(direction)))
            # Subtracting the gradient moves opposite the direction of fastest
            # increase; the learning rate determines how far that move travels.
            parameters -= learning_rate * direction
            if not np.isfinite(parameters).all():
                raise FloatingPointError("Optimization produced non-finite parameters.")
            losses.append(BatchGradientDescent._evaluate_loss(objective, parameters))

        return OptimizationTrace(parameters, np.asarray(losses), np.asarray(gradient_norms))

    @staticmethod
    def _evaluate_loss(objective: Objective, parameters: np.ndarray) -> float:
        """Evaluate an objective and reject invalid values near their source."""
        loss = float(objective(parameters))
        if not np.isfinite(loss):
            raise FloatingPointError("Objective must return a finite loss.")
        return loss

    @staticmethod
    def _evaluate_gradient(gradient: Gradient, parameters: np.ndarray) -> np.ndarray:
        """Return a finite gradient with the same shape as the parameters."""
        values = np.asarray(gradient(parameters), dtype=float)
        if values.shape != parameters.shape:
            raise ValueError("Gradient shape must match the parameter shape.")
        if not np.isfinite(values).all():
            raise FloatingPointError("Gradient must contain only finite values.")
        return values
