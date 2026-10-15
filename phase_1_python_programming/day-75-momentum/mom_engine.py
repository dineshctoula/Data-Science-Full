"""Day 75 — momentum on a lopsided bowl.

One direction is steep, the other is flat. Plain gradient descent kills
the steep coordinate in a few steps and then creeps along the flat one.
An average of recent gradients keeps a little speed on that flat axis.
Too much averaging and the path rings instead of settling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bowl_loss(weights, curvature) -> float:
    """0.5 * sum(curvature * weight^2). Minimum is the origin."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    curvature = np.asarray(curvature, dtype=float).reshape(-1)
    if len(weights) != len(curvature):
        raise ValueError("weights and curvature differ in length")
    return float(0.5 * np.sum(curvature * weights * weights))


@dataclass
class MomRun:
    beta: float
    losses: np.ndarray
    weights: np.ndarray  # one row per step, columns are the two axes

    @property
    def steps(self) -> int:
        return int(len(self.losses))


def descend(curvature=(1.0, 18.0), start=(2.0, 1.0), lr: float = 0.05, beta: float = 0.0, steps: int = 80) -> MomRun:
    """Heavy-ball style update. beta 0 is ordinary gradient descent.

    v = beta * v + (1 - beta) * grad, then step lr * v.
    The (1 - beta) keeps the first step on the same scale as a plain step
    when beta is 0, and shrinks it when beta is near 1.
    """
    if not 0 <= beta < 1:
        raise ValueError("beta should be in [0, 1)")
    if lr <= 0:
        raise ValueError("learning rate should be positive")
    if steps < 1:
        raise ValueError("need at least one step")
    curvature = np.asarray(curvature, dtype=float).reshape(-1)
    if np.any(curvature <= 0):
        raise ValueError("curvature should be positive")
    weights = np.asarray(start, dtype=float).reshape(-1).copy()
    if len(weights) != len(curvature):
        raise ValueError("start and curvature differ in length")

    velocity = np.zeros_like(weights)
    losses = np.zeros(steps)
    path = np.zeros((steps, len(weights)))
    for i in range(steps):
        grad = curvature * weights
        velocity = beta * velocity + (1.0 - beta) * grad
        weights = weights - lr * velocity
        losses[i] = bowl_loss(weights, curvature)
        path[i] = weights
    return MomRun(float(beta), losses, path)
