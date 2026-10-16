"""Day 76 — Adam on the same lopsided bowl.

Gradient descent takes a huge first step on the steep axis and a tiny one
on the flat axis. Adam keeps a running mean of the gradient and of its
square, then divides. The first step is almost the same size on both
axes. That is the point of the method here, not a claim that Adam always
wins the race.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bowl_loss(weights, curvature) -> float:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    curvature = np.asarray(curvature, dtype=float).reshape(-1)
    if len(weights) != len(curvature):
        raise ValueError("weights and curvature differ in length")
    return float(0.5 * np.sum(curvature * weights * weights))


@dataclass
class OptRun:
    name: str
    losses: np.ndarray
    weights: np.ndarray
    first_step: np.ndarray

    @property
    def steps(self) -> int:
        return int(len(self.losses))


def _run_loop(update, steps: int, curvature, start) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(start, dtype=float).reshape(-1).copy()
    curvature = np.asarray(curvature, dtype=float).reshape(-1)
    losses = np.zeros(steps)
    path = np.zeros((steps, len(weights)))
    first_step = None
    for i in range(steps):
        before = weights.copy()
        weights = update(weights, i + 1)
        if first_step is None:
            first_step = before - weights
        losses[i] = bowl_loss(weights, curvature)
        path[i] = weights
    return losses, path, first_step


def gradient_descent(curvature=(1.0, 18.0), start=(2.0, 1.0), lr: float = 0.05, steps: int = 80) -> OptRun:
    curvature = np.asarray(curvature, dtype=float).reshape(-1)

    def update(weights, _t):
        return weights - lr * (curvature * weights)

    losses, path, first = _run_loop(update, steps, curvature, start)
    return OptRun("gd", losses, path, first)


def momentum(curvature=(1.0, 18.0), start=(2.0, 1.0), lr: float = 0.05, beta: float = 0.85, steps: int = 80) -> OptRun:
    curvature = np.asarray(curvature, dtype=float).reshape(-1)
    velocity = np.zeros_like(curvature)

    def update(weights, _t):
        nonlocal velocity
        grad = curvature * weights
        velocity = beta * velocity + (1.0 - beta) * grad
        return weights - lr * velocity

    losses, path, first = _run_loop(update, steps, curvature, start)
    return OptRun("momentum", losses, path, first)


def adam(
    curvature=(1.0, 18.0),
    start=(2.0, 1.0),
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    steps: int = 80,
) -> OptRun:
    """Bias-corrected Adam. Default lr is higher than GD's — it is not the same knob."""
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("betas should be in [0, 1)")
    if lr <= 0:
        raise ValueError("learning rate should be positive")
    if steps < 1:
        raise ValueError("need at least one step")
    curvature = np.asarray(curvature, dtype=float).reshape(-1)
    if np.any(curvature <= 0):
        raise ValueError("curvature should be positive")
    m = np.zeros_like(curvature)
    v = np.zeros_like(curvature)

    def update(weights, t):
        nonlocal m, v
        grad = curvature * weights
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        return weights - lr * m_hat / (np.sqrt(v_hat) + eps)

    losses, path, first = _run_loop(update, steps, curvature, start)
    return OptRun("adam", losses, path, first)
