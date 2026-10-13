"""Day 73 — the step size is a real choice.

Same line, same 40 epochs, three learning rates. A small step is still
far from the fit when the budget runs out. A moderate step settles near
the noise. A step past the stable range walks away from the data, and
the loss gets worse every epoch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_line(n: int = 240, seed: int = 73):
    """y = 3 x0 - 1.5 x1 plus noise. Both columns are already standard scale."""
    if n < 20:
        raise ValueError("need at least 20 rows")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 1.0, size=n)
    return X, y


def split_indices(n: int, test_frac: float = 0.3, seed: int = 73):
    if not 0 < test_frac < 1:
        raise ValueError("test_frac should be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    if n_test >= n:
        raise ValueError("test slice ate the whole table")
    return idx[n_test:], idx[:n_test]


def mse(y, pred) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if len(y) != len(pred) or len(y) == 0:
        raise ValueError("y and pred should be the same non-empty length")
    return float(np.mean((y - pred) ** 2))


@dataclass
class RateRun:
    lr: float
    train_mse: np.ndarray
    test_mse: np.ndarray
    coef: np.ndarray

    @property
    def epochs(self) -> int:
        return int(len(self.train_mse))


def fit_rate(X, y, lr: float, epochs: int = 40, test_frac: float = 0.3, seed: int = 73) -> RateRun:
    """Full-batch least squares. One fixed step size for the whole run."""
    if lr <= 0:
        raise ValueError("learning rate should be positive")
    if epochs < 1:
        raise ValueError("need at least one epoch")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(X) != len(y):
        raise ValueError("X and y differ in length")

    train, test = split_indices(len(X), test_frac=test_frac, seed=seed)
    design_train = np.c_[np.ones(len(train)), X[train]]
    design_test = np.c_[np.ones(len(test)), X[test]]
    y_train, y_test = y[train], y[test]
    coef = np.zeros(design_train.shape[1])
    train_mse = np.zeros(epochs)
    test_mse = np.zeros(epochs)

    for step in range(epochs):
        residual = design_train @ coef - y_train
        grad = design_train.T @ residual / len(y_train)
        coef = coef - lr * grad
        train_mse[step] = mse(y_train, design_train @ coef)
        test_mse[step] = mse(y_test, design_test @ coef)

    return RateRun(float(lr), train_mse, test_mse, coef)
