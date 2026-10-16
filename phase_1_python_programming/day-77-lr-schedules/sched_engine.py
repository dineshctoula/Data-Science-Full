"""Day 77 — a large learning rate that only works if it shrinks.

Same least-squares line as the rate and mini-batch days. Start at 1.8.
Left alone, the fit never settles. Cut the rate in half every 15 epochs,
or cool it with a cosine, and the test error comes back near the noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_line(n: int = 240, seed: int = 77):
    if n < 20:
        raise ValueError("need at least 20 rows")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 1.0, size=n)
    return X, y


def split_indices(n: int, test_frac: float = 0.3, seed: int = 77):
    if not 0 < test_frac < 1:
        raise ValueError("test_frac should be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    if n_test >= n:
        raise ValueError("test slice ate the whole table")
    return idx[n_test:], idx[:n_test]


def constant_lr(epoch: int, base: float, epochs: int) -> float:
    return float(base)


def step_decay_lr(epoch: int, base: float, epochs: int, every: int = 15) -> float:
    """Halve the rate every `every` epochs. epoch is 1-based."""
    if every < 1:
        raise ValueError("every should be at least 1")
    return float(base * (0.5 ** ((epoch - 1) // every)))


def cosine_lr(epoch: int, base: float, epochs: int) -> float:
    """Cosine from base down toward 0 over the run."""
    if epochs < 2:
        return float(base)
    # (epoch-1)/(epochs-1) goes 0 -> 1 across the run
    frac = (epoch - 1) / (epochs - 1)
    return float(base * 0.5 * (1.0 + np.cos(np.pi * frac)))


@dataclass
class SchedRun:
    name: str
    lrs: np.ndarray
    train_mse: np.ndarray
    test_mse: np.ndarray

    @property
    def epochs(self) -> int:
        return int(len(self.lrs))


def fit_schedule(X, y, schedule, name: str, base_lr: float = 1.8, epochs: int = 60, test_frac: float = 0.3, seed: int = 77) -> SchedRun:
    if base_lr <= 0:
        raise ValueError("base learning rate should be positive")
    if epochs < 1:
        raise ValueError("need at least one epoch")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    train, test = split_indices(len(X), test_frac=test_frac, seed=seed)
    design_train = np.c_[np.ones(len(train)), X[train]]
    design_test = np.c_[np.ones(len(test)), X[test]]
    y_train, y_test = y[train], y[test]
    coef = np.zeros(design_train.shape[1])
    lrs = np.zeros(epochs)
    train_mse = np.zeros(epochs)
    test_mse = np.zeros(epochs)

    for step in range(epochs):
        lr = float(schedule(step + 1, base_lr, epochs))
        if lr < 0:
            raise ValueError("schedule returned a negative rate")
        residual = design_train @ coef - y_train
        coef = coef - lr * (design_train.T @ residual / len(y_train))
        lrs[step] = lr
        train_mse[step] = float(np.mean((y_train - design_train @ coef) ** 2))
        test_mse[step] = float(np.mean((y_test - design_test @ coef) ** 2))

    return SchedRun(name, lrs, train_mse, test_mse)
