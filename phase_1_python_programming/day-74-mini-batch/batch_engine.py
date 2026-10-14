"""Day 74 — a mini-batch is several steps per pass.

Full batch waits until it has seen every training row, then takes one
step. A batch of 32 takes a step every 32 rows, so one epoch is a
handful of updates and the path is a bit noisy. A batch of one updates
on every row. Same learning rate. "Epoch" does not mean the same amount
of work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_line(n: int = 240, seed: int = 74):
    """y = 3 x0 - 1.5 x1 plus noise. Columns are already standard scale."""
    if n < 20:
        raise ValueError("need at least 20 rows")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 1.0, size=n)
    return X, y


def split_indices(n: int, test_frac: float = 0.3, seed: int = 74):
    if not 0 < test_frac < 1:
        raise ValueError("test_frac should be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    if n_test >= n:
        raise ValueError("test slice ate the whole table")
    return idx[n_test:], idx[:n_test]


@dataclass
class BatchRun:
    batch_size: int
    updates_per_epoch: int
    train_mse: np.ndarray
    test_mse: np.ndarray

    @property
    def epochs(self) -> int:
        return int(len(self.train_mse))


def fit_batches(X, y, batch_size: int, lr: float = 0.2, epochs: int = 20, test_frac: float = 0.3, seed: int = 74) -> BatchRun:
    """Shuffle the training rows each epoch and step on chunks of batch_size.

    A batch_size at least as big as the training slice is just full batch.
    """
    if batch_size < 1:
        raise ValueError("batch_size should be at least 1")
    if lr <= 0:
        raise ValueError("learning rate should be positive")
    if epochs < 1:
        raise ValueError("need at least one epoch")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(X) != len(y):
        raise ValueError("X and y differ in length")

    train, test = split_indices(len(X), test_frac=test_frac, seed=seed)
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    used = min(int(batch_size), len(train))
    # seed+1 so the shuffle is not the same stream as the split
    order_rng = np.random.default_rng(seed + 1)
    coef = np.zeros(X.shape[1] + 1)
    train_mse = np.zeros(epochs)
    test_mse = np.zeros(epochs)
    updates = int(np.ceil(len(train) / used))

    design_test = np.c_[np.ones(len(test)), X_test]
    for step in range(epochs):
        order = order_rng.permutation(len(train))
        for start in range(0, len(train), used):
            sl = order[start : start + used]
            design = np.c_[np.ones(len(sl)), X_train[sl]]
            residual = design @ coef - y_train[sl]
            coef = coef - lr * (design.T @ residual / len(sl))
        design_train = np.c_[np.ones(len(train)), X_train]
        train_mse[step] = float(np.mean((y_train - design_train @ coef) ** 2))
        test_mse[step] = float(np.mean((y_test - design_test @ coef) ** 2))

    return BatchRun(used, updates, train_mse, test_mse)
