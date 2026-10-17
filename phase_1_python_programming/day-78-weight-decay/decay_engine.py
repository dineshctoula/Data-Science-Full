"""Day 78 — weight decay inside gradient descent.

36 training rows and 42 columns. Plain gradient descent can almost
memorize that, noise columns included. Weight decay adds a pull toward
zero on every slope at every step. A little of it trims the noise
weights. A lot of it flattens the real slopes as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_wide(n: int = 80, n_noise: int = 40, seed: int = 78):
    """y = 2 x0 - 1.2 x1 plus noise. Every other column is junk."""
    if n < 20:
        raise ValueError("need at least 20 rows")
    if n_noise < 0:
        raise ValueError("n_noise can't be negative")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2 + n_noise))
    y = 2.0 * X[:, 0] - 1.2 * X[:, 1] + rng.normal(0, 1.0, size=n)
    return X, y


def three_way(n: int, train_frac: float = 0.45, val_frac: float = 0.25, seed: int = 78):
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("train and val fractions should leave a test slice")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    if n_train < 2 or n_val < 1 or n_train + n_val >= n:
        raise ValueError("split ate a slice")
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def fit_decay(X, y, decay: float, lr: float = 0.1, epochs: int = 600) -> np.ndarray:
    """Full-batch least squares with weight decay. Returns intercept first.

    The intercept is left alone. Shrinking it would just drag every
    prediction toward zero, which has nothing to do with overfitting.
    """
    if decay < 0:
        raise ValueError("decay can't be negative")
    if lr <= 0:
        raise ValueError("learning rate should be positive")
    if epochs < 1:
        raise ValueError("need at least one epoch")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(X) != len(y):
        raise ValueError("X and y differ in length")

    design = np.c_[np.ones(len(X)), X]
    coef = np.zeros(design.shape[1])
    for _ in range(epochs):
        grad = design.T @ (design @ coef - y) / len(y)
        pull = decay * coef
        pull[0] = 0.0
        coef = coef - lr * (grad + pull)
    return coef


def mse_of(X, y, coef) -> float:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.c_[np.ones(len(X)), X] @ coef
    return float(np.mean((y - pred) ** 2))


@dataclass
class DecayTrial:
    decay: float
    train_mse: float
    val_mse: float
    test_mse: float
    noise_norm: float  # size of the junk-column weights, not the real two
    x0: float
    x1: float


def sweep(X, y, decays, seed: int = 78, lr: float = 0.1, epochs: int = 600):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    decays = [float(d) for d in decays]
    if not decays:
        raise ValueError("no decay values")
    train, val, test = three_way(len(X), seed=seed)
    trials = []
    for decay in decays:
        coef = fit_decay(X[train], y[train], decay, lr=lr, epochs=epochs)
        trials.append(
            DecayTrial(
                decay=decay,
                train_mse=mse_of(X[train], y[train], coef),
                val_mse=mse_of(X[val], y[val], coef),
                test_mse=mse_of(X[test], y[test], coef),
                # coef[0] is the intercept, [1] and [2] are the real columns
                noise_norm=float(np.linalg.norm(coef[3:])),
                x0=float(coef[1]),
                x1=float(coef[2]),
            )
        )
    return trials, (len(train), len(val), len(test))


def pick_by_val(trials) -> DecayTrial:
    """Lowest validation MSE. A tie goes to the smaller decay."""
    if not trials:
        raise ValueError("no trials")
    return min(trials, key=lambda trial: (trial.val_mse, trial.decay))
