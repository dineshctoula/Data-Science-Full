"""Day 70 — shuffle a column and see if the error moves.

A fitted model can carry a column that never mattered. Permute that
column on the validation rows, score again, and the error should barely
change. Permute a column the fit actually uses and the error jumps.
The coefficients stay as they were. Only the validation copy is shuffled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_table(n: int = 240, seed: int = 70):
    """y depends on columns 0 and 1. Columns 2 and 3 are noise."""
    if n < 40:
        raise ValueError("need at least 40 rows")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 2.2 * X[:, 0] - 1.4 * X[:, 1] + 0.45 * rng.normal(size=n)
    return X, y


def split_indices(n: int, test_frac: float = 0.3, seed: int = 70):
    if not 0 < test_frac < 1:
        raise ValueError("test_frac should be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    if n_test >= n:
        raise ValueError("test slice ate the whole table")
    return idx[n_test:], idx[:n_test]


def fit_ols(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(X) != len(y):
        raise ValueError("X and y differ in length")
    design = np.c_[np.ones(len(X)), X]
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def predict_ols(X, coef):
    X = np.asarray(X, dtype=float)
    coef = np.asarray(coef, dtype=float).reshape(-1)
    return np.c_[np.ones(len(X)), X] @ coef


def mse(y, pred) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if len(y) != len(pred) or len(y) == 0:
        raise ValueError("y and pred should be the same non-empty length")
    return float(np.mean((y - pred) ** 2))


@dataclass
class ColumnRise:
    index: int
    mean_rise: float
    std_rise: float


def permutation_rises(X_train, y_train, X_val, y_val, repeats: int = 20, seed: int = 70):
    """How much validation MSE rises when one column is shuffled, averaged."""
    if repeats < 1:
        raise ValueError("need at least one repeat")
    X_train = np.asarray(X_train, dtype=float)
    X_val = np.asarray(X_val, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("train and val should have the same columns")

    coef = fit_ols(X_train, y_train)
    base = mse(y_val, predict_ols(X_val, coef))
    rng = np.random.default_rng(seed)
    rises = []
    for j in range(X_val.shape[1]):
        draws = []
        for _ in range(repeats):
            shuffled = X_val.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            draws.append(mse(y_val, predict_ols(shuffled, coef)) - base)
        rises.append(ColumnRise(j, float(np.mean(draws)), float(np.std(draws))))
    return base, coef, rises
