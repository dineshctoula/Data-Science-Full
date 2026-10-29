"""Day 87 — ridge regression on the log-ratio to the baseline.

target  r = log1p(y) - log1p(baseline)
model   r ≈ b0 + Z @ beta         (Z = standardized features, b0 not penalized)
predict y_hat = expm1(log1p(baseline) + b0 + Z @ beta), floored at 0

Fitting on the log scale makes every effect a percentage: rain takes off
some share of the riders an hour would have had, whether it's 3 a.m. or
5 p.m. It also lands nearer the typical hour than the average one, which
is what MAE rewards: the exp of a mean log is a median-like value, not a mean.

The ridge solution is closed form: (A'A + n·lam·P) beta = A'r, with P the
identity except a 0 for the intercept, as on Days 78 and 82.
"""

from __future__ import annotations

import numpy as np


class CorrectionRidge:
    def __init__(self, lam: float = 1e-2):
        if lam < 0:
            raise ValueError("lam should be zero or more")
        self.lam = lam
        self.mean = None
        self.std = None
        self.coef = None
        self.names = None

    def fit(self, X, y, base):
        # keep the column names before X becomes a plain array
        self.names = tuple(X.columns) if hasattr(X, "columns") else None
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        base = np.asarray(base, dtype=float).reshape(-1)
        if X.ndim != 2 or not len(X) == len(y) == len(base) or len(y) == 0:
            raise ValueError("X, y, and base need one row per hour")
        if not (np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(base).all()):
            raise ValueError("drop hours with missing features before fitting")
        if (y < 0).any() or (base < 0).any():
            raise ValueError("counts can't be negative")
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        # a column that never varies in training (e.g. a rare holiday mismatch) is left unscaled
        self.std[self.std == 0] = 1.0
        Z = (X - self.mean) / self.std
        A = np.column_stack([np.ones(len(Z)), Z])
        r = np.log1p(y) - np.log1p(base)
        penalty = np.full(A.shape[1], self.lam * len(Z))
        penalty[0] = 0.0
        self.coef = np.linalg.solve(A.T @ A + np.diag(penalty), A.T @ r)
        return self

    def log_ratio(self, X) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("fit first")
        Z = (np.asarray(X, dtype=float) - self.mean) / self.std
        return self.coef[0] + Z @ self.coef[1:]

    def predict(self, X, base) -> np.ndarray:
        base = np.asarray(base, dtype=float).reshape(-1)
        # rentals can't go below zero, whatever the log-ratio says
        return np.maximum(0.0, np.expm1(np.log1p(base) + self.log_ratio(X)))

    def raw_effects(self) -> np.ndarray:
        """Slopes per one unit of each original column (not per std), on the log-ratio scale."""
        if self.coef is None:
            raise RuntimeError("fit first")
        return self.coef[1:] / self.std
