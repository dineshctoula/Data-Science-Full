"""Day 60 — phase 3 wrap-up.

One small exam-score example that uses a few of the ideas from this phase
without dragging every earlier file in: a summary, a correlation, ordinary
least squares, a holdout, and a bootstrap interval on the hours slope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_exam_scores(n: int = 160, seed: int = 60, noise: float = 6.0):
    """Hours, sleep, and a prior score. The exam is mostly hours."""
    rng = np.random.default_rng(seed)
    hours = np.clip(rng.normal(5.5, 1.6, size=n), 0.5, 12)
    sleep = np.clip(rng.normal(7.0, 0.9, size=n), 4.0, 10)
    prior = rng.normal(72, 8, size=n)
    score = (
        22
        + 4.2 * hours
        + 1.5 * sleep
        + 0.30 * prior
        + rng.normal(0, noise, size=n)
    )
    X = np.column_stack([hours, sleep, prior])
    names = ("hours", "sleep", "prior")
    return X, score, names


def column_summary(X, y, names):
    """Mean, sample std, and correlation with the target for each column."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X and y don't line up")
    rows = []
    y_std = float(y.std(ddof=1))
    for j, name in enumerate(names):
        col = X[:, j]
        col_std = float(col.std(ddof=1))
        if col_std == 0 or y_std == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(col, y)[0, 1])
        rows.append(
            {
                "name": name,
                "mean": float(col.mean()),
                "std": col_std,
                "corr_with_y": corr,
            }
        )
    return rows


@dataclass
class OLSFit:
    intercept: float
    coef: np.ndarray
    names: tuple

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.intercept + X @ self.coef

    def summary(self) -> str:
        bits = ", ".join(f"{n}={c:.2f}" for n, c in zip(self.names, self.coef))
        return f"ols intercept={self.intercept:.2f}, {bits}"


def fit_ols(X, y, names=None) -> OLSFit:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(X) != len(y) or len(X) == 0:
        raise ValueError("X and y length mismatch")
    if names is None:
        names = tuple(f"x{j}" for j in range(X.shape[1]))
    design = np.c_[np.ones(len(X)), X]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return OLSFit(intercept=float(beta[0]), coef=beta[1:].copy(), names=tuple(names))


def r2_score(y, pred) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    ss_res = float(np.sum((y - pred) ** 2))
    return 1.0 - ss_res / ss_tot


def mse(y, pred) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    return float(np.mean((y - pred) ** 2))


def train_test_split(X, y, test_frac: float = 0.25, seed: int = 60):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(1, int(round(len(X) * test_frac)))
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test]


def bootstrap_slope(X, y, feature: int = 0, n_boot: int = 200, seed: int = 60):
    """Percentile interval for one coefficient. Resample rows with replacement."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if n_boot < 20:
        raise ValueError("n_boot should be at least 20")
    rng = np.random.default_rng(seed)
    n = len(X)
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        take = rng.integers(0, n, size=n)
        fit = fit_ols(X[take], y[take])
        slopes[i] = fit.coef[feature]
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return {
        "mean": float(slopes.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "samples": slopes,
    }
