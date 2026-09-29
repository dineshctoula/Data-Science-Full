"""Day 58 — ridge and lasso.

Plain least squares uses every column, junk included. Ridge pulls every
coefficient toward zero but rarely kills one. Lasso can set some exactly
to zero, which is handy when half the columns are noise.

Intercept is not penalized in either fit. Otherwise a big mean just
looks like a coefficient that needs shrinking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def soft_threshold(z, gamma):
    """The lasso update: pull z toward 0 by gamma, stop at 0."""
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0


@dataclass
class LinearFit:
    intercept: float
    coef: np.ndarray
    kind: str
    lam: float

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.intercept + X @ self.coef

    def summary(self) -> str:
        nz = int(np.sum(np.abs(self.coef) > 1e-8))
        return (
            f"{self.kind} λ={self.lam:g} → intercept={self.intercept:.3f}, "
            f"nonzero coefs={nz}/{len(self.coef)}"
        )


def _as_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("X needs to be a non-empty 2d matrix")
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("non-finite values in X or y")
    return X, y


def fit_ridge(X, y, lam: float = 1.0) -> LinearFit:
    """Solve (X'X + λI) beta = X'y. Column 0 of the design is the intercept.

    λ = 0 is ordinary least squares. The penalty matrix has a 0 in the
    intercept slot so we don't shrink the mean.
    """
    if lam < 0:
        raise ValueError("lambda must be >= 0")
    X, y = _as_xy(X, y)
    n, p = X.shape
    design = np.c_[np.ones(n), X]
    penalty = np.eye(p + 1)
    penalty[0, 0] = 0.0
    xtx = design.T @ design + lam * penalty
    xty = design.T @ y
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        # rare: duplicate columns and lam=0. fall back to least squares
        beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    return LinearFit(intercept=float(beta[0]), coef=beta[1:].copy(), kind="ridge", lam=float(lam))


def fit_lasso(X, y, lam: float = 1.0, max_iter: int = 400, tol: float = 1e-5) -> LinearFit:
    """Coordinate descent on centered columns.

    Loss is 0.5 * ||y - Xb||^2 + λ ||b||_1. The 0.5 keeps the update as
    soft_threshold(x_j · r, λ) / ||x_j||^2.
    """
    if lam < 0:
        raise ValueError("lambda must be >= 0")
    if max_iter < 1:
        raise ValueError("max_iter should be >= 1")
    X, y = _as_xy(X, y)
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean
    n, p = Xc.shape
    # column energy; a constant column can't be used
    col_norm2 = np.sum(Xc * Xc, axis=0)
    beta = np.zeros(p)
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        old = beta.copy()
        for j in range(p):
            if col_norm2[j] < 1e-12:
                beta[j] = 0.0
                continue
            # residual as if feature j weren't in the model yet
            resid = yc - Xc @ beta + Xc[:, j] * beta[j]
            rho = float(Xc[:, j] @ resid)
            beta[j] = soft_threshold(rho, lam) / col_norm2[j]
        if np.max(np.abs(beta - old)) < tol:
            break

    intercept = y_mean - float(x_mean @ beta)
    fit = LinearFit(intercept=intercept, coef=beta.copy(), kind="lasso", lam=float(lam))
    fit.n_iter_ = n_iter  # handy when a path refuses to settle
    return fit


def mse(y, pred) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    return float(np.mean((y - pred) ** 2))


def coefficient_path(X, y, lams, kind: str = "lasso") -> np.ndarray:
    """Shape (len(lams), n_features). One row of coefficients per lambda."""
    lams = list(lams)
    if kind not in {"ridge", "lasso"}:
        raise ValueError("kind must be ridge or lasso")
    rows = []
    fitter = fit_lasso if kind == "lasso" else fit_ridge
    for lam in lams:
        rows.append(fitter(X, y, lam=float(lam)).coef)
    return np.vstack(rows)


def make_sparse_regression(n: int = 120, seed: int = 58):
    """Two real signals, three noise columns, and one copy of x0.

    The copy is there so ridge and lasso disagree: ridge splits the weight
    across x0 and its twin, lasso tends to keep one and drop the other.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    twin = x0 + rng.normal(scale=0.05, size=n)
    noise = rng.normal(size=(n, 3))
    X = np.column_stack([x0, x1, twin, noise])
    # true weights: 3, -2, and zeros. twin is redundant with x0.
    y = 3.0 * x0 - 2.0 * x1 + rng.normal(scale=0.6, size=n)
    names = ("x0", "x1", "x0_copy", "noise_a", "noise_b", "noise_c")
    return X, y, names


def train_test_split(X, y, test_frac: float = 0.3, seed: int = 58):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(1, int(round(len(X) * test_frac)))
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test]
