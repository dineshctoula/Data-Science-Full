"""Day 65 — leakage from a prep step that saw the holdout.

Picking the "best" columns by correlating them with y on the whole table
uses the answers from the rows you later call a test set. The test error
comes out too pretty. Split first, then correlate, then fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_wide_table(n: int = 80, n_noise: int = 40, seed: int = 65):
    """Two real columns and a pile of noise. Small n makes a leak easier to see."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n, 2))
    noise = rng.normal(size=(n, n_noise))
    X = np.column_stack([signal, noise])
    y = 2.0 * signal[:, 0] - 1.4 * signal[:, 1] + rng.normal(0, 1.0, size=n)
    return X, y


def split_indices(n: int, test_frac: float = 0.35, seed: int = 65):
    if not 0 < test_frac < 1:
        raise ValueError("test_frac should be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    if n_test >= n:
        raise ValueError("test slice ate the whole table")
    return idx[n_test:], idx[:n_test]


def abs_correlation(X, y):
    """|corr| of each column with y. A constant column scores 0, not NaN."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    y_center = y - y.mean()
    y_ss = float(np.dot(y_center, y_center))
    scores = np.zeros(X.shape[1])
    if y_ss == 0:
        return scores
    for j in range(X.shape[1]):
        col = X[:, j]
        c = col - col.mean()
        ss = float(np.dot(c, c))
        if ss == 0:
            continue
        scores[j] = abs(float(np.dot(c, y_center)) / np.sqrt(ss * y_ss))
    return scores


def top_features(X, y, k: int = 3):
    if k < 1 or k > X.shape[1]:
        raise ValueError("k is outside the number of columns")
    scores = abs_correlation(X, y)
    # mergesort keeps the lower index when two scores tie
    order = np.argsort(-scores, kind="mergesort")
    return order[:k]


def _ols_mse(Xtr, ytr, Xte, yte) -> float:
    design = np.c_[np.ones(len(Xtr)), Xtr]
    beta = np.linalg.lstsq(design, ytr, rcond=None)[0]
    pred = np.c_[np.ones(len(Xte)), Xte] @ beta
    return float(np.mean((yte - pred) ** 2))


@dataclass
class LeakCompare:
    honest_mse: float
    leaky_mse: float
    honest_cols: tuple
    leaky_cols: tuple

    @property
    def optimism(self) -> float:
        # positive means the leaky number looks better than it should
        return self.honest_mse - self.leaky_mse

    def summary(self) -> str:
        return (
            f"honest MSE={self.honest_mse:.3f}  leaky MSE={self.leaky_mse:.3f}  "
            f"gap={self.optimism:.3f}"
        )


def compare_selection(X, y, k: int = 3, test_frac: float = 0.35, seed: int = 65) -> LeakCompare:
    """Same split. Leaky version ranks columns using every row, including test y."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    train, test = split_indices(len(X), test_frac=test_frac, seed=seed)

    honest_cols = tuple(int(j) for j in top_features(X[train], y[train], k=k))
    leaky_cols = tuple(int(j) for j in top_features(X, y, k=k))

    honest = _ols_mse(X[np.ix_(train, honest_cols)], y[train], X[np.ix_(test, honest_cols)], y[test])
    leaky = _ols_mse(X[np.ix_(train, leaky_cols)], y[train], X[np.ix_(test, leaky_cols)], y[test])
    return LeakCompare(honest, leaky, honest_cols, leaky_cols)


def repeat_gap(X_factory, n_seeds: int = 20, k: int = 3, base_seed: int = 65) -> np.ndarray:
    """One optimism number per seed. X_factory(seed) -> X, y."""
    gaps = []
    for i in range(n_seeds):
        seed = base_seed + i
        X, y = X_factory(seed)
        gaps.append(compare_selection(X, y, k=k, seed=seed).optimism)
    return np.asarray(gaps, dtype=float)
