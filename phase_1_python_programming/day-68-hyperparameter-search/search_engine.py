"""Day 68 — pick the setting on a validation slice.

k = 1 makes the training error zero because each row is its own neighbor.
That is the worst k on a held-out slice. The search should read the
validation column, then score the winner once on the test rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_curve(n: int = 280, noise: float = 0.35, seed: int = 68):
    """A bend plus noise. Neighbors help; copying one neighbor does not."""
    if n < 30:
        raise ValueError("need at least 30 rows")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n)
    y = np.sin(1.4 * x) + noise * rng.normal(size=n)
    return x, y


def three_way(n: int, train_frac: float = 0.5, val_frac: float = 0.25, seed: int = 68):
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("train and val fractions should leave a test slice")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    if n_train < 2 or n_val < 1 or n_train + n_val >= n:
        raise ValueError("split ate a slice")
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def knn_predict(x_train, y_train, x_query, k: int):
    x_train = np.asarray(x_train, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    x_query = np.asarray(x_query, dtype=float).reshape(-1)
    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train differ in length")
    if k < 1 or k > len(x_train):
        raise ValueError("k should be between 1 and the number of training rows")
    # one row of distances per query
    dist = np.abs(x_train - x_query[:, None])
    neighbors = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    return y_train[neighbors].mean(axis=1)


def mse(y, pred) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if len(y) != len(pred) or len(y) == 0:
        raise ValueError("y and pred should be the same non-empty length")
    return float(np.mean((y - pred) ** 2))


@dataclass
class Trial:
    k: int
    train_mse: float
    val_mse: float
    test_mse: float


def choose_by_val(trials) -> Trial:
    """Smaller validation error wins. A tie keeps the smaller k.

    test_mse is sitting on the trial for the writeup. It is not the key.
    """
    if not trials:
        raise ValueError("no trials")
    return min(trials, key=lambda trial: (trial.val_mse, trial.k))


def run_search(x, y, ks, seed: int = 68, train_frac: float = 0.5, val_frac: float = 0.25):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("x and y differ in length")
    ks = [int(k) for k in ks]
    if len(ks) == 0:
        raise ValueError("no k values")

    train_idx, val_idx, test_idx = three_way(len(x), train_frac=train_frac, val_frac=val_frac, seed=seed)
    x_train, y_train = x[train_idx], y[train_idx]
    trials = []
    for k in ks:
        pred_train = knn_predict(x_train, y_train, x_train, k)
        pred_val = knn_predict(x_train, y_train, x[val_idx], k)
        pred_test = knn_predict(x_train, y_train, x[test_idx], k)
        trials.append(
            Trial(
                k=k,
                train_mse=mse(y_train, pred_train),
                val_mse=mse(y[val_idx], pred_val),
                test_mse=mse(y[test_idx], pred_test),
            )
        )
    return trials
