"""Day 69 — a confident probability is not the same as a right one.

Dividing the logit by a temperature below 1 pushes 0.7 toward 0.9 and
0.3 toward 0.1. The 0.5 calls stay put, so accuracy does not move, but
the probabilities now overstate how often the event happens.
A second logistic, fit on the validation rows only, pulls them back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -25, 25)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def make_labels(n: int = 600, seed: int = 69):
    """Two noisy features and a label drawn from a known logistic curve."""
    if n < 40:
        raise ValueError("need at least 40 rows")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    logits = -0.15 + 1.3 * X[:, 0] - 0.7 * X[:, 1]
    prob = _sigmoid(logits)
    y = (rng.random(n) < prob).astype(int)
    return X, y


def split_indices(n: int, seed: int = 69, train_frac: float = 0.5, val_frac: float = 0.25):
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("fractions should leave a test slice")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    if min(n_train, n_val, n - n_train - n_val) < 1:
        raise ValueError("a slice came out empty")
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


class LogisticGD:
    """Binary logistic fit. Columns are standardized from the training rows."""

    def __init__(self, lr: float = 0.4, epochs: int = 500):
        if epochs < 1:
            raise ValueError("need at least one epoch")
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.mu = None
        self.sd = None
        self.coef = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y):
            raise ValueError("X and y differ in length")
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd == 0, 1.0, sd)
        design = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        coef = np.zeros(design.shape[1])
        for _ in range(self.epochs):
            prob = _sigmoid(design @ coef)
            coef = coef - self.lr * (design.T @ (prob - y) / len(y))
        self.coef = coef
        return self

    def predict_proba(self, X):
        if self.coef is None:
            raise RuntimeError("fit first")
        X = np.asarray(X, dtype=float)
        design = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        return _sigmoid(design @ self.coef)


def sharpen(prob, temperature: float):
    """temperature < 1 makes the probabilities more extreme. Ranking stays."""
    if temperature <= 0:
        raise ValueError("temperature should be positive")
    return _sigmoid(_logit(prob) / float(temperature))


def brier(y, prob) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    if len(y) != len(prob) or len(y) == 0:
        raise ValueError("y and prob should be the same non-empty length")
    return float(np.mean((prob - y) ** 2))


def accuracy_at_half(y, prob) -> float:
    y = np.asarray(y).reshape(-1).astype(int)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    calls = (prob >= 0.5).astype(int)
    return float(np.mean(calls == y))


@dataclass
class Bin:
    mean_prob: float
    freq: float
    count: int


def reliability_bins(y, prob, n_bins: int = 8):
    """Mean predicted probability against the fraction of positives in each bin."""
    if n_bins < 2:
        raise ValueError("need at least two bins")
    y = np.asarray(y, dtype=float).reshape(-1)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(Bin(float(prob[mask].mean()), float(y[mask].mean()), count))
    return bins


def expected_calibration_error(y, prob, n_bins: int = 8) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    bins = reliability_bins(y, prob, n_bins=n_bins)
    total = 0.0
    n = len(y)
    for item in bins:
        total += (item.count / n) * abs(item.mean_prob - item.freq)
    return float(total)


def fit_platt(prob, y, lr: float = 0.15, epochs: int = 800):
    """Logistic map from logit(prob) to the label. Fit this on validation rows."""
    y = np.asarray(y, dtype=float).reshape(-1)
    z = _logit(prob).reshape(-1)
    if len(z) != len(y) or len(y) == 0:
        raise ValueError("prob and y should be the same non-empty length")
    design = np.c_[np.ones(len(z)), z]
    # start at the identity map so a short run does not wander off
    coef = np.array([0.0, 1.0])
    for _ in range(epochs):
        pred = _sigmoid(design @ coef)
        coef = coef - lr * (design.T @ (pred - y) / len(y))
    return coef


def apply_platt(prob, coef):
    coef = np.asarray(coef, dtype=float).reshape(-1)
    z = _logit(prob).reshape(-1)
    return _sigmoid(coef[0] + coef[1] * z)


@dataclass
class CalScore:
    name: str
    accuracy: float
    brier: float
    ece: float
    bins: list


def score_probs(name, y, prob, n_bins: int = 8) -> CalScore:
    return CalScore(
        name=name,
        accuracy=accuracy_at_half(y, prob),
        brier=brier(y, prob),
        ece=expected_calibration_error(y, prob, n_bins=n_bins),
        bins=reliability_bins(y, prob, n_bins=n_bins),
    )


def holdout_compare(X, y, temperature: float = 0.5, seed: int = 69, n_bins: int = 8):
    """Fit on train, sharpen, then undo the sharpen using validation labels only."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1).astype(int)
    train, val, test = split_indices(len(X), seed=seed)
    model = LogisticGD().fit(X[train], y[train])
    raw_val = model.predict_proba(X[val])
    raw_test = model.predict_proba(X[test])
    sharp_val = sharpen(raw_val, temperature)
    sharp_test = sharpen(raw_test, temperature)
    platt_coef = fit_platt(sharp_val, y[val])
    fixed_test = apply_platt(sharp_test, platt_coef)
    y_test = y[test]
    return {
        "raw": score_probs("raw", y_test, raw_test, n_bins=n_bins),
        "sharp": score_probs("sharp", y_test, sharp_test, n_bins=n_bins),
        "platt": score_probs("platt", y_test, fixed_test, n_bins=n_bins),
        "platt_coef": platt_coef,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
    }
