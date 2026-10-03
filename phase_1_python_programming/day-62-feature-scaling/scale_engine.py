"""Day 62 — feature scaling.

Age is around 35. Income is around 55,000. A nearest-neighbor search just
adds squared differences, so income swamps age unless you put them on a
similar scale first. Fit the scaler on the training rows only — the
validation slice shouldn't leak into the mean and spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("X needs to be a non-empty 2d matrix")
    if not np.isfinite(X).all():
        raise ValueError("X has non-finite values")
    return X


class StandardScaler:
    """Subtract the mean, divide by the std. A constant column stays 0."""

    def fit(self, X):
        X = _as_matrix(X)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "mean_"):
            raise RuntimeError("call fit() before transform()")
        X = _as_matrix(X)
        if X.shape[1] != self.n_features_:
            raise ValueError("wrong number of columns")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = _as_matrix(Z)
        return Z * self.scale_ + self.mean_


class MinMaxScaler:
    """Squash each column into [0, 1]. A flat column becomes 0."""

    def fit(self, X):
        X = _as_matrix(X)
        self.min_ = X.min(axis=0)
        span = X.max(axis=0) - self.min_
        span[span < 1e-12] = 1.0
        self.span_ = span
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "min_"):
            raise RuntimeError("call fit() before transform()")
        X = _as_matrix(X)
        if X.shape[1] != self.n_features_:
            raise ValueError("wrong number of columns")
        return (X - self.min_) / self.span_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def make_age_income(n: int = 220, seed: int = 62):
    """Both columns matter equally in z-score space. Raw units hide that."""
    rng = np.random.default_rng(seed)
    age = rng.normal(35, 8, size=n)
    income = rng.normal(55000, 12000, size=n)
    age_z = (age - 35) / 8
    income_z = (income - 55000) / 12000
    logits = 0.2 + 1.1 * age_z - 1.1 * income_z
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    y = (rng.random(n) < probs).astype(int)
    X = np.column_stack([age, income])
    return X, y, ("age", "income")


def train_val_split(X, y, val_frac: float = 0.3, seed: int = 62):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(round(len(X) * val_frac)))
    val, train = idx[:n_val], idx[n_val:]
    return X[train], X[val], y[train], y[val]


class KNNClassifier:
    """Euclidean k=5. No scaling inside — that's the point of the lesson."""

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k should be >= 1")
        self.k = int(k)

    def fit(self, X, y):
        self.X_ = _as_matrix(X)
        self.y_ = np.asarray(y).reshape(-1)
        if len(self.X_) != len(self.y_):
            raise ValueError("X and y length mismatch")
        return self

    def predict(self, X):
        X = _as_matrix(X)
        d2 = ((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(d2, self.k - 1, axis=1)[:, : self.k]
        votes = self.y_[idx]
        # majority along the neighbor axis; ties go to the smaller label
        preds = []
        for row in votes:
            vals, counts = np.unique(row, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.asarray(preds)


@dataclass
class ScaleCompare:
    name: str
    val_acc: float

    def summary(self) -> str:
        return f"{self.name:<16} val={self.val_acc:.3f}"


def knn_accuracy(Xtr, ytr, Xva, yva, k: int = 5) -> float:
    model = KNNClassifier(k=k).fit(Xtr, ytr)
    pred = model.predict(Xva)
    return float(np.mean(pred == yva))


def compare_scalers(X, y, k: int = 5, val_frac: float = 0.3, seed: int = 62) -> list[ScaleCompare]:
    """Fit scalers on train only, then score kNN on the validation slice."""
    Xtr, Xva, ytr, yva = train_val_split(X, y, val_frac=val_frac, seed=seed)
    raw = knn_accuracy(Xtr, ytr, Xva, yva, k=k)

    std = StandardScaler().fit(Xtr)
    std_acc = knn_accuracy(std.transform(Xtr), ytr, std.transform(Xva), yva, k=k)

    mm = MinMaxScaler().fit(Xtr)
    mm_acc = knn_accuracy(mm.transform(Xtr), ytr, mm.transform(Xva), yva, k=k)

    return [
        ScaleCompare("raw", raw),
        ScaleCompare("standard", std_acc),
        ScaleCompare("minmax", mm_acc),
    ]
