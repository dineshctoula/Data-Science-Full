"""Day 53 — linear soft-margin SVM (primal, SGD).

Find a wide street between two classes. Points on the wrong side of the
margin get a hinge penalty; C trades off "wide street" vs "few mistakes".
No kernels today — just w·x + b, trained with plain stochastic gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def hinge(y, margin) -> float:
    """max(0, 1 - y * margin). Zero when the point is past the margin."""
    return float(max(0.0, 1.0 - y * margin))


@dataclass
class SVMResult:
    accuracy: float
    predictions: np.ndarray
    n_sv_approx: int  # how many train points sit on/inside the margin
    C: float

    def summary(self) -> str:
        return (
            f"linear SVM (C={self.C}) → accuracy={self.accuracy:.3f}, "
            f"~margin points={self.n_sv_approx}"
        )


class LinearSVM:
    def __init__(
        self,
        C: float = 1.0,
        lr: float = 0.01,
        n_epochs: int = 40,
        seed: int = 53,
    ):
        if C <= 0:
            raise ValueError("C must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if n_epochs < 1:
            raise ValueError("n_epochs should be >= 1")
        self.C = float(C)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.loss_curve_: list[float] = []
        self._labels = (-1.0, 1.0)  # internal encoding

    def _to_pm1(self, y):
        y = np.asarray(y).reshape(-1)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("binary SVM wants exactly two classes")
        # map smaller label → -1, larger → +1 (stable)
        self._class_map = {classes[0]: -1.0, classes[1]: 1.0}
        self._inv_map = {-1.0: classes[0], 1.0: classes[1]}
        return np.array([self._class_map[v] for v in y], dtype=float)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X needs to be a 2d feature matrix")
        if not np.isfinite(X).all():
            raise ValueError("X has non-finite values")
        y_pm = self._to_pm1(y)
        if len(y_pm) != len(X):
            raise ValueError("X and y length mismatch")

        n, p = X.shape
        rng = np.random.default_rng(self.seed)
        w = np.zeros(p)
        b = 0.0
        self.loss_curve_ = []

        # objective ≈ 0.5||w||^2 + C * sum hinge
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            # mild lr decay so we don't bounce forever
            eta = self.lr / (1.0 + 0.05 * epoch)
            for i in order:
                xi, yi = X[i], y_pm[i]
                margin = float(np.dot(w, xi) + b)
                # always pull w toward 0 a little (the 0.5||w||^2 bit)
                w *= 1.0 - eta
                if yi * margin < 1.0:
                    # point is inside/on wrong side of margin → push it out
                    w += eta * self.C * yi * xi
                    b += eta * self.C * yi
            # track full-batch hinge loss after each epoch
            margins = X @ w + b
            loss = 0.5 * float(np.dot(w, w)) + self.C * float(
                np.mean([hinge(yi, m) for yi, m in zip(y_pm, margins)])
            )
            self.loss_curve_.append(loss)

        self.w_ = w
        self.b_ = float(b)
        self._X_train = X.copy()
        self._y_train = y_pm.copy()
        return self

    def decision_function(self, X):
        if self.w_ is None:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.w_ + self.b_

    def predict(self, X):
        scores = self.decision_function(X)
        pm = np.where(scores >= 0.0, 1.0, -1.0)
        return np.array([self._inv_map[v] for v in pm])

    def margin_mask(self, X=None, y=None, tol: float = 1e-3):
        """Points with y*(w·x+b) <= 1 (+tol) — the soft 'support' set."""
        if self.w_ is None:
            raise RuntimeError("call fit() first")
        if X is None:
            X, y_pm = self._X_train, self._y_train
        else:
            X = np.asarray(X, dtype=float)
            if y is None:
                raise ValueError("y required when X is passed")
            # reuse the mapping from fit — don't rebuild class maps
            y = np.asarray(y).reshape(-1)
            y_pm = np.array([self._class_map[v] for v in y], dtype=float)
        margins = y_pm * self.decision_function(X)
        return margins <= 1.0 + tol

    def score(self, X, y) -> SVMResult:
        y = np.asarray(y).reshape(-1)
        preds = self.predict(X)
        acc = float(np.mean(preds == y))
        n_sv = int(self.margin_mask().sum()) if self.w_ is not None else 0
        return SVMResult(accuracy=acc, predictions=preds, n_sv_approx=n_sv, C=self.C)


def make_linearly_separable(n_per: int = 40, seed: int = 53):
    rng = np.random.default_rng(seed)
    c0 = rng.normal([-1.5, 0.0], 0.35, size=(n_per, 2))
    c1 = rng.normal([1.5, 0.2], 0.35, size=(n_per, 2))
    X = np.vstack([c0, c1])
    y = np.array([0] * n_per + [1] * n_per)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_soft_overlap(n_per: int = 50, seed: int = 53):
    """Closer blobs — needs soft margin / bigger C tradeoff."""
    rng = np.random.default_rng(seed)
    c0 = rng.normal([-0.6, 0.0], 0.55, size=(n_per, 2))
    c1 = rng.normal([0.6, 0.1], 0.55, size=(n_per, 2))
    X = np.vstack([c0, c1])
    y = np.array([0] * n_per + [1] * n_per)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train_test_split(X, y, test_frac: float = 0.3, seed: int = 53):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(1, int(round(len(X) * test_frac)))
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]
