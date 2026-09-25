"""Day 52 — gradient boosting (the sequential kind).

Unlike a random forest, each new tree looks at what the current ensemble
still gets wrong. Fit a shallow tree to the residuals, shrink it a bit
(learning_rate), add it on, repeat. Squared loss keeps the math boring
on purpose — residuals are just y - F(x).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    value: float = 0.0  # leaf = mean residual in that bucket

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _leaf_value(residuals) -> float:
    r = np.asarray(residuals, dtype=float).reshape(-1)
    if len(r) == 0:
        return 0.0
    return float(r.mean())


def _sse(residuals) -> float:
    r = np.asarray(residuals, dtype=float).reshape(-1)
    if len(r) == 0:
        return 0.0
    return float(np.sum((r - r.mean()) ** 2))


class RegressionTree:
    """Tiny CART regressor — weak learner for the booster."""

    def __init__(self, max_depth: int = 2, min_samples_split: int = 2):
        if max_depth < 1:
            raise ValueError("max_depth should be >= 1")
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.root: Optional[_Node] = None

    def _best_split(self, X, residuals):
        n, p = X.shape
        parent_sse = _sse(residuals)
        best_j, best_thr, best_drop = None, None, -1.0

        for j in range(p):
            vals = np.unique(X[:, j])
            if len(vals) < 2:
                continue
            for thr in (vals[:-1] + vals[1:]) / 2.0:
                left = X[:, j] <= thr
                n_l = int(left.sum())
                if n_l == 0 or n_l == n:
                    continue
                drop = parent_sse - (_sse(residuals[left]) + _sse(residuals[~left]))
                if drop > best_drop:
                    best_drop = drop
                    best_j = j
                    best_thr = float(thr)
        return best_j, best_thr, best_drop

    def _grow(self, X, residuals, depth):
        node = _Node(value=_leaf_value(residuals))
        if depth >= self.max_depth or len(residuals) < self.min_samples_split:
            return node
        if _sse(residuals) < 1e-12:
            return node

        j, thr, drop = self._best_split(X, residuals)
        if j is None or drop <= 1e-12:
            return node

        left = X[:, j] <= thr
        node.feature, node.threshold = j, thr
        node.left = self._grow(X[left], residuals[left], depth + 1)
        node.right = self._grow(X[~left], residuals[~left], depth + 1)
        return node

    def fit(self, X, residuals):
        X = np.asarray(X, dtype=float)
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        self.root = self._grow(X, residuals, depth=0)
        return self

    def _one(self, row, node):
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        if self.root is None:
            raise RuntimeError("tree isn't fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._one(row, self.root) for row in X], dtype=float)


@dataclass
class BoostResult:
    mse: float
    predictions: np.ndarray
    n_estimators: int
    train_loss_curve: list
    accuracy: Optional[float] = None  # only set for classification wrapper

    def summary(self) -> str:
        msg = f"boost ({self.n_estimators} trees) → mse={self.mse:.4f}"
        if self.accuracy is not None:
            msg += f", accuracy={self.accuracy:.3f}"
        return msg


class GradientBoostingRegressor:
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        min_samples_split: int = 2,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators should be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.trees_: list[RegressionTree] = []
        self.F0_: float = 0.0  # starting constant (mean of y)
        self.train_loss_: list[float] = []
        self._fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("non-finite values in X or y")

        # start with the dumbest model: predict the mean everywhere
        self.F0_ = float(y.mean())
        F = np.full(len(y), self.F0_)
        self.trees_ = []
        self.train_loss_ = []

        for _ in range(self.n_estimators):
            residual = y - F
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X, residual)
            # shrink the step — full residual fit overshoots / overfits fast
            F = F + self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.train_loss_.append(float(np.mean((y - F) ** 2)))

        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        F = np.full(len(X), self.F0_)
        lr = self.learning_rate
        for tree in self.trees_:
            F = F + lr * tree.predict(X)
        return F

    def score(self, X, y) -> BoostResult:
        y = np.asarray(y, dtype=float).reshape(-1)
        preds = self.predict(X)
        mse = float(np.mean((y - preds) ** 2))
        return BoostResult(
            mse=mse,
            predictions=preds,
            n_estimators=len(self.trees_),
            train_loss_curve=list(self.train_loss_),
        )


class GradientBoostingClassifier:
    """Binary classifier via L2 boosting on {0,1} labels.

    Not fancy logistic boosting — we just regress the labels and threshold
    at 0.5. Good enough to see the boosting curve go down.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        min_samples_split: int = 2,
    ):
        self._reg = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("this demo classifier wants exactly two classes")
        self.classes_ = classes
        # map to 0/1 in class order
        y01 = (y == classes[1]).astype(float)
        self._reg.fit(X, y01)
        return self

    def decision_function(self, X):
        return self._reg.predict(X)

    def predict(self, X):
        if self.classes_ is None:
            raise RuntimeError("call fit() before predict()")
        scores = self.decision_function(X)
        return np.where(scores >= 0.5, self.classes_[1], self.classes_[0])

    def score(self, X, y) -> BoostResult:
        y = np.asarray(y).reshape(-1)
        preds = self.predict(X)
        y01 = (y == self.classes_[1]).astype(float)
        raw = self.decision_function(X)
        mse = float(np.mean((y01 - raw) ** 2))
        acc = float(np.mean(preds == y))
        return BoostResult(
            mse=mse,
            predictions=preds,
            n_estimators=len(self._reg.trees_),
            train_loss_curve=list(self._reg.train_loss_),
            accuracy=acc,
        )

    @property
    def train_loss_(self):
        return self._reg.train_loss_


def make_regression_wave(n: int = 120, noise: float = 0.35, seed: int = 52):
    """1-D sine-ish target — nice for watching residual MSE drop."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 1))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 0] + rng.normal(0, noise, size=n)
    return X, y


def make_two_blobs_clf(n_per: int = 50, seed: int = 52):
    rng = np.random.default_rng(seed)
    c0 = rng.normal([0.0, 0.0], 0.55, size=(n_per, 2))
    c1 = rng.normal([2.2, 1.8], 0.55, size=(n_per, 2))
    X = np.vstack([c0, c1])
    y = np.array([0] * n_per + [1] * n_per)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train_test_split(X, y, test_frac: float = 0.3, seed: int = 52):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(1, int(round(len(X) * test_frac)))
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]
