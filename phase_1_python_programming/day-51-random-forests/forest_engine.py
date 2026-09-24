"""Day 51 — random forests (bagging + random feature subsets).

One noisy tree overfits. A bunch of them, each trained on a bootstrap sample
and only allowed to look at a random handful of features per split, tend to
cancel each other's quirks when you majority-vote.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np


def gini(labels) -> float:
    y = np.asarray(labels).reshape(-1)
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p * p))


def majority_label(labels):
    y = np.asarray(labels).reshape(-1)
    vals, counts = np.unique(y, return_counts=True)
    return vals[np.argmax(counts)]


@dataclass
class _Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    prediction: Optional[float] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class _Tree:
    """Single CART tree with optional max_features (the RF secret sauce)."""

    def __init__(self, max_depth=8, min_samples_split=2, max_features=None, rng=None):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features  # None → use all columns
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root: Optional[_Node] = None
        self.n_features_: int = 0

    def _best_split(self, X, y):
        n, p = X.shape
        # only poke at a random subset of columns (or all of them)
        if self.max_features is None or self.max_features >= p:
            feats = np.arange(p)
        else:
            k = max(1, int(self.max_features))
            feats = self.rng.choice(p, size=k, replace=False)

        parent = gini(y)
        best_gain, best_j, best_thr = -1.0, None, None

        for j in feats:
            vals = np.unique(X[:, j])
            if len(vals) < 2:
                continue
            # midpoints between sorted uniques — same trick as Day 50
            for thr in (vals[:-1] + vals[1:]) / 2.0:
                left = X[:, j] <= thr
                n_l = int(left.sum())
                if n_l == 0 or n_l == n:
                    continue
                child = (n_l / n) * gini(y[left]) + ((n - n_l) / n) * gini(y[~left])
                gain = parent - child
                if gain > best_gain:
                    best_gain, best_j, best_thr = gain, int(j), float(thr)

        return best_j, best_thr, best_gain

    def _grow(self, X, y, depth):
        node = _Node(prediction=majority_label(y))
        if gini(y) == 0.0 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return node

        j, thr, gain = self._best_split(X, y)
        if j is None or gain <= 1e-12:
            return node

        left = X[:, j] <= thr
        node.feature, node.threshold = j, thr
        node.left = self._grow(X[left], y[left], depth + 1)
        node.right = self._grow(X[~left], y[~left], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        self.n_features_ = X.shape[1]
        self.root = self._grow(X, y, depth=0)
        return self

    def _one(self, row, node):
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._one(row, self.root) for row in X])


@dataclass
class ForestResult:
    accuracy: float
    predictions: np.ndarray
    n_trees: int
    oob_accuracy: Optional[float] = None

    def summary(self) -> str:
        base = f"forest ({self.n_trees} trees) → accuracy={self.accuracy:.3f}"
        if self.oob_accuracy is not None:
            base += f", oob={self.oob_accuracy:.3f}"
        return base


class RandomForestClassifier:
    def __init__(
        self,
        n_trees: int = 20,
        max_depth: int = 8,
        min_samples_split: int = 2,
        max_features="sqrt",  # "sqrt" | "log2" | int | None
        seed: int = 51,
    ):
        if n_trees < 1:
            raise ValueError("n_trees should be at least 1")
        if max_depth < 1:
            raise ValueError("max_depth should be at least 1")
        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.seed = int(seed)
        self.trees_: list[_Tree] = []
        # for each tree: which training indices were *out* of the bootstrap bag
        self._oob_indices: list[np.ndarray] = []
        self._n_features: int = 0

    def _resolve_max_features(self, p: int) -> Optional[int]:
        mf = self.max_features
        if mf is None:
            return None
        if mf == "sqrt":
            return max(1, int(np.sqrt(p)))
        if mf == "log2":
            return max(1, int(np.log2(p)))
        if isinstance(mf, (int, np.integer)):
            if int(mf) < 1:
                raise ValueError("max_features int must be >= 1")
            return min(int(mf), p)
        raise ValueError("max_features must be 'sqrt', 'log2', an int, or None")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X needs to be a 2d feature matrix")
        if len(y) != len(X):
            raise ValueError("X and y length mismatch")
        if not np.isfinite(X).all():
            raise ValueError("X has non-finite values")
        if np.unique(y).size < 2:
            raise ValueError("need at least two classes")

        n, p = X.shape
        self._n_features = p
        mf = self._resolve_max_features(p)
        rng = np.random.default_rng(self.seed)

        self.trees_ = []
        self._oob_indices = []
        for t in range(self.n_trees):
            # bootstrap sample — draw n rows with replacement
            bag = rng.integers(0, n, size=n)
            oob = np.setdiff1d(np.arange(n), np.unique(bag), assume_unique=False)
            tree_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
            tree = _Tree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=mf,
                rng=tree_rng,
            )
            tree.fit(X[bag], y[bag])
            self.trees_.append(tree)
            self._oob_indices.append(oob)

        return self

    def predict(self, X):
        if not self.trees_:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._n_features:
            raise ValueError("wrong number of features")

        # stack votes: shape (n_trees, n_rows)
        votes = np.vstack([t.predict(X) for t in self.trees_])
        preds = []
        for col in range(votes.shape[1]):
            preds.append(majority_label(votes[:, col]))
        return np.asarray(preds)

    def predict_proba_binaryish(self, X):
        """Fraction of trees voting for the higher label. Fine for 2-class demos."""
        if not self.trees_:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        votes = np.vstack([t.predict(X) for t in self.trees_])
        # assume labels are whatever showed up; report P(max label)
        classes = np.unique(votes)
        hi = classes.max()
        return np.mean(votes == hi, axis=0)

    def oob_score(self, X, y) -> float:
        """Accuracy using only trees that never saw each row — free validation."""
        if not self.trees_:
            raise RuntimeError("call fit() first")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        n = len(X)
        # collect votes per row from trees where that row was OOB
        buckets: list[list] = [[] for _ in range(n)]
        for tree, oob in zip(self.trees_, self._oob_indices):
            if len(oob) == 0:
                continue
            preds = tree.predict(X[oob])
            for i, pred in zip(oob, preds):
                buckets[i].append(pred)

        correct, counted = 0, 0
        for i, votes in enumerate(buckets):
            if not votes:
                continue  # rare: row somehow in every bag
            counted += 1
            if majority_label(votes) == y[i]:
                correct += 1
        if counted == 0:
            return float("nan")
        return correct / counted

    def score(self, X, y) -> ForestResult:
        y = np.asarray(y).reshape(-1)
        preds = self.predict(X)
        acc = float(np.mean(preds == y))
        oob = None
        # only compute OOB if X looks like the training set size we fitted on
        # (caller can also call oob_score explicitly)
        return ForestResult(
            accuracy=acc,
            predictions=preds,
            n_trees=len(self.trees_),
            oob_accuracy=oob,
        )


def make_noisy_moons(n_per_class: int = 60, noise: float = 0.25, seed: int = 51):
    """Two interlocking half-circles with enough noise that one tree struggles."""
    rng = np.random.default_rng(seed)
    # upper moon
    t = np.linspace(0, np.pi, n_per_class)
    x0 = np.c_[np.cos(t), np.sin(t)]
    # lower moon, shifted
    x1 = np.c_[1 - np.cos(t), 1 - np.sin(t) - 0.5]
    X = np.vstack([x0, x1])
    X += rng.normal(0, noise, size=X.shape)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_wide_features(n: int = 120, p: int = 8, seed: int = 51):
    """More features than we need — only first two actually matter.

    Good demo for why random feature subsets help: single trees latch onto
    noise columns; the forest averages that away.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # label from a soft XOR on x0, x1, ignore the rest
    signal = (X[:, 0] > 0).astype(float) != (X[:, 1] > 0).astype(float)
    y = signal.astype(int)
    # flip a few labels so it's not trivial
    flip = rng.choice(n, size=max(1, n // 20), replace=False)
    y[flip] = 1 - y[flip]
    return X, y


def train_test_split(X, y, test_frac: float = 0.3, seed: int = 51):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(1, int(round(len(X) * test_frac)))
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


def compare_n_trees(X, y, n_list=(1, 5, 10, 25), seed: int = 51) -> dict:
    """Fit forests of growing size; return test-set accuracies via a fixed split."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.3, seed=seed)
    out = {}
    for n in n_list:
        forest = RandomForestClassifier(n_trees=n, seed=seed).fit(Xtr, ytr)
        out[n] = forest.score(Xte, yte).accuracy
    return out
