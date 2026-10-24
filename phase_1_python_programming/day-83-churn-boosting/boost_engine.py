"""Day 83 — gradient-boosted trees for churn, with log loss and histogram splits.

Day 52's booster fit squared-error trees to residuals and tried every
threshold. Two changes make it fit a 0/1 target and run fast enough
for cross-validation:

1. Newton boosting on log loss. For each row, g = p - y is the slope
   of the loss in the current log-odds and h = p (1 - p) is its
   curvature. A leaf's best value is -sum(g) / (sum(h) + lam), and a
   split is worth
       0.5 * (G_L^2 / (H_L + lam) + G_R^2 / (H_R + lam) - G^2 / (H + lam)).
   This is the XGBoost recipe, and the same Newton step as Day 82, done
   per leaf.
2. Histogram splits. Each column is cut once into at most `max_bins`
   bins on the training rows. A node then sums g and h per bin with one
   bincount, and every threshold's gain comes from a running sum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class Binner:
    """Learn cut points per column on train; map any rows to small integer codes.

    Code k means "above the first k cut points", so a split "code <= k"
    is the same as "x <= cuts[k]".
    """

    def __init__(self, max_bins: int = 32):
        if max_bins < 2:
            raise ValueError("max_bins should be at least 2")
        self.max_bins = max_bins
        self.cuts = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.cuts = []
        for j in range(X.shape[1]):
            values = np.unique(X[:, j])
            if len(values) <= self.max_bins:
                # few distinct values (every one-hot column): cut after each one but the last
                cuts = values[:-1]
            else:
                # many values: cut at quantiles, snapped to values that really occur
                qs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
                cuts = np.unique(np.quantile(X[:, j], qs, method="lower"))
            self.cuts.append(cuts)
        return self

    def transform(self, X):
        if self.cuts is None:
            raise RuntimeError("fit first")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != len(self.cuts):
            raise ValueError("column count differs from fit")
        codes = np.empty(X.shape, dtype=np.int32)
        for j, cuts in enumerate(self.cuts):
            # side="left": a value equal to a cut point gets that cut's code, i.e. goes left
            codes[:, j] = np.searchsorted(cuts, X[:, j], side="left")
        return codes

    @property
    def n_bins(self) -> int:
        return max(len(c) for c in self.cuts) + 1


@dataclass
class Tree:
    """Flat arrays; node 0 is the root and feature == -1 marks a leaf."""

    feature: np.ndarray
    code: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray
    gain: np.ndarray

    def predict(self, codes):
        node = np.zeros(len(codes), dtype=np.int64)
        rows = np.arange(len(codes))
        active = self.feature[node] >= 0
        # every row walks one level per pass until it sits on a leaf
        while active.any():
            r, nd = rows[active], node[active]
            go_left = codes[r, self.feature[nd]] <= self.code[nd]
            node[r] = np.where(go_left, self.left[nd], self.right[nd])
            active = self.feature[node] >= 0
        return self.value[node]


def build_tree(codes, g, h, n_bins, max_depth=3, lam=1.0, min_child_weight=1.0, min_gain=0.0):
    """Grow one tree depth-first on the Newton gains."""
    n, d = codes.shape
    feature, code, left, right, value, gain = [], [], [], [], [], []
    # one offset per column so a single bincount fills every column's histogram at once
    offsets = (np.arange(d) * n_bins)[None, :]

    def new_node():
        for arr in (feature, code, left, right, value, gain):
            arr.append(0)
        feature[-1] = -1
        return len(feature) - 1

    def grow(node, rows, depth):
        G, H = g[rows].sum(), h[rows].sum()
        value[node] = -G / (H + lam)
        if depth >= max_depth or len(rows) < 2:
            return
        flat = (codes[rows] + offsets).ravel()
        hist_g = np.bincount(flat, weights=np.repeat(g[rows], d), minlength=d * n_bins).reshape(d, n_bins)
        hist_h = np.bincount(flat, weights=np.repeat(h[rows], d), minlength=d * n_bins).reshape(d, n_bins)
        # left side of "code <= k" is the running sum up to bin k; the last bin would leave the right side empty
        GL = np.cumsum(hist_g, axis=1)[:, :-1]
        HL = np.cumsum(hist_h, axis=1)[:, :-1]
        GR, HR = G - GL, H - HL
        gains = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam))
        # a child with too little curvature is too small to trust its leaf value
        gains[(HL < min_child_weight) | (HR < min_child_weight)] = -np.inf
        j, k = np.unravel_index(np.argmax(gains), gains.shape)
        best = gains[j, k]
        if not np.isfinite(best) or best <= min_gain:
            return
        go_left = codes[rows, j] <= k
        feature[node], code[node], gain[node] = int(j), int(k), float(best)
        left[node] = new_node()
        right[node] = new_node()
        grow(left[node], rows[go_left], depth + 1)
        grow(right[node], rows[~go_left], depth + 1)

    root = new_node()
    grow(root, np.arange(n), 0)
    return Tree(
        np.array(feature, dtype=np.int64),
        np.array(code, dtype=np.int64),
        np.array(left, dtype=np.int64),
        np.array(right, dtype=np.int64),
        np.array(value, dtype=float),
        np.array(gain, dtype=float),
    )


class BoostedTrees:
    def __init__(
        self,
        n_trees: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        lam: float = 1.0,
        min_child_weight: float = 1.0,
        max_bins: int = 32,
    ):
        if n_trees < 1 or max_depth < 1:
            raise ValueError("n_trees and max_depth should be at least 1")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate should be in (0, 1]")
        if lam < 0 or min_child_weight < 0:
            raise ValueError("lam and min_child_weight should be zero or more")
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lam = lam
        self.min_child_weight = min_child_weight
        self.max_bins = max_bins
        self.binner = None
        self.base = None
        self.trees = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        if not set(np.unique(y)) <= {0.0, 1.0}:
            raise ValueError("y should be 0/1")
        self.binner = Binner(self.max_bins).fit(X)
        codes = self.binner.transform(X)
        if len(codes) != len(y):
            raise ValueError("X and y differ in length")
        n_bins = self.binner.n_bins
        rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        # start from the base rate's log-odds, so tree 1 explains what the rate can't
        self.base = float(np.log(rate / (1 - rate)))
        F = np.full(len(y), self.base)
        self.trees = []
        for _ in range(self.n_trees):
            p = sigmoid(F)
            g, h = p - y, p * (1 - p)
            tree = build_tree(codes, g, h, n_bins, self.max_depth, self.lam, self.min_child_weight)
            # the learning rate shrinks every leaf, so no single tree takes a big step
            tree.value *= self.learning_rate
            F += tree.predict(codes)
            self.trees.append(tree)
        return self

    def staged_decision(self, X):
        """Log-odds after 1, 2, ..., n_trees trees: shape (n_trees, n_rows)."""
        if self.trees is None:
            raise RuntimeError("fit first")
        codes = self.binner.transform(X)
        F = np.full(len(codes), self.base)
        out = np.empty((len(self.trees), len(codes)))
        for t, tree in enumerate(self.trees):
            F = F + tree.predict(codes)
            out[t] = F
        return out

    def predict_proba(self, X, n_trees: int | None = None):
        if self.trees is None:
            raise RuntimeError("fit first")
        use = len(self.trees) if n_trees is None else n_trees
        if not 1 <= use <= len(self.trees):
            raise ValueError("n_trees out of range")
        codes = self.binner.transform(X)
        F = np.full(len(codes), self.base)
        for tree in self.trees[:use]:
            F += tree.predict(codes)
        return sigmoid(F)

    def gain_by_column(self, n_columns: int) -> np.ndarray:
        """Total split gain credited to each column across all trees."""
        total = np.zeros(n_columns)
        for tree in self.trees:
            split = tree.feature >= 0
            np.add.at(total, tree.feature[split], tree.gain[split])
        return total
