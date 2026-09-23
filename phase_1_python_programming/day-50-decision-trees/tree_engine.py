"""Day 50 — decision trees (CART-ish, binary splits only).

Keep asking: which feature + threshold cleanly separates the labels?
Gini or entropy measures how mixed a bucket is; we pick the split that
drops that messiness the most. Recurse until pure, too small, or too deep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def gini(labels) -> float:
    """1 - sum(p^2). Zero when everyone has the same label."""
    y = np.asarray(labels).reshape(-1)
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p * p))


def entropy(labels) -> float:
    """Shannon entropy in bits-ish (natural log would work the same for ranking)."""
    y = np.asarray(labels).reshape(-1)
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    # skip zeros just in case — though unique shouldn't produce them
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def majority_label(labels):
    y = np.asarray(labels).reshape(-1)
    vals, counts = np.unique(y, return_counts=True)
    # if there's a tie, np.argmax picks the first one — good enough
    return vals[np.argmax(counts)]


@dataclass
class TreeNode:
    """Either a leaf (prediction set) or an internal split."""

    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    prediction: Optional[float] = None
    n_samples: int = 0
    impurity: float = 0.0
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def describe(self, indent: int = 0) -> str:
        pad = "  " * indent
        if self.is_leaf:
            return f"{pad}leaf → class {self.prediction} (n={self.n_samples}, impurity={self.impurity:.3f})"
        line = (
            f"{pad}x[{self.feature}] <= {self.threshold:.3f} "
            f"(n={self.n_samples}, impurity={self.impurity:.3f})"
        )
        return "\n".join([line, self.left.describe(indent + 1), self.right.describe(indent + 1)])


@dataclass
class TreeResult:
    accuracy: float
    predictions: np.ndarray
    depth: int
    n_leaves: int
    criterion: str

    def summary(self) -> str:
        return (
            f"tree ({self.criterion}) depth={self.depth}, leaves={self.n_leaves} "
            f"→ accuracy={self.accuracy:.3f}"
        )


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini",
    ):
        if max_depth < 1:
            raise ValueError("max_depth should be at least 1")
        if min_samples_split < 2:
            raise ValueError("min_samples_split needs to be >= 2")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be gini or entropy")
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.criterion = criterion
        self.root: Optional[TreeNode] = None
        self._feature_names: tuple[str, ...] = ()

    def _impurity(self, labels) -> float:
        return gini(labels) if self.criterion == "gini" else entropy(labels)

    def _best_split(self, X, y):
        """Brute-force over features + midpoints between sorted unique values.

        Not fancy (no histogram shortcuts) but clear, and fine for toy n.
        """
        n, p = X.shape
        parent_imp = self._impurity(y)
        best_gain = -1.0
        best_feat = None
        best_thr = None

        for j in range(p):
            col = X[:, j]
            # unique sorted values → candidate thresholds between neighbors
            vals = np.unique(col)
            if len(vals) < 2:
                continue
            # midpoints — avoids picking an exact training value as the cut
            candidates = (vals[:-1] + vals[1:]) / 2.0
            for thr in candidates:
                left_mask = col <= thr
                # both sides need at least one point or the split is useless
                n_left = int(left_mask.sum())
                if n_left == 0 or n_left == n:
                    continue
                right_mask = ~left_mask
                left_imp = self._impurity(y[left_mask])
                right_imp = self._impurity(y[right_mask])
                # weighted average impurity after the split
                child = (n_left / n) * left_imp + ((n - n_left) / n) * right_imp
                gain = parent_imp - child
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thr = float(thr)

        return best_feat, best_thr, best_gain

    def _grow(self, X, y, depth: int) -> TreeNode:
        node = TreeNode(
            n_samples=len(y),
            impurity=self._impurity(y),
            depth=depth,
            prediction=majority_label(y),
        )

        # stop if pure, too deep, or not enough rows to bother splitting
        if (
            node.impurity == 0.0
            or depth >= self.max_depth
            or len(y) < self.min_samples_split
        ):
            return node

        feat, thr, gain = self._best_split(X, y)
        # gain ~0 means nothing improves — just leaf it
        if feat is None or gain <= 1e-12:
            return node

        left_mask = X[:, feat] <= thr
        node.feature = feat
        node.threshold = thr
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X, y, feature_names=None):
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
            raise ValueError("need at least two classes to build a tree")

        if feature_names is None:
            self._feature_names = tuple(f"x{j}" for j in range(X.shape[1]))
        else:
            names = tuple(str(n) for n in feature_names)
            if len(names) != X.shape[1]:
                raise ValueError("feature_names length doesn't match columns")
            self._feature_names = names

        self.root = self._grow(X.copy(), y.copy(), depth=0)
        return self

    def _predict_one(self, row, node: TreeNode):
        while not node.is_leaf:
            # left = <= threshold (same rule as during growth)
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X):
        if self.root is None:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self._feature_names):
            raise ValueError("wrong number of features")
        return np.array([self._predict_one(row, self.root) for row in X])

    def score(self, X, y) -> TreeResult:
        y = np.asarray(y).reshape(-1)
        preds = self.predict(X)
        acc = float(np.mean(preds == y))
        return TreeResult(
            accuracy=acc,
            predictions=preds,
            depth=self.depth(),
            n_leaves=self.n_leaves(),
            criterion=self.criterion,
        )

    def depth(self) -> int:
        def _d(node: Optional[TreeNode]) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_d(node.left), _d(node.right))

        return _d(self.root)

    def n_leaves(self) -> int:
        def _n(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _n(node.left) + _n(node.right)

        return _n(self.root)

    def print_tree(self) -> str:
        if self.root is None:
            return "(empty tree)"
        return self.root.describe()


def make_axis_aligned_blobs(n_per_class: int = 40, seed: int = 50):
    """Two blobs that a shallow tree can cut with vertical/horizontal lines."""
    rng = np.random.default_rng(seed)
    # class 0 sits lower-left, class 1 upper-right — easy axis cuts
    c0 = rng.normal(loc=[1.0, 1.0], scale=0.45, size=(n_per_class, 2))
    c1 = rng.normal(loc=[3.2, 3.0], scale=0.45, size=(n_per_class, 2))
    X = np.vstack([c0, c1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    # shuffle so we don't accidentally rely on order
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_xor_like(n_per_class: int = 50, seed: int = 50):
    """Classic XOR-ish layout — needs depth > 1 because one cut isn't enough."""
    rng = np.random.default_rng(seed)
    # four corners: (0,0) and (1,1) → class 0; (0,1) and (1,0) → class 1
    half = n_per_class // 2
    corners = [
        (np.array([0.0, 0.0]), 0),
        (np.array([1.0, 1.0]), 0),
        (np.array([0.0, 1.0]), 1),
        (np.array([1.0, 0.0]), 1),
    ]
    chunks_x = []
    chunks_y = []
    for center, lab in corners:
        pts = rng.normal(loc=center, scale=0.12, size=(half, 2))
        chunks_x.append(pts)
        chunks_y.append(np.full(half, lab))
    X = np.vstack(chunks_x)
    y = np.concatenate(chunks_y)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train_test_split(X, y, test_frac: float = 0.25, seed: int = 50):
    """Tiny split helper so main.py doesn't need sklearn."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(1, int(round(len(X) * test_frac)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
