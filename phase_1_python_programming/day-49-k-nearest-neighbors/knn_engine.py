"""Day 49 — k-nearest neighbors.

Idea is simple: look at the k closest training points and let them vote.
No training step really — we just store the data and search at predict time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class KNNResult:
    """What we get back after scoring a labeled set with a chosen k."""

    k: int
    predictions: np.ndarray
    accuracy: float
    distance: str

    def summary(self) -> str:
        return f"k={self.k} ({self.distance}) → accuracy={self.accuracy:.3f}"


class KNearestNeighbors:
    def __init__(self, k: int = 5, distance: str = "euclidean"):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k should be a positive int")
        if distance not in {"euclidean", "manhattan"}:
            raise ValueError("distance must be euclidean or manhattan")
        self.k = int(k)
        self.distance = distance
        self._X = None
        self._y = None

    def fit(self, X, y):
        """Just remember the training points. That's the whole 'model'."""
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
        if self.k > len(X):
            raise ValueError(f"k={self.k} is bigger than the training set ({len(X)})")
        # store copies so callers can mutate their arrays later without breaking us
        self._X = X.copy()
        self._y = y.copy()
        return self

    def _pairwise_distances(self, query):
        if self._X is None:
            raise RuntimeError("call fit() before predict()")
        q = np.asarray(query, dtype=float)
        if q.ndim == 1:
            # single row passed as a flat vector
            q = q.reshape(1, -1)
        if q.shape[1] != self._X.shape[1]:
            raise ValueError("query has the wrong number of features")

        # broadcast: (n_query, 1, p) - (1, n_train, p)
        diff = q[:, None, :] - self._X[None, :, :]
        if self.distance == "euclidean":
            return np.sqrt((diff ** 2).sum(axis=2))
        # manhattan = sum of abs diffs
        return np.abs(diff).sum(axis=2)

    def _vote(self, neighbor_labels, neighbor_distances):
        """Majority vote. Ties → pick the label that showed up with smaller total distance."""
        counts = Counter(neighbor_labels.tolist())
        top = max(counts.values())
        tied = [lab for lab, c in counts.items() if c == top]
        if len(tied) == 1:
            return tied[0]
        # break ties by closeness (sum of distances among the tied labels)
        best_lab = tied[0]
        best_dist = float("inf")
        for lab in tied:
            mask = neighbor_labels == lab
            total = float(neighbor_distances[mask].sum())
            if total < best_dist:
                best_dist = total
                best_lab = lab
        return best_lab

    def predict(self, X):
        dists = self._pairwise_distances(X)
        preds = []
        for row in dists:
            # argsort gives nearest → farthest; take the first k
            idx = np.argsort(row)[: self.k]
            preds.append(self._vote(self._y[idx], row[idx]))
        return np.asarray(preds)

    def score(self, X, y) -> KNNResult:
        y = np.asarray(y).reshape(-1)
        preds = self.predict(X)
        if len(preds) != len(y):
            raise ValueError("prediction/label length mismatch")
        acc = float(np.mean(preds == y))
        return KNNResult(k=self.k, predictions=preds, accuracy=acc, distance=self.distance)


def leave_one_out_accuracy(X, y, k: int = 5, distance: str = "euclidean") -> float:
    """LOOCV — slow but honest for small toy sets.

    For each point: train on everyone else, predict that one point.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(X)
    if k >= n:
        raise ValueError("for LOOCV, k must be < n")
    correct = 0
    for i in range(n):
        # bit awkward indexing-wise but keeps the hold-out clear
        train_X = np.delete(X, i, axis=0)
        train_y = np.delete(y, i, axis=0)
        model = KNearestNeighbors(k=k, distance=distance).fit(train_X, train_y)
        pred = model.predict(X[i])
        if pred[0] == y[i]:
            correct += 1
    return correct / n


def compare_k_values(X, y, ks=None, distance: str = "euclidean") -> dict[int, float]:
    """Try a few k's with LOOCV so we can see the bias/variance vibe."""
    if ks is None:
        ks = [1, 3, 5, 9, 15]
    out = {}
    for k in ks:
        # skip illegal k instead of crashing the whole sweep
        if k >= len(X):
            continue
        out[int(k)] = leave_one_out_accuracy(X, y, k=int(k), distance=distance)
    if not out:
        raise ValueError("no valid k values for this dataset size")
    return out


def make_two_moons_ish(n_per_class: int = 60, noise: float = 0.18, seed: int = 49):
    """Hand-rolled interlocking clusters (not sklearn's make_moons, but same vibe)."""
    if n_per_class < 5:
        raise ValueError("need a few points per class")
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n_per_class)
    # upper crescent
    x1 = np.column_stack((np.cos(t), np.sin(t)))
    # lower crescent shifted to the right/down a bit
    x0 = np.column_stack((1 - np.cos(t), 0.5 - np.sin(t)))
    X = np.vstack((x0, x1))
    y = np.concatenate((np.zeros(n_per_class), np.ones(n_per_class)))
    X = X + rng.normal(0.0, noise, size=X.shape)
    # shuffle so we don't accidentally rely on class order
    order = rng.permutation(len(X))
    return X[order], y[order]


def make_blob_classes(n_per_class: int = 40, seed: int = 49):
    """Three messy blobs — easier than moons, good for k=1 vs larger k demos."""
    rng = np.random.default_rng(seed)
    centers = [(0, 0), (3.2, 2.5), (0.5, 3.8)]
    chunks = []
    labels = []
    for i, (cx, cy) in enumerate(centers):
        pts = rng.normal([cx, cy], [0.55, 0.55], size=(n_per_class, 2))
        chunks.append(pts)
        labels.append(np.full(n_per_class, float(i)))
    X = np.vstack(chunks)
    y = np.concatenate(labels)
    order = rng.permutation(len(X))
    return X[order], y[order]
