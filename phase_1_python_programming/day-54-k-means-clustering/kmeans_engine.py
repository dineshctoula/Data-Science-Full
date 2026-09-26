"""Day 54 — k-means clustering.

Pick k centers, assign every point to the nearest one, move each center to
the mean of its members, repeat. No labels — just "which pile do you belong
to?" Inertia (sum of squared distances to your center) is how we keep score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def pairwise_sq_distances(X, centers):
    """(n, k) matrix of squared euclidean distances — no sqrt needed for argmin."""
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    x2 = np.sum(X * X, axis=1)[:, None]
    c2 = np.sum(centers * centers, axis=1)[None, :]
    return np.maximum(x2 + c2 - 2.0 * X @ centers.T, 0.0)


def inertia(X, labels, centers) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels).reshape(-1)
    centers = np.asarray(centers, dtype=float)
    total = 0.0
    for k in range(len(centers)):
        members = X[labels == k]
        if len(members) == 0:
            continue
        total += float(np.sum((members - centers[k]) ** 2))
    return total


@dataclass
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int
    k: int

    def summary(self) -> str:
        return f"k={self.k} → inertia={self.inertia:.3f} after {self.n_iter} iters"


class KMeans:
    def __init__(
        self,
        k: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 5,
        seed: int = 54,
    ):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k should be a positive int")
        if max_iter < 1:
            raise ValueError("max_iter should be >= 1")
        if n_init < 1:
            raise ValueError("n_init should be >= 1")
        self.k = int(k)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.seed = int(seed)
        self.centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0
        self.inertia_curve_: list[float] = []

    def _init_centers(self, X, rng):
        """k-means++ style-ish: first random, then prefer far-away points.

        Not a perfect ++ implementation, but better than pure random picks.
        """
        n = len(X)
        centers = np.empty((self.k, X.shape[1]), dtype=float)
        centers[0] = X[rng.integers(0, n)]
        for j in range(1, self.k):
            d2 = pairwise_sq_distances(X, centers[:j]).min(axis=1)
            # sample proportional to distance^2 (classic ++)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n) / n
            centers[j] = X[rng.choice(n, p=probs)]
        return centers

    def _run_once(self, X, rng):
        centers = self._init_centers(X, rng)
        labels = np.zeros(len(X), dtype=int)
        curve = []

        for it in range(1, self.max_iter + 1):
            d2 = pairwise_sq_distances(X, centers)
            labels = np.argmin(d2, axis=1)

            new_centers = centers.copy()
            for j in range(self.k):
                members = X[labels == j]
                if len(members) == 0:
                    # empty cluster — kick the center to a random point
                    new_centers[j] = X[rng.integers(0, len(X))]
                else:
                    new_centers[j] = members.mean(axis=0)

            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            curve.append(inertia(X, labels, centers))
            if shift < self.tol:
                return labels, centers, curve, it

        return labels, centers, curve, self.max_iter

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X needs to be a non-empty 2d matrix")
        if not np.isfinite(X).all():
            raise ValueError("X has non-finite values")
        if self.k > len(X):
            raise ValueError(f"k={self.k} is bigger than n={len(X)}")

        rng = np.random.default_rng(self.seed)
        best_inertia = float("inf")
        best = None

        # a few random restarts — k-means is picky about init
        for _ in range(self.n_init):
            # fresh sub-rng so restarts aren't clones
            sub = np.random.default_rng(rng.integers(0, 2**31 - 1))
            labels, centers, curve, n_iter = self._run_once(X, sub)
            ine = curve[-1]
            if ine < best_inertia:
                best_inertia = ine
                best = (labels, centers, curve, n_iter)

        self.labels_, self.centers_, self.inertia_curve_, self.n_iter_ = best
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        if self.centers_ is None:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.argmin(pairwise_sq_distances(X, self.centers_), axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_.copy()

    def result(self) -> KMeansResult:
        if self.centers_ is None:
            raise RuntimeError("call fit() first")
        return KMeansResult(
            labels=self.labels_.copy(),
            centers=self.centers_.copy(),
            inertia=self.inertia_,
            n_iter=self.n_iter_,
            k=self.k,
        )


def elbow_scores(X, ks=range(1, 8), seed: int = 54) -> dict:
    """Inertia for each k — the 'elbow' plot data."""
    out = {}
    for k in ks:
        km = KMeans(k=k, n_init=5, seed=seed + k).fit(X)
        out[int(k)] = km.inertia_
    return out


def make_blobs(k: int = 3, n_per: int = 40, seed: int = 54):
    rng = np.random.default_rng(seed)
    # spread centers around a circle so they don't sit on top of each other
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centers = np.c_[2.2 * np.cos(angles), 2.2 * np.sin(angles)]
    chunks = []
    labels = []
    for j, c in enumerate(centers):
        chunks.append(rng.normal(c, 0.45, size=(n_per, 2)))
        labels.append(np.full(n_per, j))
    X = np.vstack(chunks)
    y = np.concatenate(labels)  # ground truth only for checking, not for fitting
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_uneven_blobs(seed: int = 54):
    """Two tight piles + one spread-out one — k-means struggles a bit here."""
    rng = np.random.default_rng(seed)
    a = rng.normal([-2.0, 0.0], 0.3, size=(50, 2))
    b = rng.normal([2.0, 0.2], 0.3, size=(50, 2))
    c = rng.normal([0.0, 2.5], 0.9, size=(40, 2))  # noisier cloud
    X = np.vstack([a, b, c])
    y = np.array([0] * 50 + [1] * 50 + [2] * 40)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]
