"""Day 55 — DBSCAN.

K-means wants round piles and a chosen k. DBSCAN just asks: is this point
in a dense neighborhood? Core points grow a cluster; lonely points stay noise.
eps = neighborhood radius, min_samples = how many neighbors make you "core".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NOISE = -1


def euclidean_neighbors(X, i, eps):
    """Indices within eps of row i (includes i itself)."""
    diff = X - X[i]
    d = np.sqrt((diff * diff).sum(axis=1))
    return np.flatnonzero(d <= eps)


@dataclass
class DBSCANResult:
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    eps: float
    min_samples: int

    def summary(self) -> str:
        return (
            f"dbscan(eps={self.eps}, min_samples={self.min_samples}) → "
            f"{self.n_clusters} clusters, {self.n_noise} noise"
        )


class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_samples < 1:
            raise ValueError("min_samples should be >= 1")
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.labels_: np.ndarray | None = None
        self.core_mask_: np.ndarray | None = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X needs to be a non-empty 2d matrix")
        if not np.isfinite(X).all():
            raise ValueError("X has non-finite values")

        n = len(X)
        labels = np.full(n, NOISE, dtype=int)
        visited = np.zeros(n, dtype=bool)
        core_mask = np.zeros(n, dtype=bool)
        cluster_id = 0

        # precompute neighborhoods once — toy n, so O(n^2) is fine
        neighborhoods = [euclidean_neighbors(X, i, self.eps) for i in range(n)]

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = neighborhoods[i]
            if len(neighbors) < self.min_samples:
                # not dense enough to start a cluster (might still get
                # claimed later as a border point)
                continue

            core_mask[i] = True
            labels[i] = cluster_id
            # grow from the seed — list so we can append newly found cores
            seeds = list(neighbors)
            seen_in_seeds = set(int(j) for j in neighbors)
            ptr = 0
            while ptr < len(seeds):
                j = seeds[ptr]
                ptr += 1
                if not visited[j]:
                    visited[j] = True
                    neigh_j = neighborhoods[j]
                    if len(neigh_j) >= self.min_samples:
                        core_mask[j] = True
                        for k in neigh_j:
                            k = int(k)
                            if k not in seen_in_seeds:
                                seeds.append(k)
                                seen_in_seeds.add(k)
                # noise or untouched → pull into this cluster
                if labels[j] == NOISE:
                    labels[j] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        self.core_mask_ = core_mask
        self._X = X.copy()
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_.copy()

    def result(self) -> DBSCANResult:
        if self.labels_ is None:
            raise RuntimeError("call fit() first")
        n_clusters = int(len(set(self.labels_) - {NOISE}))
        n_noise = int(np.sum(self.labels_ == NOISE))
        return DBSCANResult(
            labels=self.labels_.copy(),
            n_clusters=n_clusters,
            n_noise=n_noise,
            eps=self.eps,
            min_samples=self.min_samples,
        )


def make_two_moons(n_per: int = 60, noise: float = 0.08, seed: int = 55):
    """Interlocking moons — k-means hates these, density clustering doesn't."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n_per)
    upper = np.c_[np.cos(t), np.sin(t)]
    lower = np.c_[1 - np.cos(t), -np.sin(t) + 0.5]
    X = np.vstack([upper, lower])
    X += rng.normal(0, noise, size=X.shape)
    y = np.array([0] * n_per + [1] * n_per)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_blobs_with_outliers(n_per: int = 40, n_out: int = 12, seed: int = 55):
    """Two tight blobs plus some junk floating around."""
    rng = np.random.default_rng(seed)
    a = rng.normal([-1.4, 0.0], 0.25, size=(n_per, 2))
    b = rng.normal([1.4, 0.2], 0.25, size=(n_per, 2))
    junk = rng.uniform(-3.5, 3.5, size=(n_out, 2))
    X = np.vstack([a, b, junk])
    # ground truth: 0, 1, and -1 for junk
    y = np.array([0] * n_per + [1] * n_per + [NOISE] * n_out)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]
