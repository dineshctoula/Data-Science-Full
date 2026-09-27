"""Day 56 — PCA, the covariance way.

Center the columns, look at which directions wiggle together the most,
keep the top few eigenvectors. Projecting onto those is the compressed
version; multiplying back is the reconstruction (a bit blurry if you
dropped components).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAResult:
    n_components: int
    explained_ratio: np.ndarray
    reconstruction_mse: float

    def summary(self) -> str:
        kept = float(self.explained_ratio.sum())
        return (
            f"pca(k={self.n_components}) → kept {kept:.1%} of variance, "
            f"recon MSE={self.reconstruction_mse:.4f}"
        )


class PCA:
    def __init__(self, n_components: int = 2):
        if not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError("n_components should be a positive int")
        self.n_components = int(n_components)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None  # rows = PCs, biggest first
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("need at least 2 rows")
        if not np.isfinite(X).all():
            raise ValueError("X has non-finite values")
        if self.n_components > X.shape[1]:
            raise ValueError("n_components bigger than number of features")

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # sample covariance — divide by n-1, same habit as np.cov
        n = len(X)
        cov = (Xc.T @ Xc) / (n - 1)
        # eigh: ascending eigenvalues, columns are eigenvectors
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)  # tiny negatives from float noise
        evecs = evecs[:, order]

        total = float(evals.sum())
        ratios = evals / total if total > 0 else np.zeros_like(evals)

        self.components_ = evecs.T  # (p, p)
        self.explained_variance_ = evals
        self.explained_variance_ratio_ = ratios
        return self

    def transform(self, X, n_components=None):
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("call fit() before transform()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self.mean_):
            raise ValueError("wrong number of features")
        k = self.n_components if n_components is None else int(n_components)
        if k < 1 or k > self.components_.shape[0]:
            raise ValueError("bad n_components for transform")
        W = self.components_[:k]
        return (X - self.mean_) @ W.T

    def inverse_transform(self, Z):
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("call fit() first")
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        k = Z.shape[1]
        W = self.components_[:k]
        return Z @ W + self.mean_

    def reconstruction_mse(self, X) -> float:
        X = np.asarray(X, dtype=float)
        Z = self.transform(X)
        hat = self.inverse_transform(Z)
        return float(np.mean((X - hat) ** 2))

    def result(self, X) -> PCAResult:
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("call fit() first")
        k = self.n_components
        return PCAResult(
            n_components=k,
            explained_ratio=self.explained_variance_ratio_[:k].copy(),
            reconstruction_mse=self.reconstruction_mse(X),
        )


def make_stretched_cloud(n: int = 200, seed: int = 56):
    """2D blob that's long on one axis — PC1 should hug that axis."""
    rng = np.random.default_rng(seed)
    # generate along principal axes then rotate ~30 degrees
    raw = rng.normal(size=(n, 2))
    raw[:, 0] *= 3.0
    raw[:, 1] *= 0.4
    theta = np.deg2rad(30)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X = raw @ rot.T + np.array([1.5, -0.5])
    return X


def make_noisy_signal(n: int = 150, p: int = 6, seed: int = 56):
    """A 2-D signal buried in extra noise columns.

    First two columns carry the real variation; the rest are small jitter.
    PCA should dump most variance into the first couple of components.
    """
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    signal = np.c_[2.5 * z1, 1.2 * z2]
    noise = rng.normal(scale=0.15, size=(n, p - 2))
    # mix signal into a couple of noise cols so it's not trivially column 0/1
    X = np.zeros((n, p))
    X[:, 0] = signal[:, 0] + 0.2 * noise[:, 0]
    X[:, 1] = 0.3 * signal[:, 0] + signal[:, 1]
    X[:, 2:] = noise
    return X
