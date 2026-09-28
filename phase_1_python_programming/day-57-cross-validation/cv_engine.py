"""Day 57 — k-fold cross-validation.

One holdout split on a small set is basically a coin flip. Chop the rows
into k chunks, train on the rest, score the held-out chunk, then rotate.
The average (and the spread) is a steadier read than a single lucky split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def kfold_indices(n: int, k: int = 5, seed: int = 57, shuffle: bool = True):
    """Yield (train_idx, test_idx) for each fold.

    Leftover rows when n isn't divisible by k get sprinkled across the
    first few folds so we don't drop anyone.
    """
    if k < 2:
        raise ValueError("k needs to be at least 2")
    if n < k:
        raise ValueError(f"n={n} is smaller than k={k}")

    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(idx)

    folds = np.array_split(idx, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


def stratified_kfold_indices(y, k: int = 5, seed: int = 57):
    """Same idea, but each class is split on its own so folds stay balanced."""
    y = np.asarray(y).reshape(-1)
    if k < 2:
        raise ValueError("k needs to be at least 2")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("need at least two classes to stratify")

    rng = np.random.default_rng(seed)
    # bucket of test indices per fold
    fold_tests = [[] for _ in range(k)]
    for lab in classes:
        members = np.flatnonzero(y == lab)
        if len(members) < k:
            raise ValueError(f"class {lab} has fewer rows than k={k}")
        members = rng.permutation(members)
        chunks = np.array_split(members, k)
        for i, chunk in enumerate(chunks):
            fold_tests[i].extend(chunk.tolist())

    n = len(y)
    for i in range(k):
        test = np.array(sorted(fold_tests[i]), dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[test] = False
        train = np.flatnonzero(mask)
        yield train, test


class NearestCentroid:
    """Dumb classifier: predict the class whose mean is closest.

    Just need something with fit/predict so the CV loop has a model to chew on.
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # squared distance to each centroid
        d2 = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(d2, axis=1)]


@dataclass
class CVResult:
    scores: np.ndarray
    mean: float
    std: float
    k: int
    stratified: bool

    def summary(self) -> str:
        kind = "stratified" if self.stratified else "plain"
        return (
            f"{kind} {self.k}-fold → mean={self.mean:.3f} "
            f"(std={self.std:.3f})"
        )


def cross_val_accuracy(X, y, k: int = 5, stratified: bool = True, seed: int = 57) -> CVResult:
    """Fit a fresh nearest-centroid on each train fold, score accuracy on test."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")
    if not np.isfinite(X).all():
        raise ValueError("X has non-finite values")

    splitter = stratified_kfold_indices if stratified else kfold_indices
    if stratified:
        folds = splitter(y, k=k, seed=seed)
    else:
        folds = splitter(len(X), k=k, seed=seed)

    scores = []
    for train, test in folds:
        model = NearestCentroid().fit(X[train], y[train])
        preds = model.predict(X[test])
        scores.append(float(np.mean(preds == y[test])))

    arr = np.asarray(scores, dtype=float)
    return CVResult(
        scores=arr,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        k=k,
        stratified=stratified,
    )


def single_split_accuracy(X, y, test_frac: float = 0.3, seed: int = 57) -> float:
    """One random holdout — the thing CV is supposed to be less jumpy than."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(1, int(round(len(X) * test_frac)))
    test, train = idx[:n_test], idx[n_test:]
    model = NearestCentroid().fit(X[train], y[train])
    preds = model.predict(X[test])
    return float(np.mean(preds == y[test]))


def repeated_holdout(X, y, n_repeats: int = 8, test_frac: float = 0.3, seed: int = 57) -> np.ndarray:
    """A handful of different holdouts so we can see how much one split wobbles."""
    return np.array(
        [single_split_accuracy(X, y, test_frac=test_frac, seed=seed + i) for i in range(n_repeats)]
    )


def make_two_class_cloud(n_per: int = 40, sep: float = 1.6, seed: int = 57):
    rng = np.random.default_rng(seed)
    a = rng.normal([0.0, 0.0], 0.7, size=(n_per, 2))
    b = rng.normal([sep, 0.4], 0.7, size=(n_per, 2))
    X = np.vstack([a, b])
    y = np.array([0] * n_per + [1] * n_per)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]
