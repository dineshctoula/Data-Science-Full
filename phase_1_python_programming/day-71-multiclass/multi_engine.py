"""Day 71 — three classes, one logistic each.

One-vs-rest fits a yes/no model per class. At prediction time the row
goes to whichever model is most sure. A tie keeps the smaller class index.
The classes overlap, so the confusion matrix matters more than the headline
accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_blobs(n_per_class: int = 90, sep: float = 1.4, noise: float = 1.1, seed: int = 71):
    """Three blobs on a triangle. Closer blobs trade more mistakes."""
    if n_per_class < 10:
        raise ValueError("need at least 10 rows per class")
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [sep, 0.0],
            [-sep / 2, sep * 0.85],
            [-sep / 2, -sep * 0.85],
        ]
    )
    parts = []
    labels = []
    for class_id, center in enumerate(centers):
        parts.append(center + noise * rng.normal(size=(n_per_class, 2)))
        labels.append(np.full(n_per_class, class_id, dtype=int))
    return np.vstack(parts), np.concatenate(labels)


def split_indices(n: int, test_frac: float = 0.3, seed: int = 71):
    if not 0 < test_frac < 1:
        raise ValueError("test_frac should be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    if n_test >= n:
        raise ValueError("test slice ate the whole table")
    return idx[n_test:], idx[:n_test]


def _fit_binary(X, y, lr: float = 0.4, epochs: int = 400):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    design = np.c_[np.ones(len(X)), (X - mu) / sd]
    coef = np.zeros(design.shape[1])
    y = np.asarray(y, dtype=float)
    for _ in range(epochs):
        linear = np.clip(design @ coef, -20, 20)
        prob = 1.0 / (1.0 + np.exp(-linear))
        coef = coef - lr * (design.T @ (prob - y) / len(y))
    return mu, sd, coef


def _positive_proba(mu, sd, coef, X):
    design = np.c_[np.ones(len(X)), (X - mu) / sd]
    linear = np.clip(design @ coef, -20, 20)
    return 1.0 / (1.0 + np.exp(-linear))


@dataclass
class MultiResult:
    accuracy: float
    majority_accuracy: float
    confusion: np.ndarray
    n_classes: int

    def recall(self, class_id: int) -> float:
        row = self.confusion[class_id]
        total = int(row.sum())
        if total == 0:
            return 0.0
        return float(row[class_id] / total)


class OneVsRest:
    def __init__(self, lr: float = 0.4, epochs: int = 400):
        self.lr = lr
        self.epochs = epochs
        self.models = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1).astype(int)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes")
        self.models = []
        for class_id in self.classes_:
            binary = (y == class_id).astype(float)
            self.models.append(_fit_binary(X, binary, lr=self.lr, epochs=self.epochs))
        return self

    def predict_proba(self, X):
        if not self.models:
            raise RuntimeError("fit first")
        X = np.asarray(X, dtype=float)
        columns = [_positive_proba(mu, sd, coef, X) for mu, sd, coef in self.models]
        return np.column_stack(columns)

    def predict(self, X):
        # argmax breaks a tie toward the earlier class
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def confusion_matrix(y_true, y_pred, n_classes: int):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred should be the same non-empty length")
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for actual, guessed in zip(y_true, y_pred):
        if actual < 0 or guessed < 0 or actual >= n_classes or guessed >= n_classes:
            raise ValueError("label outside 0..n_classes-1")
        matrix[actual, guessed] += 1
    return matrix


def evaluate(X, y, seed: int = 71, test_frac: float = 0.3) -> MultiResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1).astype(int)
    train, test = split_indices(len(X), test_frac=test_frac, seed=seed)
    model = OneVsRest().fit(X[train], y[train])
    pred = model.predict(X[test])
    n_classes = int(y.max()) + 1
    matrix = confusion_matrix(y[test], pred, n_classes)
    accuracy = float(np.mean(pred == y[test]))
    majority = int(np.bincount(y[train]).argmax())
    majority_accuracy = float(np.mean(y[test] == majority))
    return MultiResult(accuracy, majority_accuracy, matrix, n_classes)
