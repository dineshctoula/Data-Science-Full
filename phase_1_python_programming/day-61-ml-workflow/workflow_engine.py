"""Day 61 — the boring ML loop: train, check a held-out slice, pick a model.

Three contenders on the same split so the comparison isn't "I used a different
seed for the one I liked." Majority class is the floor. Nearest centroid and
a small logistic fit have to beat that on the validation rows, not on the
rows they just trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_signup_data(n: int = 240, seed: int = 61):
    """Two real features plus one noise column. Labels are 0/1."""
    rng = np.random.default_rng(seed)
    visits = rng.normal(3.0, 1.5, size=n)
    spend = rng.normal(15.0, 6.0, size=n)
    noise = rng.normal(0.0, 1.0, size=n)
    # intercept chosen so the class split isn't 90/10 — majority would
    # look brilliant and hide whether the other models learned anything
    logits = -1.6 + 0.7 * visits + 0.03 * spend
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(int)
    X = np.column_stack([visits, spend, noise])
    names = ("visits", "spend", "noise")
    return X, y, names


def train_val_split(X, y, val_frac: float = 0.3, seed: int = 61):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if not 0 < val_frac < 1:
        raise ValueError("val_frac should be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(round(len(X) * val_frac)))
    val, train = idx[:n_val], idx[n_val:]
    return X[train], X[val], y[train], y[val]


def accuracy(y, pred) -> float:
    y = np.asarray(y).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if len(y) == 0:
        raise ValueError("empty labels")
    return float(np.mean(y == pred))


class MajorityClass:
    """Predict whatever label showed up most often in training."""

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1)
        vals, counts = np.unique(y, return_counts=True)
        self.label_ = vals[np.argmax(counts)]
        return self

    def predict(self, X):
        X = np.asarray(X)
        n = len(X) if X.ndim > 1 else 1
        return np.full(n, self.label_)


class NearestCentroid:
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        self.centers_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        d2 = ((X[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(d2, axis=1)]


def _sigmoid(z):
    z = np.clip(z, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticGD:
    """A few gradient steps. Not tuned to death — just a third option."""

    def __init__(self, lr: float = 0.15, n_steps: int = 250):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if n_steps < 1:
            raise ValueError("n_steps should be >= 1")
        self.lr = float(lr)
        self.n_steps = int(n_steps)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, p = X.shape
        # spend is tens, visits are single digits — raw GD chases the big column
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-8] = 1.0
        self.scale_ = scale
        Xs = (X - self.mean_) / self.scale_
        w = np.zeros(p)
        b = 0.0
        for _ in range(self.n_steps):
            prob = _sigmoid(Xs @ w + b)
            err = prob - y
            w -= self.lr * (Xs.T @ err) / n
            b -= self.lr * float(err.mean())
        self.w_ = w
        self.b_ = float(b)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xs = (X - self.mean_) / self.scale_
        prob = _sigmoid(Xs @ self.w_ + self.b_)
        return (prob >= 0.5).astype(int)


@dataclass
class ModelScore:
    name: str
    train_acc: float
    val_acc: float

    def summary(self) -> str:
        return f"{self.name:<18} train={self.train_acc:.3f}  val={self.val_acc:.3f}"


def compare_models(X, y, val_frac: float = 0.3, seed: int = 61) -> list[ModelScore]:
    """Fit each model on the train slice. The val number is the one we trust."""
    Xtr, Xva, ytr, yva = train_val_split(X, y, val_frac=val_frac, seed=seed)
    specs = [
        ("majority", MajorityClass()),
        ("nearest centroid", NearestCentroid()),
        ("logistic", LogisticGD()),
    ]
    scores = []
    for name, model in specs:
        model.fit(Xtr, ytr)
        scores.append(
            ModelScore(
                name=name,
                train_acc=accuracy(ytr, model.predict(Xtr)),
                val_acc=accuracy(yva, model.predict(Xva)),
            )
        )
    return scores


def pick_by_validation(scores: list[ModelScore]) -> ModelScore:
    if not scores:
        raise ValueError("no scores")
    # ties: keep the earlier model. majority is first, so a tie doesn't
    # pretend a fancier model won.
    return max(scores, key=lambda s: s.val_acc)
