"""Day 66 — a rare class makes accuracy look fine.

If 94% of the rows are the boring class, "always say boring" is already
about 94% accurate and it never catches the rows you actually care about.
Weighting the rare class, or copying it until the counts match, trades
some of that accuracy for recall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_rare_events(n: int = 800, rate: float = 0.06, gap: float = 0.9, noise: float = 1.15, seed: int = 66):
    """Class 1 is uncommon and only a little shifted on the first column.

    The second column is noise. A tight gap means a plain fit would rather
    stay quiet than risk a pile of false alarms.
    """
    if not 0 < rate < 0.5:
        raise ValueError("rate should be a minority fraction")
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    signal = np.where(y == 1, rng.normal(gap, noise, size=n), rng.normal(0.0, noise, size=n))
    junk = rng.normal(size=n)
    return np.column_stack([signal, junk]), y


def split_indices(n: int, test_frac: float = 0.3, seed: int = 66):
    if not 0 < test_frac < 1:
        raise ValueError("test_frac should be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    if n_test >= n:
        raise ValueError("test slice ate the whole table")
    return idx[n_test:], idx[:n_test]


def oversample_minority(X, y, seed: int = 0):
    """Copy the smaller class until both sides have the same number of rows.

    This is a training trick. Don't run it on the test rows or the score
    is no longer the real mix.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0 or len(pos) == len(neg):
        return X.copy(), y.copy()
    rng = np.random.default_rng(seed)
    if len(pos) < len(neg):
        extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
    else:
        extra = rng.choice(neg, size=len(pos) - len(neg), replace=True)
    keep = np.concatenate([np.arange(len(y)), extra])
    rng.shuffle(keep)
    return X[keep], y[keep]


@dataclass
class ClfScore:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int

    def summary(self) -> str:
        return (
            f"acc={self.accuracy:.3f}  prec={self.precision:.3f}  "
            f"rec={self.recall:.3f}  f1={self.f1:.3f}  "
            f"tp={self.tp} fp={self.fp} fn={self.fn}"
        )


def score_predictions(y_true, y_pred) -> ClfScore:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    if len(y_true) == 0:
        raise ValueError("empty predictions")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    accuracy = (tp + tn) / len(y_true)
    # no positive calls -> precision is empty; record 0 so the table still prints
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ClfScore(accuracy, precision, recall, f1, tp, fp, fn, tn)


class WeightedLogit:
    """Logistic regression. pos_weight multiplies the gradient of class 1."""

    def __init__(self, lr: float = 0.35, epochs: int = 700, pos_weight: float = 1.0):
        if pos_weight <= 0:
            raise ValueError("pos_weight should be positive")
        if epochs < 1:
            raise ValueError("need at least one epoch")
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.pos_weight = float(pos_weight)
        self.mu = None
        self.sd = None
        self.coef = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        labels = set(np.unique(y).tolist())
        if not labels <= {0.0, 1.0}:
            raise ValueError("y should be 0/1")
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd == 0, 1.0, sd)
        design = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        coef = np.zeros(design.shape[1])
        sample_w = np.where(y == 1.0, self.pos_weight, 1.0)
        if float(sample_w.sum()) == 0:
            raise ValueError("no rows")
        for _ in range(self.epochs):
            linear = np.clip(design @ coef, -25, 25)
            prob = 1.0 / (1.0 + np.exp(-linear))
            grad = design.T @ (sample_w * (prob - y)) / sample_w.sum()
            coef = coef - self.lr * grad
        self.coef = coef
        return self

    def predict_proba(self, X):
        if self.coef is None:
            raise RuntimeError("fit first")
        X = np.asarray(X, dtype=float)
        design = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        linear = np.clip(design @ self.coef, -25, 25)
        return 1.0 / (1.0 + np.exp(-linear))

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def holdout_compare(X, y, seed: int = 66, pos_weight: float = 12.0, test_frac: float = 0.3):
    """Four scores on one split: majority, plain logit, weighted, oversampled."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1).astype(int)
    train, test = split_indices(len(X), test_frac=test_frac, seed=seed)
    y_te = y[test]
    X_te = X[test]

    majority = score_predictions(y_te, np.zeros(len(test), dtype=int))

    plain_model = WeightedLogit(pos_weight=1.0).fit(X[train], y[train])
    plain = score_predictions(y_te, plain_model.predict(X_te))

    weighted_model = WeightedLogit(pos_weight=pos_weight).fit(X[train], y[train])
    weighted = score_predictions(y_te, weighted_model.predict(X_te))

    X_bal, y_bal = oversample_minority(X[train], y[train], seed=seed)
    bal_model = WeightedLogit(pos_weight=1.0).fit(X_bal, y_bal)
    balanced = score_predictions(y_te, bal_model.predict(X_te))

    return {
        "majority": majority,
        "plain": plain,
        "weighted": weighted,
        "oversample": balanced,
        "train_pos": int(y[train].sum()),
        "test_pos": int(y_te.sum()),
        "balanced_rows": int(len(y_bal)),
    }
