"""Day 64 — missing values.

A blank is not a zero. Filling it with the column mean is fine when the
blanks are random. It throws away information when the blank itself means
something — here, people with lower scores left "hours" empty more often.

The fill value comes from the training rows. Using the whole column, including
the holdout, peeks at data you are not supposed to have yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_study_table(n: int = 220, seed: int = 64):
    """hours, sleep -> score. Then punch holes in hours, more often for low scores."""
    rng = np.random.default_rng(seed)
    hours = rng.normal(5.0, 1.5, size=n)
    sleep = rng.normal(7.0, 0.8, size=n)
    score = 30 + 6.0 * hours + 1.2 * sleep + rng.normal(0, 4.0, size=n)

    # probability of a blank rises as the score falls. That's the signal
    # a plain mean-fill will smear over.
    z = (score - score.mean()) / score.std()
    p_missing = 1.0 / (1.0 + np.exp(1.2 * z))
    missing = rng.random(n) < p_missing
    hours_obs = hours.astype(float).copy()
    hours_obs[missing] = np.nan

    X = np.column_stack([hours_obs, sleep])
    names = ("hours", "sleep")
    return X, score, names


def missing_report(X, names):
    X = np.asarray(X, dtype=float)
    rows = []
    for j, name in enumerate(names):
        col = X[:, j]
        n_miss = int(np.isnan(col).sum())
        rows.append(
            {
                "name": name,
                "n_missing": n_miss,
                "rate": n_miss / len(col),
            }
        )
    return rows


class SimpleImputer:
    """Fill NaN with the train mean or median. Optional 0/1 column for 'was blank'."""

    def __init__(self, strategy: str = "mean", add_indicator: bool = False):
        if strategy not in {"mean", "median"}:
            raise ValueError("strategy must be mean or median")
        self.strategy = strategy
        self.add_indicator = bool(add_indicator)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X needs to be a non-empty 2d matrix")
        fills = []
        for j in range(X.shape[1]):
            col = X[:, j]
            observed = col[~np.isnan(col)]
            if len(observed) == 0:
                raise ValueError(f"column {j} is entirely missing")
            if self.strategy == "mean":
                fills.append(float(observed.mean()))
            else:
                fills.append(float(np.median(observed)))
        self.fill_ = np.array(fills, dtype=float)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "fill_"):
            raise RuntimeError("call fit() before transform()")
        X = np.asarray(X, dtype=float).copy()
        if X.shape[1] != self.n_features_:
            raise ValueError("wrong number of columns")
        indicator = np.isnan(X).astype(float)
        for j in range(X.shape[1]):
            blank = np.isnan(X[:, j])
            X[blank, j] = self.fill_[j]
        if self.add_indicator:
            # only keep indicator columns that were sometimes missing in spirit —
            # still append one per feature so the shape is predictable
            return np.column_stack([X, indicator])
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def drop_incomplete(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    keep = ~np.isnan(X).any(axis=1)
    return X[keep], y[keep]


def _ols_predict(Xtr, ytr, Xte):
    design = np.c_[np.ones(len(Xtr)), Xtr]
    beta = np.linalg.lstsq(design, ytr, rcond=None)[0]
    pred = np.c_[np.ones(len(Xte)), Xte] @ beta
    return pred


def _mse(y, pred):
    return float(np.mean((np.asarray(y) - np.asarray(pred)) ** 2))


def train_val_split(X, y, val_frac: float = 0.25, seed: int = 64):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(round(len(X) * val_frac)))
    val, train = idx[:n_val], idx[n_val:]
    return X[train], X[val], y[train], y[val]


@dataclass
class ImputeResult:
    name: str
    n_train: int
    mse: float

    def summary(self) -> str:
        return f"{self.name:<22} n_train={self.n_train:<4} MSE={self.mse:.2f}"


def compare_impute(X, y, seed: int = 64) -> list[ImputeResult]:
    """Drop vs mean vs median vs mean+indicator. Imputers fit on train only."""
    Xtr, Xva, ytr, yva = train_val_split(X, y, seed=seed)
    out = []

    Xtr_d, ytr_d = drop_incomplete(Xtr, ytr)
    Xva_d, yva_d = drop_incomplete(Xva, yva)
    pred = _ols_predict(Xtr_d, ytr_d, Xva_d)
    out.append(ImputeResult("drop rows", len(ytr_d), _mse(yva_d, pred)))

    for strategy in ("mean", "median"):
        imp = SimpleImputer(strategy=strategy).fit(Xtr)
        pred = _ols_predict(imp.transform(Xtr), ytr, imp.transform(Xva))
        out.append(ImputeResult(f"{strategy} fill", len(ytr), _mse(yva, pred)))

    ind = SimpleImputer(strategy="mean", add_indicator=True).fit(Xtr)
    pred = _ols_predict(ind.transform(Xtr), ytr, ind.transform(Xva))
    out.append(ImputeResult("mean + indicator", len(ytr), _mse(yva, pred)))
    return out
