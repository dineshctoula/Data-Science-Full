"""Day 63 — categorical encoding.

A model wants numbers. "small / medium / large" has an order, so an integer
code is fair. "email / ads / referral" does not, so one-hot columns are safer
than pretending referral = 2 means "twice email."

Fit the categories on the training rows. A label that only shows up later
should not grow a new column after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_str(values):
    values = np.asarray(values).reshape(-1)
    if len(values) == 0:
        raise ValueError("need at least one value")
    return np.array([str(v) for v in values])


class OrdinalEncoder:
    """Map labels to 0, 1, 2, ... Unseen labels become -1."""

    def __init__(self, order=None):
        # pass order when you know it (small < medium < large).
        # leave it None and fit() will sort the strings it sees.
        self.order = None if order is None else tuple(str(x) for x in order)

    def fit(self, values):
        seen = _as_str(values)
        if self.order is None:
            cats = tuple(sorted(set(seen.tolist())))
        else:
            # keep the order you asked for, but only categories that appear
            present = set(seen.tolist())
            cats = tuple(c for c in self.order if c in present)
            if not cats:
                raise ValueError("none of the ordered categories showed up")
        self.categories_ = cats
        self.index_ = {c: i for i, c in enumerate(cats)}
        return self

    def transform(self, values):
        if not hasattr(self, "index_"):
            raise RuntimeError("call fit() before transform()")
        seen = _as_str(values)
        return np.array([self.index_.get(v, -1) for v in seen], dtype=float)

    def fit_transform(self, values):
        return self.fit(values).transform(values)


class OneHotEncoder:
    """One column per training category. Unknown rows stay all zeros."""

    def fit(self, values):
        seen = _as_str(values)
        self.categories_ = tuple(sorted(set(seen.tolist())))
        self.index_ = {c: i for i, c in enumerate(self.categories_)}
        return self

    def transform(self, values):
        if not hasattr(self, "categories_"):
            raise RuntimeError("call fit() before transform()")
        seen = _as_str(values)
        out = np.zeros((len(seen), len(self.categories_)), dtype=float)
        for i, v in enumerate(seen):
            j = self.index_.get(v)
            if j is not None:
                out[i, j] = 1.0
        return out

    def fit_transform(self, values):
        return self.fit(values).transform(values)


def make_plan_data(n: int = 180, seed: int = 63):
    """Plan is ordered. Channel is not. Score depends on both, plus hours."""
    rng = np.random.default_rng(seed)
    plans = np.array(["basic", "plus", "pro"])
    channels = np.array(["email", "ads", "referral"])
    plan = rng.choice(plans, size=n, p=[0.5, 0.3, 0.2])
    channel = rng.choice(channels, size=n)
    hours = rng.normal(4.0, 1.2, size=n)
    plan_bonus = {"basic": 0.0, "plus": 8.0, "pro": 18.0}
    channel_bonus = {"email": 0.0, "ads": 3.0, "referral": 11.0}
    score = (
        40
        + 3.0 * hours
        + np.array([plan_bonus[p] for p in plan])
        + np.array([channel_bonus[c] for c in channel])
        + rng.normal(0, 4.0, size=n)
    )
    return plan, channel, hours, score


def train_val_split(plan, channel, hours, score, val_frac: float = 0.25, seed: int = 63):
    n = len(score)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val, train = idx[:n_val], idx[n_val:]

    def take(i):
        return plan[i], channel[i], hours[i], score[i]

    return take(train), take(val)


@dataclass
class FitResult:
    name: str
    mse: float
    r2: float

    def summary(self) -> str:
        return f"{self.name:<22} MSE={self.mse:.2f}  R²={self.r2:.3f}"


def _ols(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    design = np.c_[np.ones(len(X)), X]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return beta


def _predict(beta, X):
    X = np.asarray(X, dtype=float)
    design = np.c_[np.ones(len(X)), X]
    return design @ beta


def _mse(y, pred):
    return float(np.mean((y - pred) ** 2))


def _r2(y, pred):
    y = np.asarray(y, dtype=float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot


def compare_encodings(plan, channel, hours, score, seed: int = 63) -> list[FitResult]:
    """Three designs, same split.

    Flipping basic/plus/pro to pro/plus/basic does not change a linear fit:
    the new codes are just 2 minus the old ones, so the slope changes sign
    and the predictions stay put. One-hot of plan and channel is the design
    that can use the channel, which has no order.
    """
    (p_tr, c_tr, h_tr, y_tr), (p_va, c_va, h_va, y_va) = train_val_split(
        plan, channel, hours, score, seed=seed
    )
    results = []

    right = OrdinalEncoder(order=("basic", "plus", "pro"))
    Xtr = np.column_stack([h_tr, right.fit_transform(p_tr)])
    Xva = np.column_stack([h_va, right.transform(p_va)])
    beta = _ols(Xtr, y_tr)
    pred = _predict(beta, Xva)
    results.append(FitResult("ordinal plan", _mse(y_va, pred), _r2(y_va, pred)))

    wrong = OrdinalEncoder(order=("pro", "plus", "basic"))
    Xtr = np.column_stack([h_tr, wrong.fit_transform(p_tr)])
    Xva = np.column_stack([h_va, wrong.transform(p_va)])
    beta = _ols(Xtr, y_tr)
    pred = _predict(beta, Xva)
    results.append(FitResult("flipped plan order", _mse(y_va, pred), _r2(y_va, pred)))

    plan_oh = OneHotEncoder().fit(p_tr)
    chan_oh = OneHotEncoder().fit(c_tr)
    Xtr = np.column_stack([h_tr, plan_oh.transform(p_tr), chan_oh.transform(c_tr)])
    Xva = np.column_stack([h_va, plan_oh.transform(p_va), chan_oh.transform(c_va)])
    beta = _ols(Xtr, y_tr)
    pred = _predict(beta, Xva)
    results.append(FitResult("one-hot plan+channel", _mse(y_va, pred), _r2(y_va, pred)))
    return results
