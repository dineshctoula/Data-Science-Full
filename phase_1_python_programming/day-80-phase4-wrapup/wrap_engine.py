"""Day 80 — phase 4 wrap-up.

One churn table run through the habits from Days 61–79, in order:
split three ways before touching anything, fill and encode and scale
with numbers taken from the training rows only, train a logistic model
with mini-batch Adam, weight decay, and early stopping, pick the decay
and the threshold on validation, and open the test slice once at the end.
"""

from __future__ import annotations

import numpy as np

NUMERIC = ("tenure", "spend", "calls")
CATEGORICAL = ("plan", "region")
PLANS = ("basic", "plus", "pro")
REGIONS = ("north", "south", "east", "west")


def sigmoid(z):
    # clip so exp() can't overflow on a wild logit
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# ---------------------------------------------------------------------------
# data and the three-way split
# ---------------------------------------------------------------------------


def make_churn_table(n: int = 900, seed: int = 80, missing_rate: float = 0.10):
    """Customers with tenure, spend, support calls, plan, and region.

    Churn rises with support calls and falls with tenure. The plan matters,
    the region does not. Spend goes missing more often for customers who
    leave, so the missing flag carries a little signal of its own.
    """
    if n < 10:
        raise ValueError("need at least 10 rows")
    if not 0.0 <= missing_rate < 0.5:
        raise ValueError("missing_rate should be in [0, 0.5)")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 61, size=n).astype(float)
    spend = np.clip(rng.normal(55.0, 18.0, size=n), 10.0, None)
    calls = rng.poisson(1.5, size=n).astype(float)
    plan = rng.choice(PLANS, size=n, p=[0.5, 0.3, 0.2])
    region = rng.choice(REGIONS, size=n)

    plan_effect = np.select([plan == "basic", plan == "pro"], [0.5, -0.6], default=0.0)
    logit = -1.3 - 0.045 * (tenure - 30.0) + 0.55 * (calls - 1.5) + 0.012 * (spend - 55.0) + plan_effect
    y = (rng.random(n) < sigmoid(logit)).astype(int)

    # leavers skip the spend question twice as often as the base rate,
    # stayers half as often
    miss_p = np.where(y == 1, 2.0 * missing_rate, 0.5 * missing_rate)
    spend[rng.random(n) < miss_p] = np.nan

    table = {"tenure": tenure, "spend": spend, "calls": calls, "plan": plan, "region": region}
    return table, y


def take_rows(table, idx):
    return {name: col[idx] for name, col in table.items()}


def split_three(table, y, fracs=(0.6, 0.2, 0.2), seed: int = 80):
    """Shuffle once, then cut into train / validation / test.

    This happens before any filling or scaling, so the held-out rows never
    leak into a median or a standard deviation (Day 65).
    """
    if len(fracs) != 3 or min(fracs) <= 0 or abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError("fracs should be three positive numbers that sum to 1")
    y = np.asarray(y).reshape(-1)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * fracs[0]))
    n_val = int(round(n * fracs[1]))
    if n_train < 1 or n_val < 1 or n - n_train - n_val < 1:
        raise ValueError("a slice came out empty; use more rows")
    cuts = (idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :])
    return [(take_rows(table, part), y[part]) for part in cuts]


# ---------------------------------------------------------------------------
# preprocessing fit on the training rows only
# ---------------------------------------------------------------------------


class Prep:
    """Median fill + missing flag, one-hot, then standard scaling.

    Every number it uses (medians, category levels, means, stds) comes from
    the table passed to fit(). transform() only reuses them.
    """

    def __init__(self):
        self.medians = None
        self.levels = None
        self.mean = None
        self.std = None
        self.names = None

    def fit(self, table):
        self.medians = {}
        for col in NUMERIC:
            values = np.asarray(table[col], dtype=float)
            if np.all(np.isnan(values)):
                raise ValueError(f"{col} is empty in the training rows")
            self.medians[col] = float(np.nanmedian(values))
        # sorted so the column order doesn't depend on which label showed up first
        self.levels = {col: sorted(set(table[col].tolist())) for col in CATEGORICAL}
        raw = self._unscaled(table)
        self.mean = raw.mean(axis=0)
        self.std = raw.std(axis=0)
        # a constant column would divide by zero; leave it unscaled instead
        self.std[self.std == 0] = 1.0
        return self

    def _unscaled(self, table):
        cols, names = [], []
        for col in NUMERIC:
            values = np.asarray(table[col], dtype=float)
            missing = np.isnan(values)
            cols.append(np.where(missing, self.medians[col], values))
            names.append(col)
            if col == "spend":
                # Day 64: the fact that it was blank is worth keeping
                cols.append(missing.astype(float))
                names.append("spend_missing")
        for col in CATEGORICAL:
            labels = np.asarray(table[col])
            for level in self.levels[col]:
                # a label never seen in training gets all zeros (Day 63)
                cols.append((labels == level).astype(float))
                names.append(f"{col}={level}")
        self.names = tuple(names)
        return np.column_stack(cols)

    def transform(self, table):
        if self.mean is None:
            raise RuntimeError("fit first")
        return (self._unscaled(table) - self.mean) / self.std

    def fit_transform(self, table):
        return self.fit(table).transform(table)
