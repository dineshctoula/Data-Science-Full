"""Day 80 — phase 4 wrap-up.

One churn table run through the habits from Days 61–79, in order:
split three ways before touching anything, fill and encode and scale
with numbers taken from the training rows only, train a logistic model
with mini-batch Adam, weight decay, and early stopping, pick the decay
and the threshold on validation, and open the test slice once at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def log_loss(y, p) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.clip(np.asarray(p, dtype=float).reshape(-1), 1e-12, 1 - 1e-12)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("y and p should be the same non-empty length")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def scores_at(y, p, threshold: float = 0.5):
    """Accuracy, precision, recall, and F1 for one cut-off."""
    y = np.asarray(y).reshape(-1).astype(int)
    pred = (np.asarray(p, dtype=float).reshape(-1) >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(np.mean(pred == y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def roc_auc(y, p) -> float:
    """Chance that a random leaver scores above a random stayer. Ties count half."""
    y = np.asarray(y).reshape(-1).astype(int)
    p = np.asarray(p, dtype=float).reshape(-1)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need both classes for AUC")
    diff = pos[:, None] - neg[None, :]
    return float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))


# ---------------------------------------------------------------------------
# the model: logistic + mini-batch Adam + weight decay + early stopping
# ---------------------------------------------------------------------------


@dataclass
class FitHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    best_epoch: int = 0


class ChurnLogistic:
    """Logistic regression trained with the optimizer pieces from Days 72–78."""

    def __init__(
        self,
        decay: float = 0.0,
        lr: float = 0.005,
        batch: int = 32,
        max_epochs: int = 300,
        patience: int = 25,
        seed: int = 0,
    ):
        if decay < 0:
            raise ValueError("decay should be zero or more")
        if lr <= 0 or batch < 1 or max_epochs < 1 or patience < 1:
            raise ValueError("lr, batch, max_epochs, and patience should be positive")
        self.decay = decay
        self.lr = lr
        self.batch = batch
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.w = None
        self.b = 0.0
        self.history = None

    def fit(self, X, y, X_val, y_val):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        X_val = np.asarray(X_val, dtype=float)
        y_val = np.asarray(y_val, dtype=float).reshape(-1)
        if len(X) != len(y) or len(X_val) != len(y_val):
            raise ValueError("rows and labels differ in length")
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        # Adam keeps a running mean (m) and running square (v) per weight
        m_w, v_w = np.zeros(d), np.zeros(d)
        m_b, v_b = 0.0, 0.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        history = FitHistory()
        best = (np.inf, w.copy(), b)
        since_best = 0
        for epoch in range(1, self.max_epochs + 1):
            order = rng.permutation(n)
            for start in range(0, n, self.batch):
                rows = order[start : start + self.batch]
                err = sigmoid(X[rows] @ w + b) - y[rows]
                # the decay term pulls the slopes toward zero; the intercept is left free
                grad_w = X[rows].T @ err / len(rows) + self.decay * w
                grad_b = float(err.mean())

                step += 1
                m_w = beta1 * m_w + (1 - beta1) * grad_w
                v_w = beta2 * v_w + (1 - beta2) * grad_w**2
                m_b = beta1 * m_b + (1 - beta1) * grad_b
                v_b = beta2 * v_b + (1 - beta2) * grad_b**2
                # bias correction: early averages start at zero and read too small
                fix1, fix2 = 1 - beta1**step, 1 - beta2**step
                w = w - self.lr * (m_w / fix1) / (np.sqrt(v_w / fix2) + eps)
                b = b - self.lr * (m_b / fix1) / (np.sqrt(v_b / fix2) + eps)

            # the losses are reported without the penalty so decays compare fairly
            train_loss = log_loss(y, sigmoid(X @ w + b))
            val_loss = log_loss(y_val, sigmoid(X_val @ w + b))
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            if val_loss < best[0] - 1e-6:
                best = (val_loss, w.copy(), b)
                history.best_epoch = epoch
                since_best = 0
            else:
                since_best += 1
                if since_best >= self.patience:
                    break

        if not np.all(np.isfinite(best[1])):
            raise FloatingPointError("training blew up; try a smaller lr")
        # Day 72: keep the weights from the best validation epoch, not the last one
        _, self.w, self.b = best
        self.history = history
        return self

    def predict_proba(self, X):
        if self.w is None:
            raise RuntimeError("fit first")
        return sigmoid(np.asarray(X, dtype=float) @ self.w + self.b)
