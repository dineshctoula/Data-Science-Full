"""Day 79 — dropout on a small network that can memorize.

16 noisy points of a sine curve and a 64-unit tanh layer. With every
unit on all the time, the network bends through the noise. Dropout
switches off a random slice of the hidden units on each training step,
so no single unit can carry a wiggle on its own.

Inverted dropout: the surviving units are scaled up by 1/keep during
training, so prediction just runs the full network with nothing to undo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_curve(n_train: int = 16, n_val: int = 40, n_test: int = 300, noise: float = 0.35, seed: int = 79):
    """Three separate draws of sin(1.6 x) plus noise, x in [-2, 2]."""
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("every slice needs at least one row")
    rng = np.random.default_rng(seed)

    def draw(n):
        x = rng.uniform(-2.0, 2.0, size=n)
        return x, np.sin(1.6 * x) + noise * rng.normal(size=n)

    return draw(n_train), draw(n_val), draw(n_test)


class TinyNet:
    """x -> tanh(64 units) -> y. Full-batch gradient descent on MSE."""

    def __init__(self, hidden: int = 64, drop: float = 0.0, lr: float = 0.04, epochs: int = 8000, seed: int = 0):
        if not 0.0 <= drop < 1.0:
            raise ValueError("drop should be in [0, 1)")
        if hidden < 1:
            raise ValueError("need at least one hidden unit")
        if lr <= 0 or epochs < 1:
            raise ValueError("lr should be positive and epochs at least 1")
        self.hidden = hidden
        self.drop = drop
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.params = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if len(x) != len(y):
            raise ValueError("x and y differ in length")
        rng = np.random.default_rng(self.seed)
        W1 = rng.normal(0.0, 1.0, size=(1, self.hidden))
        b1 = rng.normal(0.0, 1.0, size=self.hidden)
        W2 = rng.normal(0.0, 1.0 / np.sqrt(self.hidden), size=(self.hidden, 1))
        b2 = np.zeros(1)
        keep = 1.0 - self.drop

        for _ in range(self.epochs):
            h = np.tanh(x @ W1 + b1)
            if self.drop > 0:
                # a fresh mask every step; survivors get scaled by 1/keep
                mask = (rng.random(h.shape) < keep) / keep
            else:
                mask = np.ones_like(h)
            h_used = h * mask
            out = h_used @ W2 + b2

            d_out = 2.0 * (out - y) / len(x)
            grad_W2 = h_used.T @ d_out
            grad_b2 = d_out.sum(axis=0)
            # a dropped unit gets no gradient on this step
            d_h = (d_out @ W2.T) * mask * (1.0 - h * h)
            grad_W1 = x.T @ d_h
            grad_b1 = d_h.sum(axis=0)

            W1 -= self.lr * grad_W1
            b1 -= self.lr * grad_b1
            W2 -= self.lr * grad_W2
            b2 -= self.lr * grad_b2

        if not all(np.all(np.isfinite(p)) for p in (W1, b1, W2, b2)):
            raise FloatingPointError("training blew up; try a smaller lr")
        self.params = (W1, b1, W2, b2)
        return self

    def predict(self, x):
        if self.params is None:
            raise RuntimeError("fit first")
        W1, b1, W2, b2 = self.params
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        # no mask here: every unit is on when we predict
        return (np.tanh(x @ W1 + b1) @ W2 + b2).ravel()


def mse(y, pred) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if len(y) != len(pred) or len(y) == 0:
        raise ValueError("y and pred should be the same non-empty length")
    return float(np.mean((y - pred) ** 2))


@dataclass
class DropTrial:
    drop: float
    train_mse: float
    val_mse: float
    test_mse: float
    worst_test: float  # the unluckiest of the starting seeds


def sweep(data, drops, n_seeds: int = 5, **net_kwargs):
    """Average each dropout rate over several starting seeds.

    One seed can make a rate look good or bad by accident. The same seeds
    are reused for every rate so the comparison is like for like.
    """
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = data
    drops = [float(d) for d in drops]
    if not drops:
        raise ValueError("no dropout rates")
    if n_seeds < 1:
        raise ValueError("need at least one seed")
    trials = []
    for drop in drops:
        tr, va, te = [], [], []
        for seed in range(n_seeds):
            net = TinyNet(drop=drop, seed=seed, **net_kwargs).fit(x_tr, y_tr)
            tr.append(mse(y_tr, net.predict(x_tr)))
            va.append(mse(y_va, net.predict(x_va)))
            te.append(mse(y_te, net.predict(x_te)))
        trials.append(DropTrial(drop, float(np.mean(tr)), float(np.mean(va)), float(np.mean(te)), float(np.max(te))))
    return trials


def pick_by_val(trials) -> DropTrial:
    """Lowest mean validation MSE. A tie keeps the smaller rate."""
    if not trials:
        raise ValueError("no trials")
    return min(trials, key=lambda trial: (trial.val_mse, trial.drop))
