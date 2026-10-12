"""Day 72 — stop when the validation loss turns around.

The training loss on a wide logistic fit keeps dropping. Past a point it
is fitting noise columns, and the validation loss climbs. The weights
from the best validation epoch are the ones to keep. The last epoch is
there so you can see the damage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_noisy_classes(n: int = 220, n_noise: int = 20, sep: float = 1.8, seed: int = 72):
    """Two real columns, then a pile of weaker noise columns.

    The label comes from the real columns only. A long fit can still
    memorize the noise ones.
    """
    if n < 40:
        raise ValueError("need at least 40 rows")
    if n_noise < 1:
        raise ValueError("need at least one noise column")
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n, 2))
    junk = 0.8 * rng.normal(size=(n, n_noise))
    X = np.column_stack([signal, junk])
    logits = sep * signal[:, 0] + 0.9 * signal[:, 1]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(int)
    return X, y


def three_way(n: int, n_train: int = 70, n_val: int = 60, seed: int = 72):
    if n_train < 5 or n_val < 5 or n_train + n_val >= n:
        raise ValueError("split should leave a test slice")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def log_loss(y, probs) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    probs = np.clip(np.asarray(probs, dtype=float).reshape(-1), 1e-8, 1.0 - 1e-8)
    if len(y) != len(probs) or len(y) == 0:
        raise ValueError("y and probs should be the same non-empty length")
    return float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))


def accuracy(y, probs, threshold: float = 0.5) -> float:
    y = np.asarray(y).reshape(-1).astype(int)
    pred = (np.asarray(probs).reshape(-1) >= threshold).astype(int)
    return float(np.mean(pred == y))


def _sigmoid(linear):
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -25, 25)))


@dataclass
class StopResult:
    train_loss: np.ndarray
    val_loss: np.ndarray
    best_epoch: int  # 1-based, matches the printed step
    best_val: float
    test_loss_best: float
    test_loss_final: float
    test_acc_best: float
    test_acc_final: float

    @property
    def epochs(self) -> int:
        return int(len(self.train_loss))


def fit_until(X, y, n_train: int = 70, n_val: int = 60, lr: float = 0.25, epochs: int = 200, seed: int = 72) -> StopResult:
    """Full-batch logistic. Scale from the training rows. Watch validation each step."""
    if epochs < 1:
        raise ValueError("need at least one epoch")
    if lr <= 0:
        raise ValueError("learning rate should be positive")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(X) != len(y):
        raise ValueError("X and y differ in length")

    train, val, test = three_way(len(X), n_train=n_train, n_val=n_val, seed=seed)
    mu = X[train].mean(axis=0)
    sd = X[train].std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)

    def design(idx):
        return np.c_[np.ones(len(idx)), (X[idx] - mu) / sd]

    D_train, D_val, D_test = design(train), design(val), design(test)
    y_train, y_val, y_test = y[train], y[val], y[test]
    coef = np.zeros(D_train.shape[1])
    best_coef = coef.copy()
    best_val = np.inf
    best_epoch = 0
    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)

    for step in range(epochs):
        probs = _sigmoid(D_train @ coef)
        grad = D_train.T @ (probs - y_train) / len(y_train)
        coef = coef - lr * grad
        train_loss[step] = log_loss(y_train, _sigmoid(D_train @ coef))
        val_now = log_loss(y_val, _sigmoid(D_val @ coef))
        val_loss[step] = val_now
        # first time it bottoms out wins a tie, so a flat tail does not move the pick
        if val_now < best_val:
            best_val = val_now
            best_epoch = step + 1
            best_coef = coef.copy()

    test_best = _sigmoid(D_test @ best_coef)
    test_final = _sigmoid(D_test @ coef)
    return StopResult(
        train_loss=train_loss,
        val_loss=val_loss,
        best_epoch=best_epoch,
        best_val=float(best_val),
        test_loss_best=log_loss(y_test, test_best),
        test_loss_final=log_loss(y_test, test_final),
        test_acc_best=accuracy(y_test, test_best),
        test_acc_final=accuracy(y_test, test_final),
    )
