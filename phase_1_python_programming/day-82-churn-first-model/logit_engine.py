"""Day 82 — L2-penalized logistic regression fit by Newton's method.

Phase 4 trained logistic models with gradient descent and Adam, which
means picking a learning rate and an epoch count. With 23 columns the
Hessian is a 24 × 24 matrix, so Newton's method is cheap: each step
solves for the exact quadratic minimum, and it lands in under ten steps
with nothing to tune but the penalty.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class NewtonLogistic:
    def __init__(self, lam: float = 1e-3, max_iter: int = 50, tol: float = 1e-10):
        if lam < 0:
            raise ValueError("lam should be zero or more")
        if max_iter < 1:
            raise ValueError("max_iter should be at least 1")
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.intercept = None
        self.coef = None
        self.n_iter = 0

    def _objective(self, A, y, w, penalty):
        # mean log loss plus (lam / 2) * ||slopes||^2
        z = A @ w
        # log(1 + e^z) - y z, written with logaddexp so a large |z| can't overflow
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        return float(loss + 0.5 * np.sum(penalty * w * w))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X should be 2-D with one row per label")
        if not set(np.unique(y)) <= {0.0, 1.0}:
            raise ValueError("y should be 0/1")
        n, d = X.shape
        A = np.column_stack([np.ones(n), X])
        w = np.zeros(d + 1)
        # the intercept (index 0) is not penalized, same as Day 78
        penalty = np.full(d + 1, self.lam)
        penalty[0] = 0.0

        current = self._objective(A, y, w, penalty)
        for it in range(1, self.max_iter + 1):
            p = sigmoid(A @ w)
            grad = A.T @ (p - y) / n + penalty * w
            hess = (A * (p * (1 - p))[:, None]).T @ A / n + np.diag(penalty)
            step = np.linalg.solve(hess, grad)
            # halve the step until the objective goes down; a full Newton step
            # can overshoot when the starting point is far from the minimum
            scale = 1.0
            while scale > 1e-6:
                trial = w - scale * step
                value = self._objective(A, y, trial, penalty)
                if value <= current + 1e-12:
                    break
                scale /= 2
            else:
                raise FloatingPointError("Newton step could not lower the loss")
            w, current = trial, value
            self.n_iter = it
            if np.max(np.abs(scale * step)) < self.tol:
                break
        else:
            raise RuntimeError(f"no convergence in {self.max_iter} Newton steps")

        self.intercept = float(w[0])
        self.coef = w[1:].copy()
        return self

    def decision(self, X):
        if self.coef is None:
            raise RuntimeError("fit first")
        return self.intercept + np.asarray(X, dtype=float) @ self.coef

    def predict_proba(self, X):
        return sigmoid(self.decision(X))
