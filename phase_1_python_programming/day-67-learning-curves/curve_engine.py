"""Day 67 — learning curves.

A learning curve is just train error and test error as you hand the model
more rows. If both are already flat, extra rows will not save a model that
is too stiff. If the test error is still falling toward the train error,
the fit was hungry for data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_sine(n: int = 500, noise: float = 0.4, seed: int = 67):
    """y = sin(1.4 x) plus noise. A straight line cannot follow the bend."""
    if n < 20:
        raise ValueError("need at least 20 rows")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n)
    y = np.sin(1.4 * x) + noise * rng.normal(size=n)
    return x, y


def poly_design(x, degree: int):
    if degree < 0:
        raise ValueError("degree should be 0 or more")
    x = np.asarray(x, dtype=float).reshape(-1)
    cols = [np.ones(len(x))]
    for power in range(1, degree + 1):
        cols.append(x ** power)
    return np.column_stack(cols)


def fit_poly(x, y, degree: int):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("x and y differ in length")
    if len(x) < degree + 1:
        raise ValueError("not enough rows for this degree")
    design = poly_design(x, degree)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


def predict_poly(x, beta) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).reshape(-1)
    return poly_design(x, len(beta) - 1) @ beta


def _mse(y, pred) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    return float(np.mean((y - pred) ** 2))


@dataclass
class LearningCurve:
    degree: int
    sizes: np.ndarray
    train_mse: np.ndarray
    test_mse: np.ndarray

    def gap(self) -> np.ndarray:
        # positive means the training number looks better than the test
        return self.test_mse - self.train_mse


def learning_curve(x, y, degree: int, sizes, n_test: int = 100, repeats: int = 16, seed: int = 67) -> LearningCurve:
    """Mean train/test MSE at each size.

    One test slice stays put. Each point averages `repeats` random subsets
    drawn from whatever is left.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("x and y differ in length")
    if repeats < 1:
        raise ValueError("need at least one repeat")
    if n_test < 1 or n_test >= len(x):
        raise ValueError("n_test should leave a training pool")

    sizes = [int(s) for s in sizes]
    if len(sizes) == 0:
        raise ValueError("no sizes")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    test = order[:n_test]
    pool = order[n_test:]
    for size in sizes:
        if size < degree + 1 or size > len(pool):
            raise ValueError(f"size {size} does not fit degree {degree} and the pool")

    train_mse = np.zeros(len(sizes))
    test_mse = np.zeros(len(sizes))
    for i, size in enumerate(sizes):
        train_draws = []
        test_draws = []
        for rep in range(repeats):
            # mix size into the seed so two sizes don't reuse the same draw
            draw = np.random.default_rng(seed + 10007 * (rep + 1) + size)
            subset = draw.choice(pool, size=size, replace=False)
            beta = fit_poly(x[subset], y[subset], degree)
            train_draws.append(_mse(y[subset], predict_poly(x[subset], beta)))
            test_draws.append(_mse(y[test], predict_poly(x[test], beta)))
        train_mse[i] = float(np.mean(train_draws))
        test_mse[i] = float(np.mean(test_draws))

    return LearningCurve(
        degree=degree,
        sizes=np.asarray(sizes, dtype=int),
        train_mse=train_mse,
        test_mse=test_mse,
    )
