"""Day 59 — bias vs variance.

A wiggly fit chases the noise in one sample. A stiff fit misses the curve
on every sample. Average a bunch of training sets and you can split the
error into those two pieces, plus the noise you can't fit no matter what.

Here the truth is a sine with a tilt. We refit polynomials and watch
bias fall while variance climbs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def true_curve(x):
    """The function we're pretending not to know."""
    x = np.asarray(x, dtype=float)
    return np.sin(1.4 * x) + 0.25 * x


def draw_sample(n: int, noise: float, rng, x_low: float = -3.0, x_high: float = 3.0):
    x = rng.uniform(x_low, x_high, size=n)
    y = true_curve(x) + rng.normal(0.0, noise, size=n)
    return x, y


def fit_polynomial(x, y, degree: int) -> np.ndarray:
    """Highest power first, same order np.polyval expects."""
    if degree < 0:
        raise ValueError("degree must be >= 0")
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("x and y length mismatch")
    if len(x) <= degree:
        raise ValueError("need more rows than the polynomial degree")
    # polyfit warns when the vandermonde is ill-conditioned; still usable here
    return np.polyfit(x, y, degree)


def predict_polynomial(coef, x):
    return np.polyval(coef, np.asarray(x, dtype=float))


@dataclass
class BiasVariancePoint:
    degree: int
    bias2: float
    variance: float
    noise: float
    mse: float
    n_runs: int

    def summary(self) -> str:
        return (
            f"deg={self.degree}: bias²={self.bias2:.3f}  var={self.variance:.3f}  "
            f"noise={self.noise:.3f}  mse≈{self.mse:.3f}"
        )


def decompose_degrees(
    degrees,
    n_train: int = 35,
    n_runs: int = 40,
    noise: float = 0.45,
    n_grid: int = 60,
    seed: int = 59,
) -> list[BiasVariancePoint]:
    """Refit each degree on fresh samples and average the error pieces on a grid.

    bias² = mean( (average prediction - truth)^2 )
    variance = mean over x of the variance across runs
    mse = bias² + variance + noise²
    """
    degrees = [int(d) for d in degrees]
    if any(d < 0 for d in degrees):
        raise ValueError("degrees must be >= 0")
    if n_runs < 2:
        raise ValueError("need at least 2 runs to estimate variance")
    if noise < 0:
        raise ValueError("noise must be >= 0")

    rng = np.random.default_rng(seed)
    grid = np.linspace(-2.6, 2.6, n_grid)
    truth = true_curve(grid)
    noise_var = float(noise ** 2)
    out = []

    for degree in degrees:
        preds = np.empty((n_runs, n_grid))
        for i in range(n_runs):
            x, y = draw_sample(n_train, noise, rng)
            coef = fit_polynomial(x, y, degree)
            preds[i] = predict_polynomial(coef, grid)

        mean_pred = preds.mean(axis=0)
        bias2 = float(np.mean((mean_pred - truth) ** 2))
        # ddof=1 so a tiny sample of runs isn't pretending to be the population
        variance = float(np.mean(preds.var(axis=0, ddof=1)))
        out.append(
            BiasVariancePoint(
                degree=degree,
                bias2=bias2,
                variance=variance,
                noise=noise_var,
                mse=bias2 + variance + noise_var,
                n_runs=n_runs,
            )
        )
    return out


def sample_fit_curves(degree: int, n_curves: int = 12, n_train: int = 35, noise: float = 0.45, seed: int = 59):
    """A handful of fits on the same grid, for the spaghetti plot."""
    rng = np.random.default_rng(seed + degree)
    grid = np.linspace(-2.6, 2.6, 80)
    curves = []
    for _ in range(n_curves):
        x, y = draw_sample(n_train, noise, rng)
        coef = fit_polynomial(x, y, degree)
        curves.append(predict_polynomial(coef, grid))
    return grid, true_curve(grid), np.vstack(curves)
