"""Reusable correlation and simple-linear-regression tools for Day 44.

Formulas are written out so covariance, Pearson's r, least-squares slope, and
R² stay visible instead of disappearing behind a library call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelationResult:
    """A pairwise association measure with a short interpretive label."""

    method: str
    coefficient: float
    n: int

    @property
    def strength(self) -> str:
        """Return a coarse verbal label for |r|."""
        magnitude = abs(self.coefficient)
        if magnitude < 0.3:
            return "weak"
        if magnitude < 0.7:
            return "moderate"
        return "strong"

    def summary(self) -> str:
        """Return a printable one-line description of the association."""
        direction = "positive" if self.coefficient >= 0 else "negative"
        return (
            f"{self.method} r = {self.coefficient:.4f} "
            f"({self.strength} {direction}, n = {self.n})"
        )


@dataclass(frozen=True)
class RegressionResult:
    """Fitted intercept/slope plus residual diagnostics for y ≈ a + b x."""

    intercept: float
    slope: float
    predictions: np.ndarray
    residuals: np.ndarray
    r_squared: float
    rmse: float
    mae: float
    x: np.ndarray
    y: np.ndarray

    @property
    def equation(self) -> str:
        """Return the fitted line as a readable string."""
        sign = "+" if self.slope >= 0 else "-"
        return f"ŷ = {self.intercept:.4f} {sign} {abs(self.slope):.4f} x"

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Predict from new x values using the fitted intercept and slope."""
        x_new = np.asarray(values, dtype=float)
        if x_new.ndim != 1:
            raise ValueError("Prediction inputs must be a one-dimensional array.")
        if not np.isfinite(x_new).all():
            raise ValueError("Prediction inputs must contain only finite numbers.")
        return self.intercept + self.slope * x_new

    def summary(self) -> str:
        """Return a compact printable summary of the fit."""
        return (
            f"{self.equation} | R² = {self.r_squared:.4f}, "
            f"RMSE = {self.rmse:.4f}, MAE = {self.mae:.4f}"
        )


class RegressionEngine:
    """Compute correlations and ordinary-least-squares simple regression."""

    @staticmethod
    def _as_pair(x: np.ndarray, y: np.ndarray, minimum_size: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """Validate equal-length finite series and return copies."""
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1 or xs.size != ys.size or xs.size < minimum_size:
            raise ValueError(
                f"x and y must be equal-length one-dimensional arrays with at least {minimum_size} values."
            )
        if not np.isfinite(xs).all() or not np.isfinite(ys).all():
            raise ValueError("x and y must contain only finite numbers.")
        return xs.copy(), ys.copy()

    @staticmethod
    def covariance(x: np.ndarray, y: np.ndarray, sample: bool = True) -> float:
        """Return Cov(x, y) using ``n − 1`` (sample) or ``n`` (population)."""
        xs, ys = RegressionEngine._as_pair(x, y, minimum_size=2)
        centered_x = xs - xs.mean()
        centered_y = ys - ys.mean()
        # Sample covariance divides by n − 1 so one draw does not understate spread.
        divisor = xs.size - 1 if sample else xs.size
        return float((centered_x * centered_y).sum() / divisor)

    @staticmethod
    def pearson_correlation(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
        """Return Pearson's r = Cov(x, y) / (s_x s_y).

        Pearson measures linear association.  It is +1 for a perfect rising
        line, −1 for a perfect falling line, and near 0 when there is no
        linear pattern (even if a nonlinear pattern exists).
        """
        xs, ys = RegressionEngine._as_pair(x, y, minimum_size=2)
        std_x = float(xs.std(ddof=1))
        std_y = float(ys.std(ddof=1))
        if np.isclose(std_x, 0.0) or np.isclose(std_y, 0.0):
            raise ValueError("Pearson correlation is undefined when either series has zero variance.")
        coefficient = RegressionEngine.covariance(xs, ys, sample=True) / (std_x * std_y)
        # Floating-point noise can push |r| slightly past 1; clip for reporting.
        coefficient = float(np.clip(coefficient, -1.0, 1.0))
        return CorrelationResult(method="Pearson", coefficient=coefficient, n=xs.size)

    @staticmethod
    def spearman_correlation(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
        """Return Spearman's ρ: Pearson correlation of the ranked values.

        Ranking makes the coefficient sensitive to monotonic relationships,
        not only straight-line ones.
        """
        xs, ys = RegressionEngine._as_pair(x, y, minimum_size=2)
        # argsort(argsort(v)) converts each value into its rank starting at 1.
        rank_x = np.argsort(np.argsort(xs)).astype(float) + 1.0
        rank_y = np.argsort(np.argsort(ys)).astype(float) + 1.0
        pearson_on_ranks = RegressionEngine.pearson_correlation(rank_x, rank_y)
        return CorrelationResult(
            method="Spearman",
            coefficient=pearson_on_ranks.coefficient,
            n=xs.size,
        )

    @staticmethod
    def fit_simple_regression(x: np.ndarray, y: np.ndarray) -> RegressionResult:
        """Fit ŷ = a + b x by ordinary least squares.

        The slope is Cov(x, y) / Var(x).  The intercept then centers the line
        so it passes through the point (x̄, ȳ).
        """
        xs, ys = RegressionEngine._as_pair(x, y, minimum_size=2)
        var_x = float(xs.var(ddof=1))
        if np.isclose(var_x, 0.0):
            raise ValueError("Simple regression requires non-constant x values.")

        slope = RegressionEngine.covariance(xs, ys, sample=True) / var_x
        intercept = float(ys.mean() - slope * xs.mean())
        predictions = intercept + slope * xs
        residuals = ys - predictions

        # R² is the fraction of y variance explained by the linear fit.
        ss_res = float((residuals**2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        r_squared = 1.0 if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))

        return RegressionResult(
            intercept=float(intercept),
            slope=float(slope),
            predictions=predictions,
            residuals=residuals,
            r_squared=float(np.clip(r_squared, 0.0, 1.0)),
            rmse=rmse,
            mae=mae,
            x=xs,
            y=ys,
        )

    @staticmethod
    def generate_study_hours_scores(n: int = 40, seed: int = 44) -> tuple[np.ndarray, np.ndarray]:
        """Simulate exam scores that rise roughly linearly with study hours."""
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer of at least 2.")
        rng = np.random.default_rng(seed)
        hours = rng.uniform(1.0, 12.0, size=int(n))
        # True line: score ≈ 42 + 4.5 × hours, plus modest Normal noise.
        scores = 42.0 + 4.5 * hours + rng.normal(0.0, 5.0, size=int(n))
        scores = np.clip(scores, 0.0, 100.0)
        return hours, scores

    @staticmethod
    def generate_nonlinear_pair(n: int = 40, seed: int = 44) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a curved monotone relationship where Pearson understates association."""
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer of at least 2.")
        rng = np.random.default_rng(seed)
        x = np.linspace(0.2, 3.0, int(n))
        # Exponential growth is monotone, so ranks stay almost perfectly aligned
        # while the raw scatter bends away from a straight line.
        y = np.exp(1.4 * x) + rng.normal(0.0, 1.5, size=int(n))
        return x, y
