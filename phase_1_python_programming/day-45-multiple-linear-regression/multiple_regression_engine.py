"""Reusable multiple-linear-regression tools for the Day 45 lesson.

Coefficients come from the normal equations so the design matrix, intercept
column, and residual diagnostics stay visible while they are being learned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MultipleRegressionResult:
    """Fitted multi-predictor model with residual and fit diagnostics."""

    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    predictions: np.ndarray
    residuals: np.ndarray
    r_squared: float
    adjusted_r_squared: float
    rmse: float
    mae: float
    n_observations: int
    n_features: int

    @property
    def intercept(self) -> float:
        """Return the leading intercept coefficient."""
        return float(self.coefficients[0])

    @property
    def slopes(self) -> np.ndarray:
        """Return the slope coefficients that multiply each feature."""
        return self.coefficients[1:].copy()

    @property
    def equation(self) -> str:
        """Return a readable ŷ = a + b₁x₁ + … expression."""
        parts = [f"{self.intercept:.4f}"]
        for name, slope in zip(self.feature_names, self.slopes):
            sign = "+" if slope >= 0 else "-"
            parts.append(f"{sign} {abs(slope):.4f}·{name}")
        return "ŷ = " + " ".join(parts)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict from a feature matrix with the same columns used at fit time."""
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"Features must be a 2-D array with {self.n_features} columns."
            )
        if not np.isfinite(x).all():
            raise ValueError("Features must contain only finite numbers.")
        # Reattach the intercept column of ones used by the normal equations.
        design = np.column_stack((np.ones(x.shape[0]), x))
        return design @ self.coefficients

    def summary(self) -> str:
        """Return a compact printable summary of the fit."""
        return (
            f"{self.equation} | R² = {self.r_squared:.4f}, "
            f"Adj. R² = {self.adjusted_r_squared:.4f}, RMSE = {self.rmse:.4f}"
        )


class MultipleRegressionEngine:
    """Fit multiple linear regression with ordinary least squares."""

    @staticmethod
    def _validate_design(
        features: np.ndarray,
        target: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Validate X and y, then return copies plus feature labels."""
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("Features must be a non-empty two-dimensional array.")
        if y.shape[0] != x.shape[0]:
            raise ValueError("Target length must match the number of feature rows.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Features and target must contain only finite numbers.")
        # Need more rows than columns (including intercept) for a unique OLS fit.
        if x.shape[0] <= x.shape[1] + 1:
            raise ValueError("Need more observations than predictors plus the intercept.")

        if feature_names is None:
            names = tuple(f"x{index + 1}" for index in range(x.shape[1]))
        else:
            names = tuple(feature_names)
            if len(names) != x.shape[1]:
                raise ValueError("feature_names length must match the number of columns.")
        return x.copy(), y.copy(), names

    @staticmethod
    def design_matrix(features: np.ndarray) -> np.ndarray:
        """Return [1 | X] so the first coefficient is an intercept."""
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Features must be a non-empty two-dimensional array.")
        return np.column_stack((np.ones(x.shape[0]), x))

    @staticmethod
    def fit(
        features: np.ndarray,
        target: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
    ) -> MultipleRegressionResult:
        """Fit ŷ = β₀ + β₁x₁ + … + βₚxₚ by solving the normal equations.

        The estimator is β̂ = (XᵀX)⁻¹ Xᵀy.  Using ``np.linalg.solve`` avoids
        forming an explicit inverse and is more numerically stable.
        """
        x, y, names = MultipleRegressionEngine._validate_design(features, target, feature_names)
        design = MultipleRegressionEngine.design_matrix(x)
        gram = design.T @ design
        # A near-singular Gram matrix usually means collinear predictors.
        if abs(np.linalg.det(gram)) < 1e-12:
            raise ValueError("Design matrix is singular; check for collinear features.")
        rhs = design.T @ y
        coefficients = np.linalg.solve(gram, rhs)

        predictions = design @ coefficients
        residuals = y - predictions
        ss_res = float((residuals**2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r_squared = 1.0 if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot

        n, p = x.shape
        # Adjusted R² penalizes unused predictors so adding noise does not look better.
        adjusted = 1.0 - (1.0 - r_squared) * (n - 1) / (n - p - 1)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))

        return MultipleRegressionResult(
            coefficients=coefficients,
            feature_names=names,
            predictions=predictions,
            residuals=residuals,
            r_squared=float(np.clip(r_squared, 0.0, 1.0)),
            adjusted_r_squared=float(adjusted),
            rmse=rmse,
            mae=mae,
            n_observations=n,
            n_features=p,
        )

    @staticmethod
    def variance_inflation_factors(
        features: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
    ) -> dict[str, float]:
        """Return VIF for each predictor: 1 / (1 − R²_j).

        Each feature is regressed on the others.  VIF ≫ 5–10 suggests that
        column is nearly a linear combination of the rest.
        """
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError("VIF requires at least two feature columns.")
        # A dummy target is enough to reuse the shared design validation rules.
        dummy_target = np.zeros(x.shape[0])
        x, _, names = MultipleRegressionEngine._validate_design(x, dummy_target, feature_names)

        vifs: dict[str, float] = {}
        for index, name in enumerate(names):
            # Regress feature j on all remaining features to measure redundancy.
            others = np.delete(x, index, axis=1)
            target_feature = x[:, index]
            other_names = tuple(label for i, label in enumerate(names) if i != index)
            fit = MultipleRegressionEngine.fit(others, target_feature, other_names)
            if np.isclose(fit.r_squared, 1.0):
                raise ValueError(f"Feature '{name}' is perfectly collinear with the others.")
            vifs[name] = float(1.0 / (1.0 - fit.r_squared))
        return vifs

    @staticmethod
    def generate_housing_sample(
        n: int = 80, seed: int = 45
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Simulate house prices from size, bedrooms, and distance to center."""
        if not isinstance(n, (int, np.integer)) or n < 10:
            raise ValueError("n must be an integer of at least 10.")
        rng = np.random.default_rng(seed)
        size = rng.uniform(600.0, 2500.0, size=int(n))
        bedrooms = rng.integers(1, 5, size=int(n)).astype(float)
        distance = rng.uniform(0.5, 20.0, size=int(n))
        # True surface: price ≈ 40k + 80·sqft + 15k·beds − 2.5k·miles + noise.
        noise = rng.normal(0.0, 18_000.0, size=int(n))
        price = 40_000.0 + 80.0 * size + 15_000.0 * bedrooms - 2_500.0 * distance + noise
        features = np.column_stack((size, bedrooms, distance))
        names = ("sqft", "bedrooms", "distance_km")
        return features, price, names

    @staticmethod
    def generate_collinear_sample(
        n: int = 60, seed: int = 45
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Simulate predictors where x2 ≈ 2·x1 so VIF becomes large."""
        if not isinstance(n, (int, np.integer)) or n < 10:
            raise ValueError("n must be an integer of at least 10.")
        rng = np.random.default_rng(seed)
        x1 = rng.normal(0.0, 1.0, size=int(n))
        # Almost-duplicate information: x2 is a noisy multiple of x1.
        x2 = 2.0 * x1 + rng.normal(0.0, 0.05, size=int(n))
        x3 = rng.normal(0.0, 1.0, size=int(n))
        y = 3.0 + 1.5 * x1 + 0.2 * x3 + rng.normal(0.0, 0.4, size=int(n))
        features = np.column_stack((x1, x2, x3))
        return features, y, ("x1", "x2", "x3")
