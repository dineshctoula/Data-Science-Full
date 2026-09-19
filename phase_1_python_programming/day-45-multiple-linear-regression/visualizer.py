"""Matplotlib visualizations for Day 45 multiple linear regression."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from multiple_regression_engine import MultipleRegressionResult


class MultipleRegressionVisualizer:
    """Plot predictions, residuals, coefficients, and multicollinearity VIFs."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_actual_vs_predicted(
        self,
        result: MultipleRegressionResult,
        title: str = "Actual vs predicted",
    ) -> Path:
        """Scatter y against ŷ with a reference diagonal for a perfect fit."""
        if not isinstance(result, MultipleRegressionResult):
            raise ValueError("A MultipleRegressionResult is required.")

        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        axis.scatter(result.predictions, result.predictions + result.residuals, color="#4C78A8", alpha=0.75)
        # predictions + residuals recovers the original y values.
        lo = min(float(result.predictions.min()), float((result.predictions + result.residuals).min()))
        hi = max(float(result.predictions.max()), float((result.predictions + result.residuals).max()))
        axis.plot([lo, hi], [lo, hi], color="#E45756", linestyle="--", linewidth=2, label="Perfect fit")
        axis.set(
            xlabel="Predicted ŷ",
            ylabel="Actual y",
            title=f"{title}\nR² = {result.r_squared:.3f}",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "actual_vs_predicted.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_residuals(
        self,
        result: MultipleRegressionResult,
        title: str = "Residuals vs fitted values",
    ) -> Path:
        """Plot residuals against fitted values to check for leftover structure."""
        if not isinstance(result, MultipleRegressionResult):
            raise ValueError("A MultipleRegressionResult is required.")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.scatter(result.predictions, result.residuals, color="#F58518", alpha=0.85)
        # A well-specified linear model should leave residuals centered on zero.
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set(xlabel="Fitted ŷ", ylabel="Residual y − ŷ", title=title)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = self.output_dir / "residual_plot.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_coefficients(
        self,
        result: MultipleRegressionResult,
        title: str = "Estimated slope coefficients",
    ) -> Path:
        """Bar-chart the non-intercept slopes with readable feature labels."""
        if not isinstance(result, MultipleRegressionResult):
            raise ValueError("A MultipleRegressionResult is required.")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        colors = ["#54A24B" if value >= 0 else "#E45756" for value in result.slopes]
        axis.bar(list(result.feature_names), result.slopes, color=colors, alpha=0.9)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(ylabel="Coefficient", title=title)
        axis.grid(alpha=0.25, axis="y")
        figure.tight_layout()
        path = self.output_dir / "coefficient_bars.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_vif(
        self,
        vifs: dict[str, float],
        title: str = "Variance inflation factors",
    ) -> Path:
        """Plot VIF bars and mark a common multicollinearity warning threshold."""
        if not vifs:
            raise ValueError("At least one VIF value is required.")
        names = list(vifs.keys())
        values = [float(vifs[name]) for name in names]
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("VIF values must be positive and finite.")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.bar(names, values, color="#4C78A8", alpha=0.9)
        # A VIF above 10 is a common rule of thumb for serious collinearity.
        axis.axhline(10.0, color="#E45756", linestyle="--", linewidth=1.5, label="VIF = 10 warning")
        axis.set(ylabel="VIF", title=title)
        axis.grid(alpha=0.25, axis="y")
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "vif_bars.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
