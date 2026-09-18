"""Matplotlib visualizations for Day 44 correlation and regression."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from regression_engine import CorrelationResult, RegressionEngine, RegressionResult


class RegressionVisualizer:
    """Plot scatter fits, residual patterns, and Pearson-vs-Spearman contrasts."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_scatter_with_fit(
        self,
        result: RegressionResult,
        title: str = "Simple linear regression",
    ) -> Path:
        """Scatter the observations and overlay the fitted OLS line."""
        if not isinstance(result, RegressionResult):
            raise ValueError("A RegressionResult is required to plot the fitted line.")

        figure, axis = plt.subplots(figsize=(7, 4.8))
        axis.scatter(result.x, result.y, color="#4C78A8", alpha=0.8, label="Observations")
        # Sort x so the line draws left-to-right without zigzagging.
        order = np.argsort(result.x)
        axis.plot(
            result.x[order],
            result.predictions[order],
            color="#E45756",
            linewidth=2.5,
            label=result.equation,
        )
        axis.set(xlabel="x", ylabel="y", title=f"{title}\nR² = {result.r_squared:.3f}")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "scatter_with_fit.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_residuals(self, result: RegressionResult, title: str = "Residual plot") -> Path:
        """Plot residuals against fitted values to check for leftover patterns."""
        if not isinstance(result, RegressionResult):
            raise ValueError("A RegressionResult is required to plot residuals.")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.scatter(result.predictions, result.residuals, color="#F58518", alpha=0.85)
        # A good linear fit should leave residuals scattered around this zero line.
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set(xlabel="Fitted ŷ", ylabel="Residual y − ŷ", title=title)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = self.output_dir / "residual_plot.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_correlation_comparison(
        self,
        pearson: CorrelationResult,
        spearman: CorrelationResult,
        x: np.ndarray,
        y: np.ndarray,
        title: str = "Pearson vs Spearman on a curved relationship",
    ) -> Path:
        """Show a curved scatter beside bars comparing the two coefficients."""
        if not isinstance(pearson, CorrelationResult) or not isinstance(spearman, CorrelationResult):
            raise ValueError("Pearson and Spearman CorrelationResult values are required.")
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1 or xs.size != ys.size or xs.size < 2:
            raise ValueError("x and y must be equal-length one-dimensional arrays with at least two values.")

        figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        axes[0].scatter(xs, ys, color="#4C78A8", alpha=0.85)
        axes[0].set(xlabel="x", ylabel="y", title="Curved monotone scatter")
        axes[0].grid(alpha=0.25)

        labels = ["Pearson", "Spearman"]
        values = [pearson.coefficient, spearman.coefficient]
        axes[1].bar(labels, values, color=["#4C78A8", "#54A24B"], alpha=0.9)
        axes[1].set(ylim=(0, 1.05), ylabel="Correlation", title="Coefficient comparison")
        axes[1].grid(alpha=0.25, axis="y")
        for index, value in enumerate(values):
            axes[1].text(index, value + 0.02, f"{value:.3f}", ha="center")

        figure.suptitle(title, fontsize=13, fontweight="bold")
        figure.tight_layout()
        path = self.output_dir / "correlation_comparison.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
