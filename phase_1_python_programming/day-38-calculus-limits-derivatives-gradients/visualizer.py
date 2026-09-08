"""Matplotlib visualizations for Day 38 calculus concepts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class CalculusVisualizer:
    """Create concise plots linking calculus formulae to model optimization."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _as_curve(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size < 2:
            raise ValueError(f"{name} must be a one-dimensional array with at least two values.")
        return array

    def plot_tangent_line(
        self, x_values: np.ndarray, y_values: np.ndarray, point: float, slope: float
    ) -> Path:
        """Plot a function curve and its local tangent line at a chosen point."""
        x = self._as_curve(x_values, "x_values")
        y = self._as_curve(y_values, "y_values")
        if x.shape != y.shape:
            raise ValueError("x_values and y_values must have equal shapes.")
        y_at_point = float(np.interp(point, x, y))
        tangent = y_at_point + slope * (x - point)

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(x, y, color="#4C78A8", linewidth=2.5, label="f(x)")
        axis.plot(x, tangent, color="#E45756", linestyle="--", label=f"Tangent slope = {slope:.2f}")
        axis.scatter([point], [y_at_point], color="#E45756", zorder=3)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set(xlabel="x", ylabel="f(x)", title="Derivative as the slope of a tangent line")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "derivative_tangent_line.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_loss_history(self, loss_history: np.ndarray, title: str = "Gradient descent loss") -> Path:
        """Plot objective loss across gradient-descent updates on a log scale."""
        losses = self._as_curve(loss_history, "loss_history")
        if np.any(losses < 0):
            raise ValueError("Loss values cannot be negative.")
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(np.arange(losses.size), losses, color="#54A24B", linewidth=2)
        axis.scatter([0, losses.size - 1], [losses[0], losses[-1]], color="#E45756", zorder=3)
        axis.set(xlabel="Iteration", ylabel="Mean squared error", title=title)
        if np.all(losses > 0):
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = self.output_dir / "gradient_descent_loss.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
