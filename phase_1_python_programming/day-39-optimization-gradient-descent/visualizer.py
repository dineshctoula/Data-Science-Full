"""Matplotlib visualizations for Day 39 gradient-descent paths and losses."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class OptimizationVisualizer:
    """Create concise plots that show how step size and curvature affect descent."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _as_curve(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size < 2:
            raise ValueError(f"{name} must be a one-dimensional array with at least two values.")
        return array

    def plot_loss_history(self, loss_history: np.ndarray, title: str = "Gradient descent loss") -> Path:
        """Plot objective loss across gradient-descent updates on a log scale."""
        losses = self._as_curve(loss_history, "loss_history")
        if np.any(losses < 0):
            raise ValueError("Loss values cannot be negative.")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(np.arange(losses.size), losses, color="#54A24B", linewidth=2)
        axis.scatter([0, losses.size - 1], [losses[0], losses[-1]], color="#E45756", zorder=3)
        axis.set(xlabel="Iteration", ylabel="Loss", title=title)
        if np.all(losses > 0):
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = self.output_dir / "gradient_descent_loss.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_contour_path(
        self,
        parameter_history: np.ndarray,
        objective: Callable[[np.ndarray], float],
        target: np.ndarray | None = None,
        title: str = "Gradient descent on a 2-D bowl",
    ) -> Path:
        """Plot a 2-D loss contour with the recorded descent path overlaid."""
        path = np.asarray(parameter_history, dtype=float)
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 2:
            raise ValueError("parameter_history must be a (steps, 2) array with at least two points.")

        padding = 1.0
        lower = path.min(axis=0) - padding
        upper = path.max(axis=0) + padding
        if target is not None:
            minimum = np.asarray(target, dtype=float)
            if minimum.shape != (2,):
                raise ValueError("target must be a two-dimensional point.")
            lower = np.minimum(lower, minimum - padding)
            upper = np.maximum(upper, minimum + padding)
        else:
            minimum = None

        xs = np.linspace(lower[0], upper[0], 80)
        ys = np.linspace(lower[1], upper[1], 80)
        grid_x, grid_y = np.meshgrid(xs, ys)
        surface = np.empty_like(grid_x)
        # The bowl is cheap to evaluate, so a dense grid stays readable without
        # hiding the curved path that the uneven curvature produces.
        for row in range(grid_x.shape[0]):
            for column in range(grid_x.shape[1]):
                surface[row, column] = objective(np.array([grid_x[row, column], grid_y[row, column]]))

        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        contours = axis.contour(grid_x, grid_y, surface, levels=12, cmap="Blues")
        axis.clabel(contours, inline=True, fontsize=8)
        axis.plot(path[:, 0], path[:, 1], color="#E45756", linewidth=2, marker="o", markersize=3, label="Descent path")
        axis.scatter(path[0, 0], path[0, 1], color="#4C78A8", s=70, zorder=3, label="Start")
        axis.scatter(path[-1, 0], path[-1, 1], color="#54A24B", s=70, zorder=3, label="End")
        if minimum is not None:
            axis.scatter(minimum[0], minimum[1], color="#F58518", marker="*", s=140, zorder=4, label="Minimum")
        axis.set(xlabel="x", ylabel="y", title=title)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
        figure.tight_layout()
        image_path = self.output_dir / "gradient_descent_contour.png"
        figure.savefig(image_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return image_path

    def plot_learning_rate_comparison(self, loss_histories: dict[float, np.ndarray]) -> Path:
        """Plot several loss curves to make learning-rate behavior comparable."""
        if not loss_histories:
            raise ValueError("At least one learning-rate history is required.")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        all_losses: list[np.ndarray] = []
        for rate, history in sorted(loss_histories.items()):
            losses = self._as_curve(history, f"loss history for learning rate {rate}")
            if np.any(losses < 0):
                raise ValueError("Loss values cannot be negative.")
            all_losses.append(losses)
            # Plot the complete history for each rate on common axes.  Seeing
            # the curves together reveals slow progress and instability fast.
            axis.plot(np.arange(losses.size), losses, linewidth=2, label=f"rate = {rate:g}")

        axis.set(xlabel="Iteration", ylabel="Loss", title="Learning-rate comparison")
        # A zero loss is valid; only use logarithmic scaling when every value
        # can be represented on that scale.
        if np.all(np.concatenate(all_losses) > 0):
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "learning_rate_comparison.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
