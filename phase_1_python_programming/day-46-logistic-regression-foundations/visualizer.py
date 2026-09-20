"""Matplotlib visualizations for Day 46 logistic regression foundations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from logistic_engine import LogisticEngine, LogisticResult


class LogisticVisualizer:
    """Plot sigmoid curves, probability fits, loss history, and confusion counts."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_sigmoid(self, title: str = "Sigmoid maps scores to probabilities") -> Path:
        """Plot σ(z) = 1 / (1 + e^(−z)) across a wide range of scores."""
        scores = np.linspace(-8.0, 8.0, 400)
        probs = LogisticEngine.sigmoid(scores)

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(scores, probs, color="#4C78A8", linewidth=2.5, label="σ(z)")
        axis.axhline(0.5, color="#E45756", linestyle="--", linewidth=1, label="p = 0.5")
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set(xlabel="Linear score z", ylabel="Probability", title=title, ylim=(-0.05, 1.05))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "sigmoid_curve.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_probability_curve(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        result: LogisticResult,
        title: str = "Fitted pass probability vs study hours",
    ) -> Path:
        """Scatter binary labels and overlay the fitted one-feature sigmoid."""
        if not isinstance(result, LogisticResult):
            raise ValueError("A LogisticResult is required to plot the probability curve.")
        x = np.asarray(features, dtype=float).reshape(-1)
        y = np.asarray(labels, dtype=float).reshape(-1)
        if x.size != y.size or x.size < 2:
            raise ValueError("features and labels must be equal-length arrays with at least two values.")
        if result.n_features != 1:
            raise ValueError("Probability-curve plots require a one-feature model.")

        grid = np.linspace(float(x.min()), float(x.max()), 200).reshape(-1, 1)
        curve = LogisticEngine.predict_proba(grid, result.coefficients)

        figure, axis = plt.subplots(figsize=(7, 4.8))
        # Jitter the 0/1 labels slightly so overlapping points remain visible.
        axis.scatter(x, y, color="#4C78A8", alpha=0.55, label="Observed pass/fail")
        axis.plot(grid[:, 0], curve, color="#E45756", linewidth=2.5, label="Fitted P(y=1 | x)")
        axis.axhline(0.5, color="#54A24B", linestyle="--", linewidth=1, label="Threshold 0.5")
        axis.set(xlabel=result.feature_names[0], ylabel="Probability / label", title=title, ylim=(-0.05, 1.05))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "probability_curve.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_loss_history(self, result: LogisticResult, title: str = "Binary cross-entropy loss") -> Path:
        """Plot training loss across gradient-descent updates."""
        if not isinstance(result, LogisticResult):
            raise ValueError("A LogisticResult is required to plot loss history.")
        losses = np.asarray(result.loss_history, dtype=float)
        if losses.ndim != 1 or losses.size < 2:
            raise ValueError("loss_history must contain at least two values.")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(np.arange(losses.size), losses, color="#54A24B", linewidth=2)
        axis.set(xlabel="Iteration", ylabel="Mean binary cross-entropy", title=title)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = self.output_dir / "loss_history.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        title: str = "Confusion matrix",
    ) -> Path:
        """Draw a 2×2 confusion matrix with TN/FP/FN/TP annotations."""
        values = np.asarray(matrix, dtype=float)
        if values.shape != (2, 2):
            raise ValueError("Confusion matrix must have shape (2, 2).")

        figure, axis = plt.subplots(figsize=(5.5, 4.8))
        image = axis.imshow(values, cmap="Blues")
        axis.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        axis.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        labels = [["TN", "FP"], ["FN", "TP"]]
        # Annotate each cell with its role and count for quick reading.
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{labels[row][column]}\n{int(values[row, column])}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        path = self.output_dir / "confusion_matrix.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
