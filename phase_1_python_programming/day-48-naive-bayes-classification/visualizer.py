"""Matplotlib visualizations for Day 48 Naive Bayes classification."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from naive_bayes_engine import NaiveBayesEngine, NaiveBayesResult


class NaiveBayesVisualizer:
    """Plot priors, class-conditional densities, 2-D decision regions, and confusion."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_class_priors(self, result: NaiveBayesResult, title: str = "Class priors") -> Path:
        """Bar-chart the Laplace-smoothed prior for each class."""
        if not isinstance(result, NaiveBayesResult):
            raise ValueError("A NaiveBayesResult is required to plot class priors.")

        labels = [f"{stats.label:g}" for stats in result.class_stats]
        priors = [stats.prior for stats in result.class_stats]
        figure, axis = plt.subplots(figsize=(6.5, 4.5))
        axis.bar(labels, priors, color="#4C78A8", alpha=0.9)
        axis.set(xlabel="Class", ylabel="P(y)", title=title, ylim=(0, max(priors) * 1.25))
        axis.grid(alpha=0.25, axis="y")
        for index, prior in enumerate(priors):
            axis.text(index, prior + 0.01, f"{prior:.2f}", ha="center")
        figure.tight_layout()
        path = self.output_dir / "class_priors.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_feature_likelihoods(
        self,
        result: NaiveBayesResult,
        feature_index: int = 0,
        title: str = "Class-conditional feature densities",
    ) -> Path:
        """Overlay the fitted Normal likelihoods for one feature across classes."""
        if not isinstance(result, NaiveBayesResult):
            raise ValueError("A NaiveBayesResult is required to plot likelihoods.")
        if feature_index < 0 or feature_index >= result.n_features:
            raise ValueError("feature_index is out of range.")

        figure, axis = plt.subplots(figsize=(7, 4.8))
        colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
        # Span a few standard deviations around every class mean for that feature.
        means = [stats.means[feature_index] for stats in result.class_stats]
        stds = [np.sqrt(stats.variances[feature_index]) for stats in result.class_stats]
        low = min(m - 4 * s for m, s in zip(means, stds))
        high = max(m + 4 * s for m, s in zip(means, stds))
        xs = np.linspace(low, high, 300)

        for index, stats in enumerate(result.class_stats):
            density = np.exp(
                NaiveBayesEngine.gaussian_log_pdf(
                    xs, float(stats.means[feature_index]), float(stats.variances[feature_index])
                )
            )
            axis.plot(
                xs,
                density,
                color=colors[index % len(colors)],
                linewidth=2.5,
                label=f"class {stats.label:g}",
            )
        axis.set(
            xlabel=result.feature_names[feature_index],
            ylabel="Density",
            title=title,
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "feature_likelihoods.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_decision_regions(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        result: NaiveBayesResult,
        title: str = "Naive Bayes decision regions",
    ) -> Path:
        """Shade the 2-D MAP regions and overlay the training scatter."""
        if not isinstance(result, NaiveBayesResult):
            raise ValueError("A NaiveBayesResult is required to plot decision regions.")
        if result.n_features != 2:
            raise ValueError("Decision-region plots require exactly two features.")
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] != y.size:
            raise ValueError("features must be an (n, 2) array aligned with labels.")

        padding = 0.75
        x_min, x_max = x[:, 0].min() - padding, x[:, 0].max() + padding
        y_min, y_max = x[:, 1].min() - padding, x[:, 1].max() + padding
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        # Predict on the dense grid so region colors show the MAP partition.
        region = NaiveBayesEngine.predict(grid, result.class_stats).reshape(xx.shape)

        figure, axis = plt.subplots(figsize=(7, 5.5))
        axis.contourf(xx, yy, region, alpha=0.25, cmap="viridis")
        scatter = axis.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", edgecolor="white", s=40)
        axis.set(
            xlabel=result.feature_names[0],
            ylabel=result.feature_names[1],
            title=f"{title}\naccuracy = {result.accuracy:.3f}",
        )
        figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04, label="class")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        path = self.output_dir / "decision_regions.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        classes: np.ndarray,
        title: str = "Confusion matrix",
    ) -> Path:
        """Draw a labeled square confusion matrix."""
        values = np.asarray(matrix, dtype=float)
        class_list = np.asarray(classes, dtype=float).reshape(-1)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError("Confusion matrix must be square.")
        if values.shape[0] != class_list.size:
            raise ValueError("classes length must match the confusion matrix size.")

        figure, axis = plt.subplots(figsize=(5.8, 5.0))
        image = axis.imshow(values, cmap="Blues")
        ticks = [f"{label:g}" for label in class_list]
        axis.set_xticks(range(class_list.size), ticks)
        axis.set_yticks(range(class_list.size), ticks)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{int(values[row, column])}",
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
