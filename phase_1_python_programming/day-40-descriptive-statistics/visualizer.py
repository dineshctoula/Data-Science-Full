"""Matplotlib visualizations for Day 40 descriptive statistics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stats_engine import DescriptiveReport, FiveNumberSummary


class StatsVisualizer:
    """Create concise plots that connect a numeric sample to its summary."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _as_sample(values: np.ndarray, name: str = "values") -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size < 2:
            raise ValueError(f"{name} must be a one-dimensional array with at least two values.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite numbers.")
        return array

    def plot_histogram(self, values: np.ndarray, mean: float, median: float, title: str = "Distribution") -> Path:
        """Plot a histogram with the mean and median marked on the same axis."""
        sample = self._as_sample(values)
        if not np.isfinite(mean) or not np.isfinite(median):
            raise ValueError("Mean and median must be finite numbers.")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.hist(sample, bins="auto", color="#4C78A8", edgecolor="white", alpha=0.9)
        axis.axvline(mean, color="#E45756", linewidth=2, label=f"Mean = {mean:.2f}")
        axis.axvline(median, color="#54A24B", linewidth=2, linestyle="--", label=f"Median = {median:.2f}")
        axis.set(xlabel="Value", ylabel="Count", title=title)
        axis.grid(alpha=0.25, axis="y")
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "distribution_histogram.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_boxplot(self, values: np.ndarray, summary: FiveNumberSummary, title: str = "Five-number summary") -> Path:
        """Plot a box plot whose whiskers follow the same 1.5-IQR fences as the engine."""
        sample = self._as_sample(values)
        if not isinstance(summary, FiveNumberSummary):
            raise ValueError("A FiveNumberSummary is required to annotate the box plot.")

        figure, axis = plt.subplots(figsize=(7, 3.8))
        axis.boxplot(
            sample,
            vert=False,
            whis=1.5,
            patch_artist=True,
            boxprops={"facecolor": "#4C78A8", "alpha": 0.35, "color": "#4C78A8"},
            medianprops={"color": "#54A24B", "linewidth": 2},
            whiskerprops={"color": "#4C78A8"},
            capprops={"color": "#4C78A8"},
            flierprops={"marker": "o", "color": "#E45756", "markerfacecolor": "#E45756"},
        )
        axis.scatter(
            [summary.minimum, summary.q1, summary.median, summary.q3, summary.maximum],
            [1, 1, 1, 1, 1],
            color="#F58518",
            zorder=3,
            label="Five-number points",
        )
        axis.set(xlabel="Value", title=title)
        axis.set_yticks([])
        axis.grid(alpha=0.25, axis="x")
        axis.legend(loc="upper left")
        figure.tight_layout()
        path = self.output_dir / "five_number_boxplot.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_zscore_strip(self, report: DescriptiveReport, title: str = "Standardized exam scores") -> Path:
        """Plot z-scores so typical, unusual, and outlying points are comparable."""
        if not isinstance(report, DescriptiveReport):
            raise ValueError("A DescriptiveReport is required to plot z-scores.")
        z_scores = self._as_sample(report.z_scores, "z_scores")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        typical = ~report.outlier_mask
        axis.scatter(np.where(typical)[0], z_scores[typical], color="#4C78A8", label="Inside IQR fences")
        if report.outlier_count:
            axis.scatter(
                np.where(report.outlier_mask)[0],
                z_scores[report.outlier_mask],
                color="#E45756",
                zorder=3,
                label="1.5-IQR outliers",
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axhline(2.0, color="#F58518", linestyle="--", linewidth=1, label="|z| = 2")
        axis.axhline(-2.0, color="#F58518", linestyle="--", linewidth=1)
        axis.set(xlabel="Observation index", ylabel="Z-score", title=title)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "zscore_strip.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
