"""Matplotlib visualizations for Day 43 confidence intervals."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from confidence_engine import ConfidenceEngine, ConfidenceInterval


class ConfidenceVisualizer:
    """Plot interval estimates, coverage simulations, and width tradeoffs."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        # Create the folder once so every plot method can write without setup.
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_interval(
        self,
        interval: ConfidenceInterval,
        true_value: float | None = None,
        title: str = "Confidence interval",
    ) -> Path:
        """Draw one estimate with error bars for the margin of error."""
        if not isinstance(interval, ConfidenceInterval):
            raise ValueError("A ConfidenceInterval is required to plot the estimate.")

        figure, axis = plt.subplots(figsize=(7, 3.8))
        # Horizontal error bars make the lower/upper bounds easy to read.
        axis.errorbar(
            [interval.estimate],
            [0],
            xerr=[[interval.estimate - interval.lower], [interval.upper - interval.estimate]],
            fmt="o",
            color="#4C78A8",
            ecolor="#4C78A8",
            capsize=8,
            markersize=8,
            label="Point estimate ± MoE",
        )
        if true_value is not None:
            if not np.isfinite(true_value):
                raise ValueError("true_value must be finite when provided.")
            axis.axvline(true_value, color="#E45756", linestyle="--", linewidth=2, label=f"True value = {true_value:g}")
        axis.set(xlabel=interval.name, title=title, yticks=[])
        axis.grid(alpha=0.25, axis="x")
        axis.legend(loc="upper right")
        figure.tight_layout()
        path = self.output_dir / "confidence_interval.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_coverage_simulation(
        self,
        true_mean: float = 50.0,
        sigma: float = 10.0,
        n: int = 30,
        confidence: float = 0.95,
        trials: int = 40,
        seed: int = 43,
        title: str = "Repeated z-intervals and long-run coverage",
    ) -> Path:
        """Plot many random intervals; those missing the truth are highlighted."""
        if trials < 2:
            raise ValueError("trials must be at least 2 for a coverage plot.")
        rng = np.random.default_rng(seed)
        figure, axis = plt.subplots(figsize=(8, 5))
        hits = 0
        for index in range(int(trials)):
            sample = rng.normal(true_mean, sigma, size=int(n))
            interval = ConfidenceEngine.mean_z_interval(sample, sigma, confidence)
            captured = interval.contains(true_mean)
            hits += int(captured)
            color = "#54A24B" if captured else "#E45756"
            # Green segments contain the truth; red segments miss it.
            axis.hlines(index, interval.lower, interval.upper, color=color, linewidth=2)
            axis.plot(interval.estimate, index, "o", color=color, markersize=3)

        axis.axvline(true_mean, color="black", linestyle="--", linewidth=1.5, label=f"True μ = {true_mean:g}")
        empirical = hits / float(trials)
        axis.set(
            xlabel="Interval for μ",
            ylabel="Simulation index",
            title=f"{title}\nEmpirical coverage = {empirical:.1%} (target {confidence:.0%})",
        )
        axis.grid(alpha=0.25, axis="x")
        axis.legend(loc="upper right")
        figure.tight_layout()
        path = self.output_dir / "coverage_simulation.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_width_vs_confidence(
        self,
        sample: np.ndarray,
        confidences: np.ndarray | None = None,
        title: str = "Interval width grows with confidence",
    ) -> Path:
        """Show how raising confidence widens the t-interval for the same sample."""
        data = np.asarray(sample, dtype=float)
        if data.ndim != 1 or data.size < 2:
            raise ValueError("sample must be a one-dimensional array with at least two values.")
        rates = (
            np.array([0.80, 0.90, 0.95, 0.99], dtype=float)
            if confidences is None
            else np.asarray(confidences, dtype=float)
        )
        if rates.ndim != 1 or rates.size == 0:
            raise ValueError("confidences must be a non-empty one-dimensional array.")

        widths = []
        for level in rates:
            interval = ConfidenceEngine.mean_t_interval(data, float(level))
            widths.append(interval.width)

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(rates * 100.0, widths, "o-", color="#F58518", linewidth=2, markersize=7)
        axis.set(xlabel="Confidence level (%)", ylabel="Interval width", title=title)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = self.output_dir / "width_vs_confidence.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
