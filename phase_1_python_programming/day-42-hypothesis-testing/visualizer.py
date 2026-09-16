"""Matplotlib visualizations for Day 42 hypothesis testing."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hypothesis_engine import HypothesisEngine, HypothesisResult


class HypothesisVisualizer:
    """Plot null distributions, conversion bars, and chi-square contributions."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_null_distribution(
        self,
        result: HypothesisResult,
        title: str = "Null reference distribution",
    ) -> Path:
        """Shade the p-value region on a Normal or t reference curve."""
        if not isinstance(result, HypothesisResult):
            raise ValueError("A HypothesisResult is required to plot the null distribution.")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        stat = float(result.statistic)

        if result.degrees_of_freedom is None:
            # z-tests use the standard Normal as the large-sample reference.
            xs = np.linspace(-4.0, 4.0, 400)
            density = np.array(
                [np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi) for x in xs]
            )
            axis.plot(xs, density, color="#4C78A8", linewidth=2.5, label="Standard Normal")
            shade_x = xs
        else:
            df = float(result.degrees_of_freedom)
            xs = np.linspace(-4.0, 4.0, 400)
            density = np.array([HypothesisEngine._student_t_pdf(x, df) for x in xs])
            axis.plot(xs, density, color="#4C78A8", linewidth=2.5, label=f"t(df={df:g})")
            shade_x = xs

        # Two-sided shading mirrors the tail areas that define the p-value.
        if "≠" in result.alternative or "two-sided" in result.alternative.lower():
            tail = shade_x[np.abs(shade_x) >= abs(stat)]
            tail_y = np.array(
                [
                    HypothesisEngine._student_t_pdf(x, result.degrees_of_freedom)
                    if result.degrees_of_freedom is not None
                    else np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
                    for x in tail
                ]
            )
            axis.fill_between(tail, 0, tail_y, color="#E45756", alpha=0.35, label="p-value tails")
        elif ">" in result.alternative:
            tail = shade_x[shade_x >= stat]
            tail_y = np.array(
                [
                    HypothesisEngine._student_t_pdf(x, result.degrees_of_freedom)
                    if result.degrees_of_freedom is not None
                    else np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
                    for x in tail
                ]
            )
            axis.fill_between(tail, 0, tail_y, color="#E45756", alpha=0.35, label="p-value tail")
        else:
            tail = shade_x[shade_x <= stat]
            tail_y = np.array(
                [
                    HypothesisEngine._student_t_pdf(x, result.degrees_of_freedom)
                    if result.degrees_of_freedom is not None
                    else np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
                    for x in tail
                ]
            )
            axis.fill_between(tail, 0, tail_y, color="#E45756", alpha=0.35, label="p-value tail")

        axis.axvline(stat, color="#54A24B", linestyle="--", linewidth=2, label=f"statistic = {stat:.2f}")
        axis.set(xlabel="Test statistic", ylabel="Density", title=title)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "null_distribution.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_conversion_rates(
        self,
        successes_a: int,
        n_a: int,
        successes_b: int,
        n_b: int,
        title: str = "A/B conversion rates",
    ) -> Path:
        """Compare sample proportions with error bars derived from √(p(1−p)/n)."""
        for label, value in (("successes_A", successes_a), ("n_A", n_a), ("successes_B", successes_b), ("n_B", n_b)):
            if not isinstance(value, (int, np.integer)) or int(value) < 0:
                raise ValueError(f"{label} must be a non-negative integer.")
        if successes_a > n_a or successes_b > n_b:
            raise ValueError("successes cannot exceed trial counts.")

        props = np.array([successes_a / n_a, successes_b / n_b])
        # Normal-approximation standard errors for the bar chart whiskers.
        errors = np.sqrt(props * (1.0 - props) / np.array([n_a, n_b]))

        figure, axis = plt.subplots(figsize=(6.5, 4.5))
        labels = ["Control A", "Variant B"]
        axis.bar(labels, props, color=["#4C78A8", "#F58518"], alpha=0.85, yerr=errors, capsize=6)
        axis.set(ylabel="Conversion rate", title=title, ylim=(0, max(0.15, props.max() + 0.04)))
        axis.grid(alpha=0.25, axis="y")
        for index, rate in enumerate(props):
            axis.text(index, rate + errors[index] + 0.005, f"{rate:.1%}", ha="center")
        figure.tight_layout()
        path = self.output_dir / "ab_conversion_rates.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_chi_square_contributions(
        self,
        observed: np.ndarray,
        expected: np.ndarray,
        title: str = "Chi-square category contributions",
    ) -> Path:
        """Plot each category's (O − E)² / E term that sums to χ²."""
        obs = np.asarray(observed, dtype=float)
        exp = np.asarray(expected, dtype=float)
        if obs.shape != exp.shape or obs.size < 2:
            raise ValueError("observed and expected must be equal-length arrays with at least two categories.")
        contributions = (obs - exp) ** 2 / exp
        labels = [f"Cat {index + 1}" for index in range(obs.size)]

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.bar(labels, contributions, color="#E45756", alpha=0.85)
        axis.set(xlabel="Category", ylabel="(O − E)² / E", title=title)
        axis.grid(alpha=0.25, axis="y")
        figure.tight_layout()
        path = self.output_dir / "chi_square_contributions.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
