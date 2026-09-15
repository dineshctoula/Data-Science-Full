"""Matplotlib visualizations for Day 41 probability distributions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probability_engine import ProbabilityEngine


class DistributionVisualizer:
    """Create concise plots linking PMF/PDF formulas to sample histograms."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        # Create the folder once so every plot method can write without setup.
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_binomial_pmf(self, n: int, p: float, title: str = "Binomial PMF") -> Path:
        """Plot P(X = k) for every k in 0…n as a stem chart."""
        if n < 0:
            raise ValueError("n must be a non-negative integer.")
        # Evaluate the same formula used by the engine so the chart stays honest.
        ks = np.arange(0, n + 1)
        probs = np.array([ProbabilityEngine.binomial_pmf(int(k), n, p) for k in ks])

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.vlines(ks, 0, probs, color="#4C78A8", linewidth=2)
        axis.plot(ks, probs, "o", color="#4C78A8")
        axis.set(xlabel="Number of successes k", ylabel="P(X = k)", title=title)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25, axis="y")
        figure.tight_layout()
        path = self.output_dir / "binomial_pmf.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_normal_pdf_with_sample(
        self,
        samples: np.ndarray,
        mu: float = 0.0,
        sigma: float = 1.0,
        title: str = "Normal PDF vs sample",
    ) -> Path:
        """Overlay the analytic Normal density on a Monte Carlo histogram."""
        data = np.asarray(samples, dtype=float)
        if data.ndim != 1 or data.size < 2:
            raise ValueError("samples must be a one-dimensional array with at least two values.")
        if not np.isfinite(data).all():
            raise ValueError("samples must contain only finite numbers.")

        # Span a few standard deviations so both tails of the PDF are visible.
        xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        dens = np.array([ProbabilityEngine.normal_pdf(float(x), mu, sigma) for x in xs])

        figure, axis = plt.subplots(figsize=(7, 4.5))
        # density=True makes the histogram area comparable to the PDF curve.
        axis.hist(data, bins="auto", density=True, color="#4C78A8", alpha=0.35, edgecolor="white", label="Sample")
        axis.plot(xs, dens, color="#E45756", linewidth=2.5, label="Normal PDF")
        axis.axvline(mu, color="#54A24B", linestyle="--", linewidth=1.5, label=f"μ = {mu:g}")
        axis.set(xlabel="x", ylabel="Density", title=title)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "normal_pdf_sample.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_poisson_pmf(self, lam: float, k_max: int = 15, title: str = "Poisson PMF") -> Path:
        """Plot Poisson probabilities up to ``k_max`` on a stem chart."""
        if k_max < 0:
            raise ValueError("k_max must be a non-negative integer.")
        ks = np.arange(0, k_max + 1)
        probs = np.array([ProbabilityEngine.poisson_pmf(int(k), lam) for k in ks])

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.vlines(ks, 0, probs, color="#F58518", linewidth=2)
        axis.plot(ks, probs, "o", color="#F58518")
        # Mark the mean so the peak-vs-rate relationship is easy to see.
        axis.axvline(lam, color="#54A24B", linestyle="--", linewidth=1.5, label=f"λ = {lam:g}")
        axis.set(xlabel="Count k", ylabel="P(X = k)", title=title)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25, axis="y")
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "poisson_pmf.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_exponential_pdf_cdf(self, lam: float, x_max: float = 5.0, title: str = "Exponential PDF and CDF") -> Path:
        """Plot the Exponential waiting-time density beside its cumulative curve."""
        if x_max <= 0:
            raise ValueError("x_max must be positive.")
        xs = np.linspace(0.0, x_max, 300)
        pdf = np.array([ProbabilityEngine.exponential_pdf(float(x), lam) for x in xs])
        cdf = np.array([ProbabilityEngine.exponential_cdf(float(x), lam) for x in xs])

        figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].plot(xs, pdf, color="#4C78A8", linewidth=2.5)
        axes[0].set(xlabel="Waiting time x", ylabel="f(x)", title="PDF")
        axes[0].grid(alpha=0.25)
        # The CDF starts at 0 and approaches 1; marking 1 − e^(−1) ≈ 0.63 at x = 1/λ.
        axes[1].plot(xs, cdf, color="#E45756", linewidth=2.5)
        axes[1].axhline(1.0 - np.exp(-1.0), color="#54A24B", linestyle="--", linewidth=1, label="CDF at mean")
        axes[1].axvline(1.0 / lam, color="#54A24B", linestyle="--", linewidth=1)
        axes[1].set(xlabel="Waiting time x", ylabel="F(x)", title="CDF")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        figure.suptitle(title, fontsize=13, fontweight="bold")
        figure.tight_layout()
        path = self.output_dir / "exponential_pdf_cdf.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
