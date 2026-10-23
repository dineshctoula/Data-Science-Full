"""Plots for the Day 82 first churn model.

The gains curve is Day 81's KickoffPlots.gains; these are the new ones.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ModelPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        fig.tight_layout()
        path = self.out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def calibration(self, y, p, n_bins: int = 10, title="validation: predicted vs actual churn"):
        """Sort by predicted probability, cut into equal-count bins, compare means."""
        y = np.asarray(y, dtype=float).reshape(-1)
        p = np.asarray(p, dtype=float).reshape(-1)
        if len(y) != len(p) or len(y) < n_bins:
            raise ValueError("need at least one row per bin")
        # equal-count bins so every point rests on about the same number of customers
        order = np.argsort(p, kind="stable")
        bins = np.array_split(order, n_bins)
        pred = [p[b].mean() for b in bins]
        actual = [y[b].mean() for b in bins]
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.0, label="perfect calibration")
        ax.plot(pred, actual, marker="o", color="#336699", linewidth=1.6, label=f"{n_bins} bins of ~{len(y) // n_bins} customers")
        ax.set_xlabel("mean predicted churn")
        ax.set_ylabel("actual churn rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)
        return self._save(fig, "calibration.png")

    def coefficients(self, names, coef, top: int = 14, title="largest coefficients (per one std of each column)"):
        names = list(names)
        coef = np.asarray(coef, dtype=float).reshape(-1)
        if len(names) != len(coef) or not names:
            raise ValueError("names and coef should match and be non-empty")
        keep = np.argsort(-np.abs(coef), kind="stable")[:top][::-1]
        colors = ["#cc5533" if coef[j] > 0 else "#336699" for j in keep]
        fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(keep) + 1.4))
        ax.barh([names[j] for j in keep], coef[keep], color=colors)
        ax.axvline(0, color="#111111", linewidth=0.8)
        ax.set_xlabel("log-odds change  (red raises churn, blue lowers it)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        return self._save(fig, "coefficients.png")

    def bootstrap_diff(self, samples, lo, hi, label="top-20% recall: model minus baseline"):
        samples = np.asarray(samples, dtype=float).reshape(-1)
        if samples.size == 0:
            raise ValueError("no samples")
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.hist(samples, bins=40, color="#336699", edgecolor="white")
        ax.axvline(0, color="#111111", linewidth=1.0, label="no difference")
        ax.axvline(lo, color="#cc5533", linestyle="--", label=f"2.5% = {lo:+.3f}")
        ax.axvline(hi, color="#cc5533", linestyle="--", label=f"97.5% = {hi:+.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("bootstrap resamples")
        ax.set_title("paired bootstrap on the validation rows")
        ax.legend(fontsize=8)
        return self._save(fig, "bootstrap_diff.png")
