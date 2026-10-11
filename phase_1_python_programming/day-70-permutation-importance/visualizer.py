"""Plots for the Day 70 permutation walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class PermPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def rise_bars(self, rises, title="validation MSE rise when a column is shuffled"):
        if not rises:
            raise ValueError("no columns")
        labels = [f"x{item.index}" for item in rises]
        means = np.array([item.mean_rise for item in rises], dtype=float)
        stds = np.array([item.std_rise for item in rises], dtype=float)
        fig, ax = plt.subplots(figsize=(6.6, 4.3))
        ax.bar(labels, means, yerr=stds, color="#336699", capsize=4)
        ax.set_ylabel("MSE rise")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def coef_bars(self, coef, title="fitted coefficients, intercept left off"):
        coef = np.asarray(coef, dtype=float).reshape(-1)
        if len(coef) < 2:
            raise ValueError("need an intercept and at least one slope")
        slopes = coef[1:]
        labels = [f"x{i}" for i in range(len(slopes))]
        fig, ax = plt.subplots(figsize=(6.6, 4.3))
        colors = ["#cc5533" if value < 0 else "#2a7a4f" for value in slopes]
        ax.bar(labels, slopes, color=colors)
        ax.axhline(0, color="#111111", linewidth=1.0)
        ax.set_ylabel("coefficient")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "coefficients.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
