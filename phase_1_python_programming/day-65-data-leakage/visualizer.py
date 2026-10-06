"""Plots for the Day 65 leakage walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class LeakPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def gap_hist(self, gaps, title="leaky MSE looks better by this much"):
        gaps = np.asarray(gaps, dtype=float).reshape(-1)
        if len(gaps) == 0:
            raise ValueError("no gaps")
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.hist(gaps, bins=10, color="#336699", edgecolor="white")
        ax.axvline(0, color="#cc5533", linewidth=1.4, label="no leak")
        ax.axvline(float(gaps.mean()), color="#111111", linestyle="--", label=f"mean {gaps.mean():.2f}")
        ax.set_xlabel("honest MSE − leaky MSE")
        ax.set_ylabel("seeds")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        path = self.out_dir / "optimism_gap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def one_seed_bars(self, honest, leaky, title="one split"):
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.bar(["honest", "leaky"], [honest, leaky], color=["#336699", "#cc5533"])
        ax.set_ylabel("test MSE")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "one_split.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
