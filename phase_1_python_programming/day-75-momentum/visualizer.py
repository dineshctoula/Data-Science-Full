"""Plots for the Day 75 momentum walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class MomPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def loss_curves(self, runs, title="bowl loss, three amounts of momentum"):
        runs = list(runs)
        if not runs:
            raise ValueError("no runs")
        colors = ["#336699", "#2a7a4f", "#cc5533"]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for run, color in zip(runs, colors):
            steps = np.arange(1, run.steps + 1)
            ax.plot(steps, run.losses, color=color, label=f"beta {run.beta:g}")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        path = self.out_dir / "loss_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def flat_axis(self, runs, title="the flat weight is the one that still has to move"):
        runs = list(runs)
        if not runs:
            raise ValueError("no runs")
        colors = ["#336699", "#2a7a4f", "#cc5533"]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for run, color in zip(runs, colors):
            steps = np.arange(1, run.steps + 1)
            ax.plot(steps, run.weights[:, 0], color=color, label=f"beta {run.beta:g}")
        ax.axhline(0, color="#111111", linewidth=1.0)
        ax.set_xlabel("step")
        ax.set_ylabel("flat-axis weight")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "flat_axis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
