"""Plots for the Day 76 Adam walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class AdamPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def loss_curves(self, runs, title="bowl loss for GD, momentum, and Adam"):
        runs = list(runs)
        if not runs:
            raise ValueError("no runs")
        colors = ["#336699", "#2a7a4f", "#cc5533"]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for run, color in zip(runs, colors):
            steps = np.arange(1, run.steps + 1)
            ax.plot(steps, run.losses, color=color, label=run.name)
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

    def first_step_bars(self, runs, title="size of the first step on each axis"):
        runs = list(runs)
        if not runs:
            raise ValueError("no runs")
        labels = [run.name for run in runs]
        x = np.arange(len(runs))
        width = 0.36
        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        ax.bar(x - width / 2, [abs(run.first_step[0]) for run in runs], width, color="#336699", label="flat axis")
        ax.bar(x + width / 2, [abs(run.first_step[1]) for run in runs], width, color="#cc5533", label="steep axis")
        ax.set_xticks(x, labels)
        ax.set_ylabel("|first step|")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "first_steps.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
