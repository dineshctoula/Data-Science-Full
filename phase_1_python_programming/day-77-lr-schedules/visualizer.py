"""Plots for the Day 77 schedule walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class SchedPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def lr_curves(self, runs, title="how the learning rate moves"):
        runs = list(runs)
        if not runs:
            raise ValueError("no runs")
        colors = ["#336699", "#2a7a4f", "#cc5533"]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for run, color in zip(runs, colors):
            epochs = np.arange(1, run.epochs + 1)
            ax.plot(epochs, run.lrs, color=color, label=run.name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "lr_schedules.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def test_curves(self, runs, title="test MSE when the rate stays high vs when it cools"):
        runs = list(runs)
        if not runs:
            raise ValueError("no runs")
        colors = ["#336699", "#2a7a4f", "#cc5533"]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for run, color in zip(runs, colors):
            epochs = np.arange(1, run.epochs + 1)
            ax.plot(epochs, run.test_mse, color=color, label=run.name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("test MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "test_mse.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
