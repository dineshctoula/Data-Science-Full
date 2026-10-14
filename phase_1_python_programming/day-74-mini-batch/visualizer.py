"""Plots for the Day 74 mini-batch walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class BatchPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def test_curves(self, runs, title="test MSE after each pass through the training rows"):
        runs = list(runs)
        if not runs:
            raise ValueError("no runs")
        colors = ["#336699", "#2a7a4f", "#cc5533"]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for run, color in zip(runs, colors):
            epochs = np.arange(1, run.epochs + 1)
            label = f"batch {run.batch_size} ({run.updates_per_epoch}/epoch)"
            ax.plot(epochs, run.test_mse, color=color, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("test MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "test_by_epoch.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def end_bars(self, runs, title="test MSE at epoch 1 and epoch 20"):
        runs = list(runs)
        if not runs:
            raise ValueError("no runs")
        labels = [f"batch {run.batch_size}" for run in runs]
        x = np.arange(len(runs))
        width = 0.36
        fig, ax = plt.subplots(figsize=(6.6, 4.3))
        ax.bar(x - width / 2, [run.test_mse[0] for run in runs], width, color="#336699", label="epoch 1")
        ax.bar(x + width / 2, [run.test_mse[-1] for run in runs], width, color="#c4a35a", label="epoch 20")
        ax.set_xticks(x, labels)
        ax.set_ylabel("test MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "epoch_ends.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
