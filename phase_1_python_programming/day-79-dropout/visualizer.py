"""Plots for the Day 79 dropout walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class DropPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def fits(self, x_train, y_train, curves, title="what the network draws through 16 points"):
        """curves: list of (label, x_grid, y_pred). The true sine goes underneath."""
        if not curves:
            raise ValueError("no curves")
        grid = np.linspace(-2.0, 2.0, 300)
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.plot(grid, np.sin(1.6 * grid), color="#999999", linestyle="--", linewidth=1.2, label="true curve")
        colors = ["#cc5533", "#2a7a4f", "#336699"]
        for (label, xs, ys), color in zip(curves, colors):
            ax.plot(xs, ys, color=color, linewidth=1.6, label=label)
        ax.scatter(x_train, y_train, color="#111111", s=22, zorder=3, label="training points")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "fits.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def mse_bars(self, trials, title="mean error over five starting seeds"):
        if not trials:
            raise ValueError("no trials")
        labels = [f"{trial.drop:g}" for trial in trials]
        x = np.arange(len(trials))
        width = 0.27
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(x - width, [t.train_mse for t in trials], width, color="#336699", label="train")
        ax.bar(x, [t.val_mse for t in trials], width, color="#2a7a4f", label="validation")
        ax.bar(x + width, [t.test_mse for t in trials], width, color="#cc5533", label="test")
        ax.set_xticks(x, labels)
        ax.set_xlabel("dropout rate")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "dropout_grid.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
