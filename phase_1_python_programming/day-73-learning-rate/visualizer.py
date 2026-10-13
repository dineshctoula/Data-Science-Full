"""Plots for the Day 73 learning-rate walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class RatePlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def train_curves(self, runs, title="train MSE, log scale so the blow-up fits"):
        runs = list(runs)
        if not runs:
            raise ValueError("no runs")
        colors = ["#336699", "#2a7a4f", "#cc5533"]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for run, color in zip(runs, colors):
            epochs = np.arange(1, run.epochs + 1)
            ax.plot(epochs, run.train_mse, color=color, label=f"lr {run.lr:g}")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        path = self.out_dir / "train_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def final_bars(self, runs, title="test MSE after the same 40 epochs"):
        runs = list(runs)
        if not runs:
            raise ValueError("no runs")
        labels = [f"lr {run.lr:g}" for run in runs]
        values = [float(run.test_mse[-1]) for run in runs]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.bar(labels, values, color=["#336699", "#2a7a4f", "#cc5533"][: len(runs)])
        ax.set_yscale("log")
        ax.set_ylabel("test MSE")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, which="both")
        fig.tight_layout()
        path = self.out_dir / "final_test.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
