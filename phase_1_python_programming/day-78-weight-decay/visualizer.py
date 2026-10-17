"""Plots for the Day 78 weight-decay walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class DecayPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def mse_lines(self, trials, title="train, validation, and test error by decay"):
        if not trials:
            raise ValueError("no trials")
        # decay 0 can't sit on a log axis, so plot by position and label the ticks
        x = np.arange(len(trials))
        labels = [f"{trial.decay:g}" for trial in trials]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(x, [t.train_mse for t in trials], marker="o", color="#336699", label="train")
        ax.plot(x, [t.val_mse for t in trials], marker="o", color="#2a7a4f", label="validation")
        ax.plot(x, [t.test_mse for t in trials], marker="o", color="#cc5533", label="test")
        ax.set_xticks(x, labels)
        ax.set_xlabel("weight decay")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "decay_grid.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def weight_shrink(self, trials, title="real slopes and junk weights both shrink"):
        if not trials:
            raise ValueError("no trials")
        x = np.arange(len(trials))
        labels = [f"{trial.decay:g}" for trial in trials]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(x, [t.x0 for t in trials], marker="o", color="#2a7a4f", label="x0 slope (true 2.0)")
        ax.plot(x, [abs(t.x1) for t in trials], marker="o", color="#336699", label="|x1| slope (true 1.2)")
        ax.plot(x, [t.noise_norm for t in trials], marker="o", color="#cc5533", label="size of the 40 junk weights")
        ax.set_xticks(x, labels)
        ax.set_xlabel("weight decay")
        ax.set_ylabel("weight size")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "weights.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
