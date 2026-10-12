"""Plots for the Day 72 early-stopping walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class StopPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def loss_curves(self, train_loss, val_loss, best_epoch, title="validation loss turns up"):
        train_loss = np.asarray(train_loss, dtype=float).reshape(-1)
        val_loss = np.asarray(val_loss, dtype=float).reshape(-1)
        if len(train_loss) == 0 or len(train_loss) != len(val_loss):
            raise ValueError("train and val losses should match")
        if not 1 <= best_epoch <= len(train_loss):
            raise ValueError("best_epoch is outside the run")
        epochs = np.arange(1, len(train_loss) + 1)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(epochs, train_loss, color="#336699", label="train")
        ax.plot(epochs, val_loss, color="#cc5533", label="validation")
        ax.axvline(best_epoch, color="#111111", linestyle="--", linewidth=1.1, label=f"best epoch {best_epoch}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("log loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "loss_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def test_bars(self, best_loss, final_loss, title="test log loss, stopped vs last epoch"):
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.bar(["stopped", "last epoch"], [best_loss, final_loss], color=["#2a7a4f", "#cc5533"])
        ax.set_ylabel("test log loss")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "stopped_vs_final.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
