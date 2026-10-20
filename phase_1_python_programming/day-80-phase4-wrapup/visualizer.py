"""Plots for the Day 80 phase-4 wrap-up."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class WrapPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        fig.tight_layout()
        path = self.out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def loss_curves(self, history, title="train vs validation log loss"):
        """Both curves per epoch, with the kept epoch marked."""
        if not history.val_loss:
            raise ValueError("empty history")
        epochs = np.arange(1, len(history.val_loss) + 1)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(epochs, history.train_loss, color="#336699", linewidth=1.6, label="train")
        ax.plot(epochs, history.val_loss, color="#2a7a4f", linewidth=1.6, label="validation")
        # the epoch whose weights the model actually keeps
        ax.axvline(history.best_epoch, color="#cc5533", linestyle="--", label=f"kept epoch {history.best_epoch}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("log loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        return self._save(fig, "loss_curves.png")

    def decay_grid(self, trials, picked, title="validation picks the decay"):
        if not trials:
            raise ValueError("no trials")
        labels = [f"{t.decay:g}" for t in trials]
        x = np.arange(len(trials))
        width = 0.38
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(x - width / 2, [t.train_loss for t in trials], width, color="#336699", label="train")
        val_colors = ["#cc5533" if t.decay == picked.decay else "#2a7a4f" for t in trials]
        ax.bar(x + width / 2, [t.val_loss for t in trials], width, color=val_colors, label="validation")
        ax.set_xticks(x, labels)
        ax.set_xlabel("weight decay")
        ax.set_ylabel("log loss at the kept epoch")
        # the losses bunch together, so start the axis near the lowest bar to see the gaps
        low = min(min(t.train_loss, t.val_loss) for t in trials)
        ax.set_ylim(low * 0.95, None)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        return self._save(fig, "decay_grid.png")

    def importance(self, rises, title="permutation importance on validation"):
        """rises: {column: rise in log loss when that column is shuffled}."""
        if not rises:
            raise ValueError("no columns")
        ranked = sorted(rises.items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.barh([name for name, _ in ranked], [rise for _, rise in ranked], color="#336699")
        ax.axvline(0, color="#111111", linewidth=0.8)
        ax.set_xlabel("rise in validation log loss")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        return self._save(fig, "importance.png")
