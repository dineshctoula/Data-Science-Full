"""Plots for the Day 61 train / validate comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class WorkflowPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def accuracy_bars(self, scores, title="train vs validation accuracy"):
        if not scores:
            raise ValueError("no scores")
        names = [s.name for s in scores]
        train = [s.train_acc for s in scores]
        val = [s.val_acc for s in scores]
        x = np.arange(len(names))
        width = 0.36

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(x - width / 2, train, width, label="train", color="#888888")
        ax.bar(x + width / 2, val, width, label="validation", color="#336699")
        ax.set_xticks(x, names)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "train_vs_val.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def visits_scatter(self, X, y, names, title="visits vs signup"):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if "visits" not in names:
            raise ValueError("need a visits column")
        j = list(names).index("visits")
        spend_j = list(names).index("spend") if "spend" in names else None

        fig, ax = plt.subplots(figsize=(7, 4.8))
        if spend_j is None:
            ax.scatter(X[:, j], y, c=y, cmap="coolwarm", s=28, edgecolors="k", linewidths=0.3)
            ax.set_ylabel("signed up")
        else:
            ax.scatter(X[:, j], X[:, spend_j], c=y, cmap="coolwarm", s=28, edgecolors="k", linewidths=0.3)
            ax.set_ylabel("spend")
        ax.set_xlabel("visits")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "visits_scatter.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
