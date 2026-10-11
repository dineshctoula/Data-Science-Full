"""Plots for the Day 71 multiclass walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class MultiPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def confusion_heatmap(self, matrix, title="test rows: actual down, guess across"):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("confusion matrix should be square")
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046)
        n = matrix.shape[0]
        ax.set_xticks(range(n), [str(i) for i in range(n)])
        ax.set_yticks(range(n), [str(i) for i in range(n)])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="#111111")
        fig.tight_layout()
        path = self.out_dir / "confusion.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def recall_bars(self, recalls, title="per-class recall on the test slice"):
        recalls = np.asarray(recalls, dtype=float).reshape(-1)
        if len(recalls) == 0:
            raise ValueError("no recalls")
        labels = [f"class {i}" for i in range(len(recalls))]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.bar(labels, recalls, color="#336699")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("recall")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "recall.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
