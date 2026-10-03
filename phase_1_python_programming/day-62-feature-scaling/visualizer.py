"""Plots for the Day 62 scaling lesson."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ScalePlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def before_after(self, raw, scaled, names, title="columns before and after scaling"):
        raw = np.asarray(raw, dtype=float)
        scaled = np.asarray(scaled, dtype=float)
        if raw.shape != scaled.shape or raw.shape[1] != len(names):
            raise ValueError("raw, scaled, and names don't match")

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
        for ax, data, label in (
            (axes[0], raw, "raw"),
            (axes[1], scaled, "standard scaled"),
        ):
            ax.boxplot([data[:, j] for j in range(data.shape[1])], tick_labels=list(names))
            ax.set_title(label)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(title)
        fig.tight_layout()
        path = self.out_dir / "before_after.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def accuracy_bars(self, rows, title="kNN validation accuracy"):
        if not rows:
            raise ValueError("no rows")
        names = [r.name for r in rows]
        vals = [r.val_acc for r in rows]
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.bar(names, vals, color=["#888888", "#336699", "#cc5533"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("validation accuracy")
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "knn_by_scale.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
