"""Plots for the Day 66 imbalance walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class BalancePlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def metric_bars(self, names, accuracies, recalls, title="accuracy stays high when recall is zero"):
        names = list(names)
        accuracies = np.asarray(accuracies, dtype=float)
        recalls = np.asarray(recalls, dtype=float)
        if len(names) == 0 or len(names) != len(accuracies) or len(names) != len(recalls):
            raise ValueError("names, accuracies, and recalls should match")
        x = np.arange(len(names))
        width = 0.36
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(x - width / 2, accuracies, width, color="#336699", label="accuracy")
        ax.bar(x + width / 2, recalls, width, color="#cc5533", label="recall")
        ax.set_xticks(x, names)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "acc_vs_recall.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def catch_bars(self, names, true_pos, false_pos, title="catches vs false alarms"):
        names = list(names)
        true_pos = np.asarray(true_pos, dtype=float)
        false_pos = np.asarray(false_pos, dtype=float)
        if len(names) == 0 or len(names) != len(true_pos) or len(names) != len(false_pos):
            raise ValueError("names and counts should match")
        x = np.arange(len(names))
        width = 0.36
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(x - width / 2, true_pos, width, color="#2a7a4f", label="caught (tp)")
        ax.bar(x + width / 2, false_pos, width, color="#c4a35a", label="false alarms (fp)")
        ax.set_xticks(x, names)
        ax.set_ylabel("test rows")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "catches.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
