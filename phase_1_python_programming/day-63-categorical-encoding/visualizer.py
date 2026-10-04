"""Plots for the Day 63 encoding walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class EncodePlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def mean_bars(self, labels, scores, title="mean score by category", filename="means.png"):
        labels = np.asarray(labels).reshape(-1)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(labels) != len(scores):
            raise ValueError("labels and scores length mismatch")
        cats = []
        means = []
        for cat in sorted(set(labels.tolist()), key=str):
            cats.append(str(cat))
            means.append(float(scores[labels == cat].mean()))

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.bar(cats, means, color="#336699")
        ax.set_ylabel("mean score")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def mse_bars(self, rows, title="validation MSE by encoding"):
        if not rows:
            raise ValueError("no rows")
        names = [r.name for r in rows]
        vals = [r.mse for r in rows]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(names, vals, color=["#336699", "#888888", "#cc5533"])
        ax.set_ylabel("validation MSE")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "encoding_mse.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
