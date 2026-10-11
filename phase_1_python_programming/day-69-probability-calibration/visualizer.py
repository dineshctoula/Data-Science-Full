"""Plots for the Day 69 calibration walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class CalPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def reliability(self, scores, title="predicted probability vs how often it happened"):
        if not scores:
            raise ValueError("no scores")
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        ax.plot([0, 1], [0, 1], color="#111111", linewidth=1.0, label="matches")
        colors = ["#336699", "#cc5533", "#2a7a4f"]
        for score, color in zip(scores, colors):
            if not score.bins:
                continue
            xs = [item.mean_prob for item in score.bins]
            ys = [item.freq for item in score.bins]
            ax.plot(xs, ys, marker="o", color=color, label=score.name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("fraction of positives")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "reliability.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def brier_bars(self, names, scores, title="Brier score, lower is better"):
        names = list(names)
        scores = np.asarray(scores, dtype=float)
        if len(names) == 0 or len(names) != len(scores):
            raise ValueError("names and scores should match")
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.bar(names, scores, color=["#336699", "#cc5533", "#2a7a4f"][: len(names)])
        ax.set_ylabel("Brier")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "brier.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
