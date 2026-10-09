"""Plots for the Day 68 search walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt


class SearchPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def mse_lines(self, trials, title="which k the three slices like"):
        if not trials:
            raise ValueError("no trials")
        ks = [trial.k for trial in trials]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(ks, [trial.train_mse for trial in trials], marker="o", color="#336699", label="train")
        ax.plot(ks, [trial.val_mse for trial in trials], marker="o", color="#2a7a4f", label="validation")
        ax.plot(ks, [trial.test_mse for trial in trials], marker="o", color="#cc5533", label="test")
        ax.set_xlabel("k")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "k_grid.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def pick_bars(self, labels, test_mses, title="test error of three ways to pick k"):
        labels = list(labels)
        if len(labels) == 0 or len(labels) != len(test_mses):
            raise ValueError("labels and scores should match")
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar(labels, test_mses, color=["#336699", "#2a7a4f", "#cc5533"])
        ax.set_ylabel("test MSE")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "picks.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
