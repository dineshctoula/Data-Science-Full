"""Plots for the Day 57 cross-validation walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class CVPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def fold_bars(self, scores, title="accuracy per fold"):
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(scores) == 0:
            raise ValueError("no fold scores")
        labels = [f"fold {i+1}" for i in range(len(scores))]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(labels, scores, color="#336699")
        ax.axhline(scores.mean(), color="#cc5533", linestyle="--", label=f"mean {scores.mean():.3f}")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "fold_bars.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def holdout_vs_cv(self, holdouts, cv_scores, title="one split wobbles, CV less so"):
        holdouts = np.asarray(holdouts, dtype=float).reshape(-1)
        cv_scores = np.asarray(cv_scores, dtype=float).reshape(-1)
        if len(holdouts) == 0 or len(cv_scores) == 0:
            raise ValueError("need both score lists")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(range(1, len(holdouts) + 1), holdouts, "o--", color="#888888", label="random holdouts")
        # cv is one number repeated as a band, plus the fold dots on the side
        ax.axhline(cv_scores.mean(), color="#336699", linewidth=2, label=f"cv mean {cv_scores.mean():.3f}")
        ax.fill_between(
            [0.5, len(holdouts) + 0.5],
            cv_scores.mean() - cv_scores.std(ddof=1),
            cv_scores.mean() + cv_scores.std(ddof=1),
            color="#336699",
            alpha=0.15,
            label="cv ± 1 std",
        )
        ax.set_xlim(0.5, len(holdouts) + 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("holdout repeat")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "holdout_vs_cv.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def k_sweep(self, ks, means, stds, title="mean accuracy vs k"):
        ks = list(ks)
        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)
        if not (len(ks) == len(means) == len(stds)):
            raise ValueError("ks / means / stds length mismatch")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(ks, means, yerr=stds, fmt="o-", color="#336699", capsize=4)
        ax.set_xlabel("k")
        ax.set_ylabel("cv accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "k_sweep.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
