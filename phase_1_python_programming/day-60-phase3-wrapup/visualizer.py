"""Plots for the Day 60 phase-3 wrap-up."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wrap_engine import fit_ols


class WrapPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def hours_vs_score(self, X, y, names, title="hours vs exam score"):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if "hours" not in names:
            raise ValueError("need an hours column")
        j = list(names).index("hours")
        hours = X[:, j]

        # hold the other columns at their means so the line is just the hours slope
        fit = fit_ols(X, y, names)
        grid = np.linspace(hours.min(), hours.max(), 40)
        baseline = np.tile(X.mean(axis=0), (len(grid), 1))
        baseline[:, j] = grid
        line = fit.predict(baseline)

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.scatter(hours, y, s=22, alpha=0.55, color="#666666", label="students")
        ax.plot(grid, line, color="#cc5533", linewidth=2, label="ols, other cols at mean")
        ax.set_xlabel("hours studied")
        ax.set_ylabel("exam score")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "hours_vs_score.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def residuals(self, y, pred, title="holdout residuals"):
        y = np.asarray(y, dtype=float).reshape(-1)
        pred = np.asarray(pred, dtype=float).reshape(-1)
        resid = y - pred
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(pred, resid, s=22, alpha=0.6, color="#336699")
        ax.axhline(0, color="#cc5533", linewidth=1.2)
        ax.set_xlabel("predicted score")
        ax.set_ylabel("residual")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "residuals.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def slope_hist(self, samples, lo, hi, title="bootstrap: hours coefficient"):
        samples = np.asarray(samples, dtype=float).reshape(-1)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(samples, bins=18, color="#336699", edgecolor="white")
        ax.axvline(lo, color="#cc5533", linestyle="--", label=f"2.5% = {lo:.2f}")
        ax.axvline(hi, color="#cc5533", linestyle="--", label=f"97.5% = {hi:.2f}")
        ax.set_xlabel("hours coefficient")
        ax.set_ylabel("bootstrap samples")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        path = self.out_dir / "hours_bootstrap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
