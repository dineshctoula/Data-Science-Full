"""Plots for the Day 59 bias-variance walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class BiasVariancePlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def tradeoff(self, points, title="bias², variance, and mse vs degree"):
        if not points:
            raise ValueError("no points")
        degrees = [p.degree for p in points]
        bias = [p.bias2 for p in points]
        var = [p.variance for p in points]
        mse = [p.mse for p in points]

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(degrees, bias, "o-", color="#cc5533", label="bias²")
        ax.plot(degrees, var, "s-", color="#336699", label="variance")
        ax.plot(degrees, mse, "D--", color="#222222", label="mse ≈ bias²+var+noise")
        ax.set_xlabel("polynomial degree")
        ax.set_ylabel("error")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "tradeoff.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def spaghetti(self, grid, truth, curves, title="fits from different samples", filename="spaghetti.png"):
        grid = np.asarray(grid, dtype=float)
        truth = np.asarray(truth, dtype=float)
        curves = np.asarray(curves, dtype=float)
        if curves.ndim != 2 or curves.shape[1] != len(grid):
            raise ValueError("curves should be (n_fits, n_grid)")

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for row in curves:
            ax.plot(grid, row, color="#336699", alpha=0.35, linewidth=1)
        ax.plot(grid, truth, color="#cc5533", linewidth=2.2, label="truth")
        ax.plot(grid, curves.mean(axis=0), color="#111111", linewidth=1.6, linestyle="--", label="mean fit")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
