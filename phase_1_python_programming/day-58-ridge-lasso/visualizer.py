"""Plots for the Day 58 ridge / lasso walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class RegPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def coef_path(self, lams, paths, names, title="coefficient path", filename="coef_path.png"):
        """paths: (n_lams, n_features)."""
        lams = np.asarray(lams, dtype=float)
        paths = np.asarray(paths, dtype=float)
        if paths.ndim != 2 or len(lams) != len(paths):
            raise ValueError("lams and path rows don't match")
        if len(names) != paths.shape[1]:
            raise ValueError("need one name per feature")

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for j, name in enumerate(names):
            ax.plot(lams, paths[:, j], label=name, linewidth=1.8)
        ax.set_xscale("log")
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_xlabel("lambda")
        ax.set_ylabel("coefficient")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def mse_vs_lambda(self, lams, ridge_mse, lasso_mse, title="test MSE vs lambda"):
        lams = np.asarray(lams, dtype=float)
        if not (len(lams) == len(ridge_mse) == len(lasso_mse)):
            raise ValueError("length mismatch")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(lams, ridge_mse, "o-", color="#336699", label="ridge")
        ax.plot(lams, lasso_mse, "s--", color="#cc5533", label="lasso")
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel("test MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "mse_vs_lambda.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def coef_bars(self, names, ols, ridge, lasso, title="coefs at one lambda"):
        names = list(names)
        ols = np.asarray(ols, dtype=float)
        ridge = np.asarray(ridge, dtype=float)
        lasso = np.asarray(lasso, dtype=float)
        if not (len(names) == len(ols) == len(ridge) == len(lasso)):
            raise ValueError("coef length mismatch")

        x = np.arange(len(names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.bar(x - width, ols, width, label="ols", color="#888888")
        ax.bar(x, ridge, width, label="ridge", color="#336699")
        ax.bar(x + width, lasso, width, label="lasso", color="#cc5533")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x, names, rotation=20)
        ax.set_ylabel("coefficient")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "coef_bars.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
