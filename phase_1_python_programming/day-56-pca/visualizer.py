"""Plots for the Day 56 PCA walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pca_engine import PCA


class PCAPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def stretched_cloud(self, model, X, title="PC axes on stretched cloud", filename="pca_axes.png"):
        if not isinstance(model, PCA) or model.components_ is None:
            raise ValueError("need a fitted PCA")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != 2:
            raise ValueError("axis sketch wants 2D data")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(X[:, 0], X[:, 1], s=18, alpha=0.55, color="#666666")
        origin = model.mean_
        # draw PC1 / PC2 scaled by sqrt(variance) so length hints at importance
        scale = np.sqrt(model.explained_variance_[:2])
        colors = ["#cc5533", "#336699"]
        for j in range(2):
            direction = model.components_[j] * scale[j] * 2.2
            ax.annotate(
                "",
                xy=origin + direction,
                xytext=origin,
                arrowprops=dict(arrowstyle="->", color=colors[j], lw=2),
            )
            ax.plot([], [], color=colors[j], label=f"PC{j+1}")
        ax.scatter([origin[0]], [origin[1]], c="black", s=40, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = self.out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def scree(self, ratios, title="scree: variance per component"):
        ratios = np.asarray(ratios, dtype=float).reshape(-1)
        if len(ratios) == 0:
            raise ValueError("empty ratios")
        idx = np.arange(1, len(ratios) + 1)
        cum = np.cumsum(ratios)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(idx, ratios, color="#336699", label="share")
        ax.plot(idx, cum, "o-", color="#cc5533", label="cumulative")
        ax.set_xlabel("component")
        ax.set_ylabel("explained variance ratio")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "scree.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def recon_scatter(self, X, X_hat, title="original vs reconstruction"):
        X = np.asarray(X, dtype=float)
        X_hat = np.asarray(X_hat, dtype=float)
        if X.shape[1] < 2:
            raise ValueError("need at least 2 features to scatter")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(X[:, 0], X[:, 1], s=18, alpha=0.45, label="original", color="#888888")
        ax.scatter(X_hat[:, 0], X_hat[:, 1], s=18, alpha=0.7, label="reconstructed", color="#cc5533")
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = self.out_dir / "reconstruction.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
