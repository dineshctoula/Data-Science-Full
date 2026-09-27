"""Plots for the Day 55 DBSCAN walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dbscan_engine import NOISE, DBSCAN


class DBSCANPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def clusters(self, model, X, title="DBSCAN clusters", filename="dbscan_clusters.png"):
        if not isinstance(model, DBSCAN) or model.labels_ is None:
            raise ValueError("need a fitted DBSCAN")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != 2:
            raise ValueError("scatter wants 2D data")

        labels = model.labels_
        fig, ax = plt.subplots(figsize=(7, 5.5))
        # noise in gray so the real clusters pop
        noise = labels == NOISE
        if noise.any():
            ax.scatter(
                X[noise, 0], X[noise, 1],
                c="#bbbbbb", s=28, label="noise", zorder=1,
            )
        clustered = ~noise
        if clustered.any():
            ax.scatter(
                X[clustered, 0], X[clustered, 1],
                c=labels[clustered], cmap="tab10",
                s=32, edgecolors="k", linewidths=0.3, zorder=2,
            )
        if model.core_mask_ is not None and model.core_mask_.any():
            cores = model.core_mask_
            ax.scatter(
                X[cores, 0], X[cores, 1],
                facecolors="none", edgecolors="black", s=90,
                linewidths=0.8, label="core", zorder=3,
            )
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = self.out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def eps_sweep(self, rows, title="clusters vs eps"):
        """rows: list of (eps, n_clusters, n_noise)."""
        if not rows:
            raise ValueError("empty sweep")
        eps = [r[0] for r in rows]
        n_cl = [r[1] for r in rows]
        n_noise = [r[2] for r in rows]

        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(eps, n_cl, "o-", color="#336699", label="n clusters")
        ax1.set_xlabel("eps")
        ax1.set_ylabel("clusters", color="#336699")
        ax2 = ax1.twinx()
        ax2.plot(eps, n_noise, "s--", color="#cc5533", label="noise points")
        ax2.set_ylabel("noise", color="#cc5533")
        ax1.set_title(title)
        ax1.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "eps_sweep.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
