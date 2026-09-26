"""Plots for the Day 54 k-means walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kmeans_engine import KMeans


class KMeansPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def clusters(self, model, X, title="k-means clusters", filename="kmeans_clusters.png"):
        if not isinstance(model, KMeans) or model.centers_ is None:
            raise ValueError("need a fitted KMeans")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != 2:
            raise ValueError("cluster scatter wants 2D data")

        labels = model.predict(X)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=28, edgecolors="k", linewidths=0.3)
        ax.scatter(
            model.centers_[:, 0],
            model.centers_[:, 1],
            marker="X",
            s=180,
            c="black",
            label="centers",
            zorder=5,
        )
        ax.set_title(f"{title}  (k={model.k}, inertia={model.inertia_:.1f})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = self.out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def elbow(self, scores: dict, title="elbow: inertia vs k"):
        if not scores:
            raise ValueError("empty scores")
        ks = sorted(scores)
        vals = [scores[k] for k in ks]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ks, vals, "o-", color="#336699", linewidth=2)
        ax.set_xlabel("k")
        ax.set_ylabel("inertia")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "elbow.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def inertia_curve(self, curve, title="inertia per iteration"):
        if not curve:
            raise ValueError("empty curve")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(range(1, len(curve) + 1), curve, color="#cc5533", linewidth=2)
        ax.set_xlabel("iteration")
        ax.set_ylabel("inertia")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "inertia_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def truth_vs_pred(self, X, y_true, y_pred, title="truth vs k-means labels"):
        """Side-by-side — labels won't match numerically, just the grouping."""
        X = np.asarray(X, dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", s=28, edgecolors="k", linewidths=0.3)
        axes[0].set_title("ground truth")
        axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap="tab10", s=28, edgecolors="k", linewidths=0.3)
        axes[1].set_title("k-means labels")
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.suptitle(title)
        fig.tight_layout()
        path = self.out_dir / "truth_vs_pred.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
