"""Plots for the Day 51 random forest walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forest_engine import RandomForestClassifier


class ForestPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def decision_map(self, model, X, y, title="forest regions", filename="forest_regions.png"):
        if not isinstance(model, RandomForestClassifier):
            raise ValueError("need a fitted RandomForestClassifier")
        if not model.trees_:
            raise RuntimeError("model isn't fitted")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if X.shape[1] != 2:
            raise ValueError("decision map wants 2D features")

        pad = 0.4
        x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
        y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
        # 160^2 is plenty; forests are slower than a single tree
        xx, yy = np.meshgrid(np.linspace(x0, x1, 160), np.linspace(y0, y1, 160))
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.contourf(xx, yy, zz, alpha=0.35, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=28)
        ax.set_title(f"{title}  ({model.n_trees} trees)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = self.out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def n_trees_curve(self, scores: dict, title="test accuracy vs n_trees"):
        if not scores:
            raise ValueError("scores dict is empty")
        ns = sorted(scores)
        vals = [scores[n] for n in ns]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ns, vals, "o-", color="#336699", linewidth=2)
        ax.set_xlabel("number of trees")
        ax.set_ylabel("test accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "n_trees_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def single_vs_forest(self, X, y, single_pred, forest_pred, title="one tree vs forest"):
        """Side-by-side scatter colored by each model's mistakes — quick gut check."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        single_ok = single_pred == y
        forest_ok = forest_pred == y

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, ok, name in (
            (axes[0], single_ok, "single tree"),
            (axes[1], forest_ok, "forest"),
        ):
            ax.scatter(X[ok, 0], X[ok, 1], c=y[ok], cmap="coolwarm", s=28, edgecolors="k", alpha=0.85)
            # mark mistakes with X
            bad = ~ok
            if bad.any():
                ax.scatter(
                    X[bad, 0],
                    X[bad, 1],
                    marker="x",
                    c="black",
                    s=55,
                    label=f"miss ({bad.sum()})",
                )
                ax.legend(loc="best", fontsize=8)
            ax.set_title(f"{name}")
            ax.grid(alpha=0.2)
        fig.suptitle(title)
        fig.tight_layout()
        path = self.out_dir / "single_vs_forest.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def confusion_simple(self, y_true, y_pred, title="Confusion (counts)"):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n = len(classes)
        mat = np.zeros((n, n), dtype=int)
        for i, a in enumerate(classes):
            for j, p in enumerate(classes):
                mat[i, j] = np.sum((y_true == a) & (y_pred == p))

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(n), [str(c) for c in classes])
        ax.set_yticks(range(n), [str(c) for c in classes])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = self.out_dir / "forest_confusion.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
