"""Quick plots for the Day 49 KNN notebook/script."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from knn_engine import KNearestNeighbors


class KNNPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def decision_map(self, model, X, y, title="KNN decision regions", filename="knn_regions.png"):
        """Paint the plane by what the model would predict on a grid."""
        if not isinstance(model, KNearestNeighbors):
            raise ValueError("need a fitted KNearestNeighbors model")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if X.shape[1] != 2:
            raise ValueError("this plot only works in 2D")

        pad = 0.6
        x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
        y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
        # 180x180 is enough to look smooth without taking forever
        xx, yy = np.meshgrid(np.linspace(x0, x1, 180), np.linspace(y0, y1, 180))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = model.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=35)
        ax.set_title(f"{title}  (k={model.k})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = self.out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def k_sweep_bars(self, scores: dict, title="LOOCV accuracy vs k"):
        if not scores:
            raise ValueError("scores dict is empty")
        ks = sorted(scores)
        vals = [scores[k] for k in ks]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ks, vals, "o-", color="#336699", linewidth=2)
        ax.set_xlabel("k")
        ax.set_ylabel("leave-one-out accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        # mark the best k so it jumps out
        best_k = max(scores, key=scores.get)
        ax.axvline(best_k, color="#cc5533", linestyle="--", alpha=0.7, label=f"best k={best_k}")
        ax.legend()
        fig.tight_layout()
        path = self.out_dir / "k_sweep.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def neighbor_example(self, model, X, y, query, title="Who are the neighbors?"):
        """Show one query point and circle its k nearest training samples."""
        if model._X is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        q = np.asarray(query, dtype=float).reshape(1, -1)
        if q.shape[1] != 2 or X.shape[1] != 2:
            raise ValueError("neighbor sketch wants 2D data")

        d = model._pairwise_distances(q)[0]
        idx = np.argsort(d)[: model.k]

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.45, s=30, label="train")
        ax.scatter(X[idx, 0], X[idx, 1], facecolors="none", edgecolors="black", s=120, linewidths=2, label="neighbors")
        ax.scatter(q[0, 0], q[0, 1], marker="*", s=220, color="gold", edgecolors="black", label="query", zorder=5)
        # radius to the farthest of the k neighbors (rough visual of the neighborhood)
        radius = d[idx[-1]]
        circle = plt.Circle((q[0, 0], q[0, 1]), radius, fill=False, linestyle="--", color="gray")
        ax.add_patch(circle)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = self.out_dir / "neighbor_example.png"
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
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = self.out_dir / "knn_confusion.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
