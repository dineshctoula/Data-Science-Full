"""Plots for the Day 50 decision tree walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tree_engine import DecisionTreeClassifier


class TreePlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def decision_map(self, model, X, y, title="tree decision regions", filename="tree_regions.png"):
        """Fill the plane with whatever class the tree would pick."""
        if not isinstance(model, DecisionTreeClassifier):
            raise ValueError("need a fitted DecisionTreeClassifier")
        if model.root is None:
            raise RuntimeError("model isn't fitted")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if X.shape[1] != 2:
            raise ValueError("decision map only handles 2D features")

        pad = 0.4
        x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
        y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
        xx, yy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.contourf(xx, yy, zz, alpha=0.35, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=32)
        ax.set_title(f"{title}  (depth≤{model.max_depth}, {model.criterion})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = self.out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def depth_sweep(self, depths, train_accs, test_accs, title="depth vs accuracy"):
        """Shows the classic overfit story: train climbs, test peaks then dips."""
        if len(depths) != len(train_accs) or len(depths) != len(test_accs):
            raise ValueError("depths / accuracies length mismatch")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(depths, train_accs, "o-", color="#336699", label="train")
        ax.plot(depths, test_accs, "s--", color="#cc5533", label="test")
        ax.set_xlabel("max_depth")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "depth_sweep.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def impurity_bars(self, labels_before, labels_left, labels_right, title="impurity drop"):
        """Tiny bar chart: parent vs left/right child gini — useful for the notes."""
        from tree_engine import gini

        vals = [
            gini(labels_before),
            gini(labels_left),
            gini(labels_right),
        ]
        names = ["parent", "left", "right"]

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#777777", "#336699", "#cc5533"]
        ax.bar(names, vals, color=colors)
        ax.set_ylabel("gini")
        ax.set_title(title)
        ax.set_ylim(0, max(0.55, max(vals) * 1.15))
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        fig.tight_layout()
        path = self.out_dir / "impurity_bars.png"
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
        path = self.out_dir / "tree_confusion.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
