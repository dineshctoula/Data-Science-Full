"""Plots for the Day 53 linear SVM walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from svm_engine import LinearSVM


class SVMPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def decision_boundary(self, model, X, y, title="linear SVM", filename="svm_boundary.png"):
        if not isinstance(model, LinearSVM) or model.w_ is None:
            raise ValueError("need a fitted LinearSVM")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if X.shape[1] != 2:
            raise ValueError("boundary plot wants 2D data")

        pad = 0.6
        x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
        y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
        xx, yy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
        zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.35)
        # decision line + margins at ±1
        ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors=["gray", "black", "gray"],
                   linestyles=["--", "-", "--"], linewidths=[1.2, 2.0, 1.2])
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=32)

        # highlight soft support / margin violators
        mask = model.margin_mask(X, y)
        if mask.any():
            ax.scatter(
                X[mask, 0], X[mask, 1],
                facecolors="none", edgecolors="black", s=110, linewidths=1.5,
                label="on/inside margin",
            )
            ax.legend(loc="best", fontsize=8)

        ax.set_title(f"{title}  (C={model.C})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = self.out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def loss_curve(self, losses, title="SVM objective vs epoch"):
        if not losses:
            raise ValueError("empty loss list")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(range(1, len(losses) + 1), losses, color="#336699", linewidth=2)
        ax.set_xlabel("epoch")
        ax.set_ylabel("0.5||w||² + C·mean hinge")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "svm_loss.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def c_sweep_bars(self, scores: dict, title="test accuracy vs C"):
        if not scores:
            raise ValueError("empty scores")
        cs = list(scores.keys())
        vals = [scores[c] for c in cs]
        labels = [str(c) for c in cs]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(labels, vals, color="#336699")
        ax.set_xlabel("C")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "c_sweep.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
