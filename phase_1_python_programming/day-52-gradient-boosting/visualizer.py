"""Plots for the Day 52 gradient boosting walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boost_engine import GradientBoostingClassifier, GradientBoostingRegressor


class BoostPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def loss_curve(self, losses, title="train MSE vs boosting round"):
        if not losses:
            raise ValueError("empty loss list")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(range(1, len(losses) + 1), losses, color="#336699", linewidth=2)
        ax.set_xlabel("boosting round")
        ax.set_ylabel("train MSE")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "loss_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def regression_fit(self, model, X, y, title="boosted fit"):
        """1-D X only — scatter + sorted prediction line."""
        if not isinstance(model, GradientBoostingRegressor):
            raise ValueError("need a GradientBoostingRegressor")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[1] != 1:
            raise ValueError("regression_fit expects a single feature")

        order = np.argsort(X[:, 0])
        xs = X[order]
        preds = model.predict(xs)

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.scatter(X[:, 0], y, s=22, alpha=0.55, color="#666666", label="data")
        ax.plot(xs[:, 0], preds, color="#cc5533", linewidth=2, label="boosted F(x)")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "regression_fit.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def decision_map(self, model, X, y, title="boosted regions", filename="boost_regions.png"):
        if not isinstance(model, GradientBoostingClassifier):
            raise ValueError("need a GradientBoostingClassifier")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if X.shape[1] != 2:
            raise ValueError("decision map wants 2D features")

        pad = 0.5
        x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
        y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
        xx, yy = np.meshgrid(np.linspace(x0, x1, 160), np.linspace(y0, y1, 160))
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.contourf(xx, yy, zz, alpha=0.35, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=28)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = self.out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def lr_compare(self, curves: dict, title="learning rate vs loss"):
        """curves: {lr_label: loss_list}"""
        if not curves:
            raise ValueError("no curves")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, losses in curves.items():
            ax.plot(range(1, len(losses) + 1), losses, label=str(label), linewidth=2)
        ax.set_xlabel("boosting round")
        ax.set_ylabel("train MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "lr_compare.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
