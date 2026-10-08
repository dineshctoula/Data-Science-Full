"""Plots for the Day 67 learning-curve walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt


class CurvePlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def two_curves(self, stiff, flexible, title="train and test error as the sample grows"):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
        for ax, curve, name in (
            (axes[0], stiff, f"degree {stiff.degree}"),
            (axes[1], flexible, f"degree {flexible.degree}"),
        ):
            ax.plot(curve.sizes, curve.train_mse, marker="o", color="#336699", label="train")
            ax.plot(curve.sizes, curve.test_mse, marker="o", color="#cc5533", label="test")
            ax.set_xlabel("training rows")
            ax.set_title(name)
            ax.grid(alpha=0.25)
            ax.legend()
        axes[0].set_ylabel("MSE")
        fig.suptitle(title)
        fig.tight_layout()
        path = self.out_dir / "learning_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def gap_lines(self, curves, title="test MSE minus train MSE"):
        curves = list(curves)
        if len(curves) == 0:
            raise ValueError("no curves")
        fig, ax = plt.subplots(figsize=(7.0, 4.3))
        colors = ["#336699", "#cc5533", "#2a7a4f"]
        for curve, color in zip(curves, colors):
            ax.plot(
                curve.sizes,
                curve.gap(),
                marker="o",
                color=color,
                label=f"degree {curve.degree}",
            )
        ax.axhline(0, color="#111111", linewidth=1.0)
        ax.set_xlabel("training rows")
        ax.set_ylabel("test − train")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "gap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
