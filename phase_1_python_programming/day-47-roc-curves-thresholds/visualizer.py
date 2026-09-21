"""Matplotlib visualizations for Day 47 ROC curves and thresholds."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from roc_engine import CurveResult, ThresholdMetrics


class RocVisualizer:
    """Plot ROC/PR curves, threshold tradeoffs, and a chosen operating point."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_roc_curve(
        self,
        curve: CurveResult,
        highlight: ThresholdMetrics | None = None,
        title: str = "ROC curve",
    ) -> Path:
        """Plot TPR against FPR with the chance diagonal and optional operating point."""
        if not isinstance(curve, CurveResult):
            raise ValueError("A CurveResult is required to plot the ROC curve.")

        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        axis.plot(curve.fpr, curve.tpr, color="#4C78A8", linewidth=2.5, label=f"ROC (AUC = {curve.auc_roc:.3f})")
        # The diagonal is a random ranking: TPR = FPR at every threshold.
        axis.plot([0, 1], [0, 1], color="#E45756", linestyle="--", linewidth=1.5, label="Chance (AUC = 0.5)")
        if highlight is not None:
            axis.scatter(
                [highlight.false_positive_rate],
                [highlight.true_positive_rate],
                color="#54A24B",
                s=80,
                zorder=3,
                label=f"Chosen t = {highlight.threshold:.2f}",
            )
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title=title, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
        figure.tight_layout()
        path = self.output_dir / "roc_curve.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_precision_recall(
        self,
        curve: CurveResult,
        highlight: ThresholdMetrics | None = None,
        title: str = "Precision–recall curve",
    ) -> Path:
        """Plot precision against recall across thresholds."""
        if not isinstance(curve, CurveResult):
            raise ValueError("A CurveResult is required to plot the PR curve.")

        figure, axis = plt.subplots(figsize=(6.5, 5.0))
        axis.plot(curve.recall, curve.precision, color="#F58518", linewidth=2.5, label="PR curve")
        if highlight is not None:
            axis.scatter(
                [highlight.recall],
                [highlight.precision],
                color="#54A24B",
                s=80,
                zorder=3,
                label=f"Chosen t = {highlight.threshold:.2f}",
            )
        axis.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(-0.02, 1.02), ylim=(-0.02, 1.05))
        axis.grid(alpha=0.25)
        axis.legend(loc="lower left")
        figure.tight_layout()
        path = self.output_dir / "precision_recall_curve.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_threshold_tradeoff(
        self,
        curve: CurveResult,
        title: str = "Metrics vs decision threshold",
    ) -> Path:
        """Overlay TPR, FPR, precision, and F1 as the cutoff changes."""
        if not isinstance(curve, CurveResult):
            raise ValueError("A CurveResult is required to plot threshold tradeoffs.")

        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        axis.plot(curve.thresholds, curve.tpr, label="TPR / recall", linewidth=2)
        axis.plot(curve.thresholds, curve.fpr, label="FPR", linewidth=2)
        axis.plot(curve.thresholds, curve.precision, label="Precision", linewidth=2)
        axis.plot(curve.thresholds, curve.f1, label="F1", linewidth=2)
        axis.set(xlabel="Threshold", ylabel="Metric value", title=title, ylim=(-0.05, 1.05))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = self.output_dir / "threshold_tradeoff.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_auc_comparison(
        self,
        labeled_aucs: dict[str, float],
        title: str = "AUC comparison",
    ) -> Path:
        """Bar-chart AUC values for contrasting ranking quality."""
        if not labeled_aucs:
            raise ValueError("At least one AUC value is required.")
        names = list(labeled_aucs.keys())
        values = [float(labeled_aucs[name]) for name in names]
        if any(not np.isfinite(value) for value in values):
            raise ValueError("AUC values must be finite.")

        figure, axis = plt.subplots(figsize=(6.5, 4.5))
        axis.bar(names, values, color=["#4C78A8", "#E45756"][: len(names)], alpha=0.9)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
        axis.set(ylabel="ROC AUC", title=title, ylim=(0.0, 1.05))
        axis.grid(alpha=0.25, axis="y")
        axis.legend()
        for index, value in enumerate(values):
            axis.text(index, value + 0.02, f"{value:.3f}", ha="center")
        figure.tight_layout()
        path = self.output_dir / "auc_comparison.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
