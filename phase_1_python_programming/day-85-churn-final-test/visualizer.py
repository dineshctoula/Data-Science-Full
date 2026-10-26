"""Plots for the Day 85 final test.

Gains, calibration, and the profit curve reuse the Day 81, 82, and 84
plot classes; the one new chart puts validation and test side by side.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class FinalPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def val_vs_test(self, cards, title="validation (Days 82–84) vs the one test opening"):
        """cards: {label: score_card dict}. One panel per metric, bars with 95% intervals."""
        if not cards:
            raise ValueError("no cards")
        metrics = [("top20_recall", "top-20% recall (higher is better)"), ("auc", "AUC (higher is better)"), ("log_loss", "log loss (lower is better)")]
        fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4.2))
        labels = list(cards)
        colors = ["#999999", "#2a7a4f", "#336699", "#cc5533"]
        x = np.arange(len(labels))
        for ax, (key, name) in zip(axes, metrics):
            points = np.array([cards[label][key][0] for label in labels])
            lo = np.array([cards[label][key][1] for label in labels])
            hi = np.array([cards[label][key][2] for label in labels])
            ax.bar(x, points, color=colors[: len(labels)], width=0.6)
            # error bars are the bootstrap interval, not a standard deviation
            ax.errorbar(x, points, yerr=[points - lo, hi - points], fmt="none", ecolor="#111111", capsize=4, linewidth=1)
            for xi, v in zip(x, points):
                # inside the bar top, so the label never sits on the error bar
                ax.annotate(f"{v:.3f}", (xi, v), ha="center", va="top", fontsize=8, color="white", xytext=(0, -4), textcoords="offset points")
            # bars start well above zero so the gaps between slices are visible
            low = float(lo.min())
            ax.set_ylim(low - 0.15 * (float(hi.max()) - low) - 0.02, float(hi.max()) + 0.02)
            ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
            ax.set_title(name, fontsize=10)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(title)
        return self._save(fig, "val_vs_test.png")

    def _save(self, fig, name):
        fig.tight_layout()
        path = self.out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
