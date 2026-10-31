"""Plots for the Day 89 final test.

The sample-week chart is Day 86's BikePlots.sample_week; these are new.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class FinalPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        fig.tight_layout()
        path = self.out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def cards(self, cards, title="MAE by model, with 95% day-block intervals"):
        """cards: {slice label: card frame}; bars grouped by model, one colour per slice."""
        if not cards:
            raise ValueError("no cards")
        models = list(next(iter(cards.values()))["model"])
        x = np.arange(len(models))
        width = 0.8 / len(cards)
        colors = ["#999999", "#336699", "#cc5533"]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for i, ((label, frame), color) in enumerate(zip(cards.items(), colors)):
            frame = frame.set_index("model").loc[models]
            xs = x - 0.4 + width * (i + 0.5)
            ax.bar(xs, frame["mae"], width, color=color, label=label)
            # the error bar is the bootstrap interval over days, not a standard deviation
            ax.errorbar(xs, frame["mae"], yerr=[frame["mae"] - frame["lo"], frame["hi"] - frame["mae"]], fmt="none", ecolor="#111111", capsize=3, linewidth=0.9)
            for xi, v in zip(xs, frame["mae"]):
                ax.annotate(f"{v:.1f}", (xi, v), ha="center", va="top", fontsize=7, color="white", xytext=(0, -3), textcoords="offset points")
        ax.set_xticks(x, [m.replace(", ", ",\n", 1) for m in models], fontsize=8)
        ax.set_ylabel("MAE (bikes per hour)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(axis="y", alpha=0.25)
        return self._save(fig, "cards.png")

    def seed_gaps(self, gaps, against: str, title=None):
        """One dot per noise draw: primary minus `against`, with its day-block interval."""
        fig, ax = plt.subplots(figsize=(7.4, 4.0))
        y = np.arange(len(gaps))
        ax.errorbar(gaps["diff"], y, xerr=[gaps["diff"] - gaps["lo"], gaps["hi"] - gaps["diff"]], fmt="o", color="#336699", ecolor="#336699", capsize=3)
        ax.axvline(0, color="#111111", linewidth=1.0)
        ax.set_yticks(y, [f"draw {s}" for s in gaps["seed"]], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"MAE difference: primary − {against} (negative = primary better)")
        ax.set_title(title or f"each simulated forecast season vs {against}")
        ax.grid(axis="x", alpha=0.25)
        return self._save(fig, "seed_gaps.png")
