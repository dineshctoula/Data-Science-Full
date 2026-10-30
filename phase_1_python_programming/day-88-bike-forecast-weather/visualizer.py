"""Plots for the Day 88 forecast-weather stress test.

The sample-week chart is Day 86's BikePlots.sample_week; these are new.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODE_COLORS = {"trained on actual": "#cc5533", "trained on forecasts": "#336699"}


class StressPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        fig.tight_layout()
        path = self.out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def survival(self, summary, refs, order, title="how much of the weather gain survives a forecast"):
        """summary: stress_engine.summarize frame; refs: reference_runs; order: scenario names, best first."""
        panels = (("val_mae", "validation (Jul–Sep 2012)"), ("cv_mae", "rolling CV inside train (Jan–Jun 2012)"))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        labels = ["perfect"] + list(order)
        x = np.arange(len(labels))
        for ax, (key, name) in zip(axes, panels):
            for mode, color in MODE_COLORS.items():
                rows = summary[summary["mode"] == mode].set_index("scenario").loc[list(order)]
                ys = [refs["perfect"][key]] + rows[key].tolist()
                sd = [0.0] + rows[f"{key}_sd"].tolist()
                # error bars are the spread over noise draws: how much luck one season of forecasts carries
                ax.errorbar(x, ys, yerr=sd, marker="o", capsize=3, color=color, label=mode)
            ax.axhline(refs["no weather"][key], color="#2a7a4f", linestyle="--", linewidth=1.2, label="no weather at all")
            ax.set_xticks(x, labels, rotation=12, fontsize=8)
            ax.set_ylabel("MAE (bikes per hour)")
            ax.set_title(name, fontsize=10)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(title)
        return self._save(fig, "survival.png")

    def rain_trust(self, summary, perfect_rain, order, title="how much the model trusts a rain forecast"):
        """The fitted multiplier for 'rain today, dry in the lag weeks', by scenario and mode."""
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        labels = ["perfect"] + list(order)
        x = np.arange(len(labels))
        for mode, color in MODE_COLORS.items():
            rows = summary[summary["mode"] == mode].set_index("scenario").loc[list(order)]
            ax.plot(x, [perfect_rain] + rows["rain_effect"].tolist(), marker="o", color=color, label=mode)
        ax.axhline(1.0, color="#999999", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=12, fontsize=8)
        ax.set_ylabel("rental multiplier when rain is called")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        return self._save(fig, "rain_trust.png")
