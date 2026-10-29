"""Plots for the Day 87 first bike-demand model.

The sample-week chart is Day 86's BikePlots.sample_week; these are new.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ModelPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        fig.tight_layout()
        path = self.out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def ladder(self, rows, title="each feature block, scored two ways"):
        """rows: list of (label, cv_mae, val_mae)."""
        if not rows:
            raise ValueError("no rows")
        labels = [r[0] for r in rows]
        x = np.arange(len(rows))
        width = 0.38
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        ax.bar(x - width / 2, [r[1] for r in rows], width, color="#336699", label="rolling CV inside train (Jan–Jun 2012)")
        ax.bar(x + width / 2, [r[2] for r in rows], width, color="#2a7a4f", label="validation (Jul–Sep 2012)")
        for xi, r in zip(x, rows):
            ax.text(xi - width / 2, r[1], f"{r[1]:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + width / 2, r[2], f"{r[2]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, labels, rotation=12, ha="right", fontsize=8)
        ax.set_ylabel("MAE (bikes per hour)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        return self._save(fig, "ladder.png")

    def cv_months(self, series, title="rolling CV: one forecast month at a time"):
        """series: {label: {month: mae}}."""
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        colors = ["#999999", "#336699", "#2a7a4f", "#cc5533"]
        for (label, months), color in zip(series.items(), colors):
            keys = list(months)
            ax.plot(keys, [months[k] for k in keys], marker="o", color=color, label=label)
        ax.set_xlabel("forecast month (fit on everything before it)")
        ax.set_ylabel("MAE (bikes per hour)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        return self._save(fig, "cv_months.png")

    def weather_curves(self, curves, title="what the model learned: rentals vs the same hour on a reference day"):
        """curves: list of (axis label, x values, multiplier values, reference label)."""
        fig, axes = plt.subplots(1, len(curves), figsize=(5.4 * len(curves), 4.0), squeeze=False)
        for ax, (label, xs, mult, ref) in zip(axes.flat, curves):
            ax.plot(xs, mult, color="#336699", linewidth=1.8)
            ax.axhline(1.0, color="#999999", linewidth=0.8)
            ax.set_xlabel(label)
            ax.set_ylabel(f"rental multiplier (1 = {ref})")
            ax.grid(alpha=0.25)
        fig.suptitle(title)
        return self._save(fig, "weather_curves.png")

    def error_by_weather(self, table, title="validation MAE by weather"):
        """table: rows of (weather label, hours, baseline mae, model mae)."""
        labels = [f"{r[0]}\n({r[1]} h)" for r in table]
        x = np.arange(len(table))
        width = 0.38
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(x - width / 2, [r[2] for r in table], width, color="#999999", label="4-week baseline")
        ax.bar(x + width / 2, [r[3] for r in table], width, color="#cc5533", label="model with weather")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel("MAE (bikes per hour)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        return self._save(fig, "error_by_weather.png")
