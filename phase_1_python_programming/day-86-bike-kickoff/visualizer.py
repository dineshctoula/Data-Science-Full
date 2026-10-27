"""Plots for the Day 86 bike-sharing kickoff."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SLICE_COLORS = {"train": "#336699", "validation": "#2a7a4f", "test": "#cc5533"}


class BikePlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        fig.tight_layout()
        path = self.out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def timeline(self, grid, ranges, title="daily rentals, with the frozen time split"):
        """Daily totals over two years; slices shaded, outage days marked."""
        daily = grid.groupby("date").agg(total=("cnt", "sum"), outage=("outage", "first"))
        # an outage day's sum would read as 0 rentals; leave it out of the line
        line = daily["total"].where(~daily["outage"])
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for _, r in ranges.iterrows():
            ax.axvspan(r["first_day"], r["last_day"] + pd.Timedelta(days=1), color=SLICE_COLORS[r["slice"]], alpha=0.08)
            ax.text(r["first_day"] + pd.Timedelta(days=4), 0.97, r["slice"], transform=ax.get_xaxis_transform(), va="top", fontsize=9, color=SLICE_COLORS[r["slice"]])
        ax.plot(line.index, line.to_numpy(), color="#333333", linewidth=0.8)
        out = daily.index[daily["outage"]]
        ax.scatter(out, np.zeros(len(out)), marker="x", color="#cc5533", zorder=4, label=f"outage days ({len(out)})")
        ax.set_ylabel("rentals per day")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.9))
        ax.grid(alpha=0.25)
        return self._save(fig, "timeline.png")

    def daily_shapes(self, shapes, title="two kinds of day: mean rentals by hour"):
        """shapes: hour x kind frame from audit_engine.daily_shapes."""
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        colors = {"working day": "#336699", "weekend / holiday": "#cc5533"}
        for kind in shapes.columns:
            ax.plot(shapes.index, shapes[kind], marker="o", markersize=3, color=colors.get(kind, "#2a7a4f"), label=kind)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("hour of day")
        ax.set_ylabel("mean rentals per hour")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        return self._save(fig, "daily_shapes.png")

    def sample_week(self, ts, actual, preds, title="one validation week: actual vs the baselines"):
        """ts, actual: one week of hours; preds: {label: values on the same hours}."""
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.plot(ts, actual, color="#111111", linewidth=1.6, label="actual")
        colors = ["#336699", "#2a7a4f", "#cc5533", "#999999"]
        for (label, values), color in zip(preds.items(), colors):
            ax.plot(ts, values, color=color, linewidth=1.0, alpha=0.9, label=label)
        ax.set_ylabel("rentals per hour")
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=3, loc="upper left")
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        return self._save(fig, "sample_week.png")
