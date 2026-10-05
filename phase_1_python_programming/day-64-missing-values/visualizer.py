"""Plots for the Day 64 missing-value walkthrough."""

from pathlib import Path

import matplotlib.pyplot as plt


class MissingPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def missing_bars(self, report, title="share of blanks"):
        if not report:
            raise ValueError("empty report")
        names = [r["name"] for r in report]
        rates = [r["rate"] for r in report]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.bar(names, rates, color="#336699")
        ax.set_ylim(0, max(0.5, max(rates) * 1.25))
        ax.set_ylabel("fraction missing")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "missing_rates.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def mse_bars(self, rows, title="holdout MSE by how we treat blanks"):
        if not rows:
            raise ValueError("no rows")
        names = [r.name for r in rows]
        vals = [r.mse for r in rows]
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        ax.bar(names, vals, color="#cc5533")
        ax.set_ylabel("validation MSE")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = self.out_dir / "impute_mse.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
