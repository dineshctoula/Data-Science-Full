"""Plots for the Day 81 churn-project kickoff."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class KickoffPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        fig.tight_layout()
        path = self.out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def churn_profile(self, panels, overall, title="churn rate by column (all 7043 customers)"):
        """panels: list of (column name, table with a label column, 'rate', and 'rows')."""
        if not panels:
            raise ValueError("no panels")
        cols = 2
        rows = int(np.ceil(len(panels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(10, 3.4 * rows), squeeze=False)
        for ax, (name, table) in zip(axes.flat, panels):
            labels = table.iloc[:, 0].astype(str).tolist()
            ax.barh(labels, table["rate"], color="#336699")
            # the dashed line is the overall rate; bars past it are the risky groups
            ax.axvline(overall, color="#cc5533", linestyle="--", linewidth=1.2)
            for y_pos, (rate, n) in enumerate(zip(table["rate"], table["rows"])):
                ax.text(rate + 0.01, y_pos, f"{rate:.0%}  (n={n})", va="center", fontsize=8)
            ax.set_xlim(0, max(0.7, float(table["rate"].max()) + 0.15))
            ax.set_title(name, fontsize=10)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.25)
        for ax in list(axes.flat)[len(panels) :]:
            ax.axis("off")
        fig.suptitle(f"{title}; dashed line = overall {overall:.1%}")
        return self._save(fig, "churn_profile.png")

    def gains(self, y, scores, frac=0.20, title="gains on validation: who gets called first"):
        """Cumulative share of leavers reached vs share of customers called.

        scores: dict {label: score array}. Ties are spread evenly, the same
        way recall_at_top shares them, so the curve passes through the metric.
        """
        y = np.asarray(y).reshape(-1).astype(int)
        if not scores:
            raise ValueError("no scores")
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        colors = ["#999999", "#2a7a4f", "#336699", "#cc5533", "#7a4f9a"]
        for (label, score), color in zip(scores.items(), colors):
            score = np.asarray(score, dtype=float).reshape(-1)
            xs, ys = [0.0], [0.0]
            for value in np.unique(score)[::-1]:
                in_group = score == value
                # a tie group is one straight segment: its leavers arrive at an even pace
                xs.append(xs[-1] + in_group.sum() / len(y))
                ys.append(ys[-1] + y[in_group].sum() / y.sum())
            ax.plot(xs, ys, color=color, linewidth=1.6, label=label)
        ax.axvline(frac, color="#111111", linestyle=":", linewidth=1.0, label=f"call list = top {frac:.0%}")
        ax.set_xlabel("share of customers called (riskiest first)")
        ax.set_ylabel("share of leavers reached")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.25)
        return self._save(fig, "gains.png")
