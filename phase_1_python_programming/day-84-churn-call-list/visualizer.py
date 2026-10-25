"""Plots for the Day 84 call-list day."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class CallPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        fig.tight_layout()
        path = self.out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def profit_curves(self, curves, marks, title="validation: profit as the call list grows"):
        """curves: {label: profit after k calls, k = 0..n}; marks: list of (label, k, profit)."""
        if not curves:
            raise ValueError("no curves")
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        colors = ["#336699", "#2a7a4f", "#999999"]
        for (label, curve), color in zip(curves.items(), colors):
            ax.plot(np.arange(len(curve)), curve, color=color, linewidth=1.6, label=label)
        markers = ["o", "s", "^", "D"]
        for (label, k, profit), marker in zip(marks, markers):
            ax.scatter([k], [profit], marker=marker, s=46, color="#cc5533", zorder=4, edgecolor="white")
            ax.annotate(f"{label}\n{k} calls, ${profit:,.0f}", (k, profit), textcoords="offset points", xytext=(8, -26), fontsize=7.5)
        ax.axhline(0, color="#111111", linewidth=0.8)
        ax.set_xlabel("customers called (best first)")
        ax.set_ylabel("profit on validation ($)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.25)
        return self._save(fig, "profit_curves.png")

    def break_even_map(self, monthly, p, cut, called, title="who gets a call: risk vs bill"):
        """Each customer as a dot; the line is the break-even churn probability for that bill."""
        monthly = np.asarray(monthly, dtype=float)
        p = np.asarray(p, dtype=float)
        called = np.asarray(called, dtype=bool)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.scatter(monthly[~called], p[~called], s=9, color="#999999", alpha=0.6, label="not called")
        ax.scatter(monthly[called], p[called], s=9, color="#cc5533", alpha=0.7, label="called")
        order = np.argsort(monthly)
        # above this line a call is expected to pay for itself
        ax.plot(monthly[order], np.asarray(cut, dtype=float)[order], color="#111111", linewidth=1.4, label="break-even probability")
        ax.set_ylim(0, 1)
        ax.set_xlabel("monthly bill ($)")
        ax.set_ylabel("predicted churn probability")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
        return self._save(fig, "break_even_map.png")

    def sensitivity(self, table, current, title="the save rate nobody has measured"):
        """table: the policy_engine.sensitivity frame; current: the assumed save rate."""
        if table.empty:
            raise ValueError("empty table")
        s = table["save_rate"]
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
        left.plot(s, table["rule_calls"], marker="o", color="#cc5533", label="break-even rule")
        left.plot(s, table["hindsight_calls"], marker=".", linestyle="--", color="#999999", label="hindsight best (peeks)")
        left.set_xlabel("save rate")
        left.set_ylabel("customers called (of the validation slice)")
        left.set_title("list size")
        right.plot(s, table["rule_profit"], marker="o", color="#cc5533", label="break-even rule")
        right.plot(s, table["top_profit"], marker="s", color="#336699", label="always top 20%")
        right.plot(s, table["hindsight_profit"], marker=".", linestyle="--", color="#999999", label="hindsight best (peeks)")
        right.axhline(0, color="#111111", linewidth=0.8)
        right.set_xlabel("save rate")
        right.set_ylabel("profit on validation ($)")
        right.set_title("profit")
        for ax in (left, right):
            ax.axvline(current, color="#111111", linestyle=":", linewidth=1.0)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
        fig.suptitle(f"{title} (dotted line = assumed {current:g})")
        return self._save(fig, "sensitivity.png")
