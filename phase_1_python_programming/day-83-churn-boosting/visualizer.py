"""Plots for the Day 83 boosting day.

The gains curve is Day 81's KickoffPlots.gains; these are the new ones.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class BoostPlots:
    def __init__(self, out_dir="output"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        fig.tight_layout()
        path = self.out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def cv_curves(self, curves, pick, logistic_cv, title="5-fold CV inside train: held-out log loss by tree count"):
        """curves: {config: (k, max_trees) array}; pick: the chosen Pick; logistic_cv: per-fold losses."""
        if not curves:
            raise ValueError("no curves")
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        cmap = plt.get_cmap("viridis")
        depths = sorted({cfg.max_depth for cfg in curves})
        for cfg, c in curves.items():
            color = cmap(depths.index(cfg.max_depth) / max(1, len(depths) - 1) * 0.85)
            style = "-" if cfg.learning_rate >= 0.1 else "--"
            ax.plot(np.arange(1, c.shape[1] + 1), c.mean(axis=0), style, color=color, linewidth=1.4, label=cfg.label())
        ax.axhline(float(np.mean(logistic_cv)), color="#cc5533", linewidth=1.4, label="Day 82 logistic, same folds")
        ax.scatter([pick.n_trees], [pick.cv_loss], color="#111111", zorder=4, s=30, label=f"pick: {pick.config.label()}, {pick.n_trees} trees")
        # the early trees are far above everything else; zoom on the part where the choices are made
        floor = min(float(c.mean(axis=0).min()) for c in curves.values())
        ax.set_ylim(floor - 0.005, floor + 0.04)
        ax.set_xlabel("trees")
        ax.set_ylabel("mean held-out log loss")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.25)
        return self._save(fig, "cv_curves.png")

    def shapes(self, panels, title="what each model learned for one column (log-odds, centered)"):
        """panels: list of (column label, x values, {model label: contribution array})."""
        if not panels:
            raise ValueError("no panels")
        fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.0), squeeze=False)
        colors = {"stumps": "#336699", "logistic": "#cc5533"}
        for ax, (label, xs, lines) in zip(axes.flat, panels):
            order = np.argsort(xs, kind="stable")
            for name, ys in lines.items():
                ys = np.asarray(ys, dtype=float)
                # centered so both curves share a zero; only the shape is comparable
                ys = ys - ys.mean()
                drawstyle = "steps-post" if name == "stumps" else "default"
                ax.plot(np.asarray(xs)[order], ys[order], color=colors.get(name, "#2a7a4f"), drawstyle=drawstyle, linewidth=1.6, label=name)
            ax.axhline(0, color="#999999", linewidth=0.8)
            ax.set_xlabel(label)
            ax.set_ylabel("contribution to log-odds")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
        fig.suptitle(title)
        return self._save(fig, "shapes.png")

    def gain_importance(self, names, gains, top: int = 12, title="booster: total split gain by column"):
        names = list(names)
        gains = np.asarray(gains, dtype=float).reshape(-1)
        if len(names) != len(gains) or not names:
            raise ValueError("names and gains should match and be non-empty")
        share = gains / gains.sum() if gains.sum() > 0 else gains
        keep = np.argsort(-share, kind="stable")[:top][::-1]
        fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(keep) + 1.4))
        ax.barh([names[j] for j in keep], share[keep], color="#336699")
        ax.set_xlabel("share of all split gain")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        return self._save(fig, "gain_importance.png")
