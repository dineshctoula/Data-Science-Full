"""Day 86 — the forecasting task, its metric, and the bars to beat.

The task: at midnight, forecast all 24 hours of the next day. A forecast
for day D may use every count up to the end of day D-1, never anything
from D itself. Every baseline here is built from shifts of at least 24
hours, so it obeys that by construction.

Headline metric: MAE, in bikes per hour. WAPE (total absolute error /
total rentals) rides along because demand grows: an MAE of 50 means more
in a slow January than in a busy September.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bike_data import TARGET


def mae(y, pred) -> float:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if len(y) != len(pred) or len(y) == 0:
        raise ValueError("y and pred should be the same non-empty length")
    return float(np.mean(np.abs(y - pred)))


def rmse(y, pred) -> float:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def wape(y, pred) -> float:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    total = np.sum(np.abs(y))
    if total == 0:
        raise ValueError("WAPE is undefined when every actual is 0")
    return float(np.sum(np.abs(y - pred)) / total)


def lag(grid: pd.DataFrame, hours: int) -> pd.Series:
    """The count `hours` earlier on the complete hourly grid (NaN across outages)."""
    if hours < 24:
        # anything shorter would peek inside the day being forecast
        raise ValueError("a day-ahead forecast can only use lags of 24 hours or more")
    return grid[TARGET].shift(hours)


def recent_weeks_mean(grid: pd.DataFrame, weeks: int = 4) -> pd.Series:
    """Mean of the same hour on the same weekday over the last `weeks` weeks.

    An outage week is skipped rather than averaged in; if every week in the
    window is an outage, the forecast is NaN.
    """
    if weeks < 1:
        raise ValueError("weeks should be at least 1")
    stack = pd.concat([lag(grid, 168 * k) for k in range(1, weeks + 1)], axis=1)
    return stack.mean(axis=1, skipna=True)


def train_profile(grid: pd.DataFrame, train_mask) -> pd.Series:
    """Training-period mean by (working day?, hour). Frozen: it never sees growth after train."""
    keys = ["workingday", "hr"]
    table = grid.loc[train_mask].groupby(keys)[TARGET].mean()
    return pd.Series(table.reindex(pd.MultiIndex.from_frame(grid[keys])).to_numpy(), index=grid.index)


def baselines(grid: pd.DataFrame, train_mask) -> dict:
    return {
        "yesterday, same hour": lag(grid, 24),
        "last week, same hour": lag(grid, 168),
        "mean of last 4 weeks, same hour": recent_weeks_mean(grid, 4),
        "training-period profile": train_profile(grid, train_mask),
    }


def score_table(y, preds: dict) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        rows.append({"baseline": name, "mae": mae(y, p), "rmse": rmse(y, p), "wape": wape(y, p)})
    return pd.DataFrame(rows)


def day_block_bootstrap(y, pred_a, pred_b, days, metric=mae, n_boot: int = 2000, seed: int = 86):
    """Paired interval for metric(a) - metric(b), resampling whole days.

    The 24 hours of one day share weather, events, and mood; resampling
    single hours would pretend they are 24 separate pieces of evidence and
    give intervals that are far too narrow.
    """
    y = np.asarray(y, dtype=float)
    a, b = np.asarray(pred_a, dtype=float), np.asarray(pred_b, dtype=float)
    days = np.asarray(days)
    uniq, day_idx = np.unique(days, return_inverse=True)
    rows_of = [np.flatnonzero(day_idx == d) for d in range(len(uniq))]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([rows_of[d] for d in pick])
        diffs[i] = metric(y[rows], a[rows]) - metric(y[rows], b[rows])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(metric(y, a) - metric(y, b)), "lo": float(lo), "hi": float(hi), "days": int(len(uniq))}
