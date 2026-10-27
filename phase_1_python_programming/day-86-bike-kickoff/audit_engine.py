"""Day 86 — describe the hourly grid before forecasting anything.

All of it is description: where the gaps are, how fast demand grows,
what a day looks like, and which columns or lags would be cheating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bike_data import LEAKY, TARGET


def gap_report(grid: pd.DataFrame) -> pd.DataFrame:
    """One row per day that lost any hours: how many, and whether it counts as an outage."""
    days = grid.groupby("date").agg(missing=("missing", "sum"), outage=("outage", "first"))
    return days[days["missing"] > 0].reset_index()


def monthly_level(grid: pd.DataFrame) -> pd.DataFrame:
    """Mean rentals per hour for each calendar month, outage hours left out."""
    month = grid["ts"].dt.to_period("M")
    out = grid.groupby(month)[TARGET].mean().rename("per_hour").reset_index()
    out["ts"] = out["ts"].dt.to_timestamp()
    return out


def year_over_year(grid: pd.DataFrame) -> pd.Series:
    """2012 level divided by 2011 level, month by month."""
    level = monthly_level(grid)
    level["month"] = level["ts"].dt.month
    level["year"] = level["ts"].dt.year
    wide = level.pivot(index="month", columns="year", values="per_hour")
    return (wide[2012] / wide[2011]).rename("ratio")


def daily_shapes(grid: pd.DataFrame) -> pd.DataFrame:
    """Mean rentals by hour of day, working days vs everything else."""
    kind = np.where(grid["workingday"] == 1, "working day", "weekend / holiday")
    return grid.groupby([kind, grid["hr"]])[TARGET].mean().unstack(0)


def leak_check(grid: pd.DataFrame) -> dict:
    """Do the leaky columns rebuild the target exactly?"""
    # recorded hours outside outage days (an outage day's counts were blanked on purpose)
    recorded = ~grid["missing"] & grid[TARGET].notna()
    parts = grid.loc[recorded, list(LEAKY)].sum(axis=1)
    return {
        "rows": int(recorded.sum()),
        "exact": bool((parts == grid.loc[recorded, TARGET]).all()),
        # registered alone already explains most of the target, which is the trap
        "corr_registered": float(np.corrcoef(grid.loc[recorded, "registered"], grid.loc[recorded, TARGET])[0, 1]),
    }


def lag_correlations(grid: pd.DataFrame, lags=(1, 2, 24, 48, 168)) -> pd.DataFrame:
    """Correlation of each hour's count with the count `lag` hours earlier.

    A day-ahead forecast is made at midnight for all 24 hours of the next
    day, so for 11 p.m. tomorrow the newest known hour is 24 hours old, and
    for 1 a.m. it is 2 hours old. Only lags of 24 or more are available for
    every hour of the forecast day.
    """
    y = grid[TARGET]
    rows = []
    for lag in lags:
        both = pd.concat([y, y.shift(lag)], axis=1).dropna()
        rows.append({"lag_hours": lag, "corr": float(both.corr().iloc[0, 1]), "usable_day_ahead": lag >= 24})
    return pd.DataFrame(rows)


def weather_effect(grid: pd.DataFrame) -> pd.DataFrame:
    """Rentals per hour by the weather category (1 clear ... 4 heavy rain/snow)."""
    return grid.groupby("weathersit")[TARGET].agg(per_hour="mean", hours="size").reset_index()
