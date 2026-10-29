"""Day 87 — day-ahead features, built as corrections to the 4-week baseline.

Plain regression on lags lost to the Day 86 baseline: it re-learned what
the baseline already knew, and fit quiet 3 a.m. hours as hard as rush hour.
So the model keeps the baseline as an offset and only learns how far today
should differ from it, on the log scale:

    log1p(forecast) = log1p(mean of the same hour over the last 4 weeks) + X @ beta

That changes what the features should be. The baseline is an average of
four past weeks, *including their weather and holidays*. So a feature
describes how today differs from those weeks: f(today) minus the mean of
f over the four lag weeks. A wet Tuesday after four dry ones gets a big
rain difference; a wet Tuesday after four wet ones gets none.

Temperature and humidity enter through hinges max(0, x - knot). A straight
line in temperature, learned mostly on cool months, said "warmer than the
last weeks means more riders" and got July badly wrong; the hinges let
warming help in spring and stop helping in summer heat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import project_paths  # noqa: F401  (puts the Day 86 modules on the path)
from baseline_engine import recent_weeks_mean
from bike_data import TARGET

WEEKS = 4
TEMP_KNOTS = (0.3, 0.5, 0.7)  # temp is degrees C / 41, so roughly 12, 20, and 29 C
HUM_KNOTS = (0.6, 0.8)
PEAK_HOURS = (7, 8, 9, 16, 17, 18, 19)


def lag_weeks_mean(series: pd.Series, weeks: int = WEEKS) -> pd.Series:
    """Mean of the same hour over the last `weeks` weeks: what the baseline 'saw'."""
    return pd.concat([series.shift(168 * k) for k in range(1, weeks + 1)], axis=1).mean(axis=1)


def versus_lags(series: pd.Series, weeks: int = WEEKS) -> pd.Series:
    """Today's value minus the lag weeks' mean of the same quantity."""
    return series - lag_weeks_mean(series, weeks)


def hinge_block(series: pd.Series, knots, name: str) -> pd.DataFrame:
    """[x, max(0, x - k1), max(0, x - k2), ...], each as a difference from the lag weeks."""
    cols = {f"{name}_vs_lags": versus_lags(series)}
    for k in knots:
        cols[f"{name}_over_{k:g}_vs_lags"] = versus_lags(np.maximum(0.0, series - k))
    return pd.DataFrame(cols)


def offset(grid: pd.DataFrame) -> pd.Series:
    """The Day 86 baseline the model corrects; NaN where its history is missing."""
    return recent_weeks_mean(grid, WEEKS)


def blocks(grid: pd.DataFrame) -> dict:
    """Every feature group, each a frame on the full hourly grid."""
    kind = grid["hr"].astype(int).astype(str) + "_" + np.where(grid["workingday"] == 1, "work", "off")
    shape = pd.get_dummies(kind, prefix="hour", dtype=float)

    # yesterday's whole-day mean is known at midnight; compare it with the lag weeks' days
    day_mean = grid.groupby("date")[TARGET].mean()
    yesterday = grid["date"].map(day_mean.shift(1)).astype(float)
    lag_days = pd.concat([day_mean.shift(7 * k) for k in range(1, WEEKS + 1)], axis=1).mean(axis=1)
    calendar = pd.DataFrame(
        {
            "holiday_vs_lags": versus_lags(grid["holiday"].astype(float)),
            "drift": np.log1p(yesterday) - np.log1p(grid["date"].map(lag_days).astype(float)),
        }
    )

    rain = (grid["weathersit"] >= 3).astype(float)
    mist = (grid["weathersit"] == 2).astype(float)
    peak = ((grid["workingday"] == 1) & grid["hr"].isin(PEAK_HOURS)).astype(float)
    weather = pd.concat(
        [
            hinge_block(grid["temp"], TEMP_KNOTS, "temp"),
            hinge_block(grid["hum"], HUM_KNOTS, "hum"),
            pd.DataFrame(
                {
                    "rain_vs_lags": versus_lags(rain),
                    "mist_vs_lags": versus_lags(mist),
                    # rain at rush hour hits commuters who would otherwise ride
                    "rain_at_peak_vs_lags": versus_lags(rain * peak),
                }
            ),
        ],
        axis=1,
    )
    return {"hour shape": shape, "holiday + drift": calendar, "weather": weather}


# each rung adds one block; the middle rung is the honest day-ahead forecast
LADDER = {
    "baseline × hour shape": ("hour shape",),
    "+ holiday, drift (no weather)": ("hour shape", "holiday + drift"),
    "+ weather (actual, an upper bound)": ("hour shape", "holiday + drift", "weather"),
}


def design(grid: pd.DataFrame, rung: str, cache: dict | None = None) -> pd.DataFrame:
    parts = cache if cache is not None else blocks(grid)
    return pd.concat([parts[name] for name in LADDER[rung]], axis=1)
