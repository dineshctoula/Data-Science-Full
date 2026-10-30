"""Day 88 — Day 87's features, with today's weather taken from a forecast.

Each weather feature is f(today) minus the lag weeks' mean of f. The lag
weeks are in the past, so their weather was observed: that half stays
actual. Only today's half comes from the (simulated) forecast.

With a perfect forecast these columns equal Day 87's exactly, which the
tests check; the column names are kept so Day 87's plots and lookups work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import project_paths  # noqa: F401
from features import HUM_KNOTS, LADDER, PEAK_HOURS, TEMP_KNOTS, blocks, lag_weeks_mean

WEATHER_RUNG = "+ weather (actual, an upper bound)"
NO_WEATHER_RUNG = "+ holiday, drift (no weather)"


def _today_vs_lags(today: pd.Series, actual: pd.Series) -> pd.Series:
    # the forecast stands in for today; the lag weeks keep what really happened
    return today - lag_weeks_mean(actual)


def _hinges(today: pd.Series, actual: pd.Series, knots, name: str) -> dict:
    cols = {f"{name}_vs_lags": _today_vs_lags(today, actual)}
    for k in knots:
        cols[f"{name}_over_{k:g}_vs_lags"] = _today_vs_lags(np.maximum(0.0, today - k), np.maximum(0.0, actual - k))
    return cols


def weather_block(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """The Day 87 weather columns, with today's half read from `forecast`."""
    peak = ((actual["workingday"] == 1) & actual["hr"].isin(PEAK_HOURS)).astype(float)

    def flags(df):
        return (df["weathersit"] >= 3).astype(float), (df["weathersit"] == 2).astype(float)

    rain_a, mist_a = flags(actual)
    rain_f, mist_f = flags(forecast)
    cols = {}
    cols.update(_hinges(forecast["temp"], actual["temp"], TEMP_KNOTS, "temp"))
    cols.update(_hinges(forecast["hum"], actual["hum"], HUM_KNOTS, "hum"))
    cols["rain_vs_lags"] = _today_vs_lags(rain_f, rain_a)
    cols["mist_vs_lags"] = _today_vs_lags(mist_f, mist_a)
    cols["rain_at_peak_vs_lags"] = _today_vs_lags(rain_f * peak, rain_a * peak)
    return pd.DataFrame(cols, index=actual.index)


def design(actual: pd.DataFrame, forecast: pd.DataFrame | None, rung: str, cache: dict | None = None) -> pd.DataFrame:
    """Columns for one Day 87 rung; the weather rung reads today's weather from `forecast`."""
    parts = cache if cache is not None else blocks(actual)
    frames = []
    for name in LADDER[rung]:
        if name == "weather":
            if forecast is None:
                raise ValueError("the weather rung needs a forecast frame (pass the actual grid for a perfect one)")
            frames.append(weather_block(actual, forecast))
        else:
            # hour shape, holiday, and drift don't involve weather at all
            frames.append(parts[name])
    return pd.concat(frames, axis=1)
