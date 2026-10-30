"""Day 88 — turn actual weather into what a day-ahead forecast might have said.

The dataset only has the weather that happened. A real midnight forecast
would have had a weather *forecast*, which is wrong in particular ways:

- temperature: off by a few degrees, and mostly off the same way all day
  (a forecast that's 2 °C too warm at 9 a.m. is usually too warm at 3 p.m.),
  so the error is a per-day bias plus a smaller hourly wobble
- humidity: off by several percentage points, hour to hour
- rain: some rainy hours are missed, and some dry hours get a false alarm;
  timing is the weakest part of a rain forecast
- mist / cloud: sometimes called when it isn't, and vice versa

None of the error sizes below come from the data. They're assumptions in
the range of published day-ahead verification, and the point of running
three of them is to see how much the answer depends on that choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    temp_day_c: float  # std of the per-day temperature bias, in °C
    temp_hour_c: float  # std of the extra hour-to-hour temperature error, in °C
    hum_pp: float  # std of the humidity error, in percentage points
    rain_hit: float  # share of actual rain hours the forecast also calls rain
    rain_false: float  # share of dry hours the forecast wrongly calls rain
    mist_flip: float  # chance an hour's mist/cloud call is the wrong way round

    def __post_init__(self):
        if min(self.temp_day_c, self.temp_hour_c, self.hum_pp) < 0:
            raise ValueError("error sizes should be zero or more")
        for p in (self.rain_hit, self.rain_false, self.mist_flip):
            if not 0 <= p <= 1:
                raise ValueError("rates should be in [0, 1]")


PERFECT = Scenario("perfect (actual weather)", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
SCENARIOS = (
    Scenario("good forecast", temp_day_c=1.5, temp_hour_c=0.7, hum_pp=6.0, rain_hit=0.70, rain_false=0.03, mist_flip=0.10),
    Scenario("typical forecast", temp_day_c=2.5, temp_hour_c=1.0, hum_pp=10.0, rain_hit=0.55, rain_false=0.05, mist_flip=0.20),
    Scenario("poor forecast", temp_day_c=4.0, temp_hour_c=1.5, hum_pp=15.0, rain_hit=0.40, rain_false=0.08, mist_flip=0.30),
)

# the dataset stores temperature as °C / 41 and humidity as a 0-1 share
TEMP_SCALE_C = 41.0


def perturb(grid: pd.DataFrame, scenario: Scenario, seed: int) -> pd.DataFrame:
    """A copy of `grid` whose temp, hum, and weathersit are forecast-grade.

    Counts, calendar, and everything else are untouched. The caller decides
    which hours use the noisy copy (only "today"); history stays actual.
    """
    rng = np.random.default_rng(seed)
    out = grid.copy()
    days, day_idx = np.unique(grid["date"].to_numpy(), return_inverse=True)
    n = len(grid)

    # one bias per day, shared by its 24 hours, plus a little independent wobble per hour
    day_bias = rng.normal(0.0, scenario.temp_day_c, size=len(days))[day_idx]
    hour_noise = rng.normal(0.0, scenario.temp_hour_c, size=n)
    out["temp"] = grid["temp"] + (day_bias + hour_noise) / TEMP_SCALE_C

    out["hum"] = np.clip(grid["hum"] + rng.normal(0.0, scenario.hum_pp / 100.0, size=n), 0.0, 1.0)

    actual = grid["weathersit"].astype(int).to_numpy()
    rain = actual >= 3
    called_rain = np.where(rain, rng.random(n) < scenario.rain_hit, rng.random(n) < scenario.rain_false)
    mist = actual == 2
    called_mist = np.where(rng.random(n) < scenario.mist_flip, ~mist, mist)
    # rain wins over mist; otherwise clear (1). A real "heavy rain" (4) stays 4 when it's caught
    forecast = np.where(called_rain, np.maximum(actual, 3), np.where(called_mist, 2, 1))
    out["weathersit"] = forecast.astype(float)
    return out


def realized_errors(actual: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """What a scenario's draw actually did, so the assumptions can be read back."""
    rain_a = actual["weathersit"] >= 3
    rain_f = forecast["weathersit"] >= 3
    return {
        "temp_mae_c": float(np.mean(np.abs(forecast["temp"] - actual["temp"])) * TEMP_SCALE_C),
        "hum_mae_pp": float(np.mean(np.abs(forecast["hum"] - actual["hum"])) * 100.0),
        "rain_hit": float(rain_f[rain_a].mean()) if rain_a.any() else float("nan"),
        "rain_false": float(rain_f[~rain_a].mean()),
    }
