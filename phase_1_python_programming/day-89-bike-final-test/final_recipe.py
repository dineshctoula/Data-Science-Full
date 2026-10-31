"""Day 89 — every choice for Project 2's final test, written before test is opened.

Committed on its own, ahead of any code that reads test hours. The Day 85
gate fingerprints this file and final_engine.py on the first opening and
refuses to reopen test if either has changed.

Where each choice came from:
- features, penalty: Day 87 (rolling CV inside train picked lam = 0.1)
- train the weather model on forecast-grade weather: Day 88 (training on
  actual weather and forecasting from a forecast lost to no weather at all)
- primary vs comparison: Day 88's CV favoured weather at typical forecast
  quality (40.5 vs 45.3) while summer validation called it a tie (44.8 vs
  44.7), so both are named here rather than picked after seeing test
"""

from __future__ import annotations

import project_paths  # noqa: F401
from weather_noise import SCENARIOS

LAM = 0.1

# the model that gets the headline number, and the one it must beat
PRIMARY = "weather, trained and scored on typical forecasts"
COMPARISON = "no weather"
# shown for context only; neither is a forecast anyone could run at midnight
REFERENCES = ("4-week baseline", "weather, actual (upper bound)")

# Day 88's middle scenario; test weather is drawn the same way, never the actual
FORECAST = next(s for s in SCENARIOS if s.name == "typical forecast")
# ten seasons of simulated forecasts; every number on the test card is reported over all ten
SEEDS = tuple(range(10))

# validation has no job left, so the final fit uses everything before test
FIT_ON = ("train", "validation")

# Sandy (2012-10-29/30) sits inside test. Day 87's drift reads yesterday's whole-day
# mean, which doesn't exist after an outage day; without a rule, 2012-10-31 would
# silently drop out of scoring. Rule: drift falls back to the last non-outage day.
# This follows from the outage flags alone and changes nothing in validation.
DRIFT_FALLBACK = "last non-outage day"

# intervals resample whole days, as on Days 86-88
N_BOOT = 2000
BOOT_SEED = 89

# picked now so the chart can't be chosen to flatter: Thanksgiving week
SAMPLE_WEEK = ("2012-11-19", "2012-11-25")
