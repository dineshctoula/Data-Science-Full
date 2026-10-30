"""Day 88 — how much of the weather gain survives a forecast?

Two ways to train, the same way to forecast:

- "trained on actual": fit on train with the weather that happened (Day
  87's model), then forecast with forecast-grade weather. This is what you
  get by building a model on historical observations and deploying it.
- "trained on forecasts": fit on train with forecast-grade weather too,
  so the model learns how far to trust a forecast. A noisy input pulls a
  fitted slope toward zero (errors-in-variables); here that is the point,
  because the noise at fit time matches the noise at forecast time.

Every scenario is run over several noise draws; one draw is one season of
forecasts, and the spread across draws shows how much luck is in a number.
The ridge penalty stays at Day 87's rolling-CV pick, so nothing new is tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import project_paths  # noqa: F401
from baseline_engine import mae
from bike_data import TARGET
from correction_model import CorrectionRidge
from evaluate_engine import rolling_folds, usable
from features import blocks, offset
from forecast_features import NO_WEATHER_RUNG, WEATHER_RUNG, design
from weather_noise import Scenario, perturb

LAM = 0.1
MODES = ("trained on actual", "trained on forecasts")


@dataclass
class Run:
    scenario: str
    mode: str
    seed: int
    val_mae: float
    cv_mae: float
    rain_effect: float  # the fitted multiplier for "rain today, dry in the lag weeks"
    val_pred: pd.Series


def _fit(grid, X, base, mask):
    return CorrectionRidge(LAM).fit(X[mask], grid.loc[mask, TARGET], base[mask])


def _score(grid, model, X, base, mask):
    pred = pd.Series(model.predict(X[mask], base[mask]), index=X.index[mask])
    return pred, mae(grid.loc[mask, TARGET], pred)


def reference_runs(grid: pd.DataFrame, slices: pd.Series) -> dict:
    """The two fixed points: no weather at all, and perfect (actual) weather."""
    cache = blocks(grid)
    base = offset(grid)
    out = {}
    for label, X in (("no weather", design(grid, None, NO_WEATHER_RUNG, cache)), ("perfect", design(grid, grid, WEATHER_RUNG, cache))):
        ok = usable(grid, X, base)
        model = _fit(grid, X, base, (slices == "train") & ok)
        pred, val = _score(grid, model, X, base, (slices == "validation") & ok)
        cv = np.mean([_score(grid, _fit(grid, X, base, fit & ok), X, base, fc & ok)[1] for _, fit, fc in rolling_folds(grid, slices)])
        out[label] = {"val_mae": val, "cv_mae": float(cv), "val_pred": pred, "model": model}
    return out


def run_scenario(grid: pd.DataFrame, slices: pd.Series, scenario: Scenario, seeds) -> list:
    """Every seed × both training modes for one scenario."""
    cache = blocks(grid)
    base = offset(grid)
    X_actual = design(grid, grid, WEATHER_RUNG, cache)
    folds = rolling_folds(grid, slices)
    runs = []
    for seed in seeds:
        noisy = perturb(grid, scenario, seed)
        X_fc = design(grid, noisy, WEATHER_RUNG, cache)
        # usable() depends only on history, the baseline, and outages, so it's the same for both frames
        ok = usable(grid, X_fc, base)
        for mode in MODES:
            X_fit = X_actual if mode == "trained on actual" else X_fc
            model = _fit(grid, X_fit, base, (slices == "train") & ok)
            # the forecast side always sees forecast-grade weather
            pred, val = _score(grid, model, X_fc, base, (slices == "validation") & ok)
            cv = np.mean([_score(grid, _fit(grid, X_fit, base, fit & ok), X_fc, base, fc & ok)[1] for _, fit, fc in folds])
            rain = float(np.exp(dict(zip(model.names, model.raw_effects()))["rain_vs_lags"]))
            runs.append(Run(scenario.name, mode, int(seed), val, float(cv), rain, pred))
    return runs


def summarize(runs, refs: dict) -> pd.DataFrame:
    """Mean and spread over seeds, plus the share of the weather gain that survives.

    share kept = (no-weather MAE − scenario MAE) / (no-weather MAE − perfect MAE)
    1 means the forecast is as good as knowing the weather; 0 means it adds nothing.
    """
    df = pd.DataFrame([{k: getattr(r, k) for k in ("scenario", "mode", "seed", "val_mae", "cv_mae", "rain_effect")} for r in runs])
    rows = []
    for (scenario, mode), g in df.groupby(["scenario", "mode"], sort=False):
        row = {"scenario": scenario, "mode": mode, "seeds": len(g)}
        for key in ("val_mae", "cv_mae"):
            gain = refs["no weather"][key] - refs["perfect"][key]
            row[key] = float(g[key].mean())
            row[f"{key}_sd"] = float(g[key].std(ddof=1)) if len(g) > 1 else 0.0
            row[f"{key}_kept"] = float((refs["no weather"][key] - g[key].mean()) / gain)
        row["rain_effect"] = float(g["rain_effect"].mean())
        rows.append(row)
    return pd.DataFrame(rows)
