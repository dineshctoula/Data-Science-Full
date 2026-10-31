"""Day 89 — fit Project 2's named models and score them on whichever hours they're given.

Nothing here picks a slice. Test hours only reach these functions through
the gate in main.py, so the tests can run everything on train and
validation hours without the test slice ever being read.

All models are scored on the same hours: the scorable ones where every
named model has a full feature row. That keeps every comparison like for like.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import project_paths  # noqa: F401
from baseline_engine import day_block_bootstrap, mae, rmse, wape
from bike_data import TARGET
from correction_model import CorrectionRidge
from evaluate_engine import usable
from features import WEEKS, blocks, offset
from forecast_features import NO_WEATHER_RUNG, WEATHER_RUNG, design
from weather_noise import perturb


def robust_blocks(grid: pd.DataFrame) -> dict:
    """Day 87's feature blocks, with drift that survives an outage yesterday.

    drift = log1p(yesterday's mean) - log1p(lag weeks' daily means). After an
    outage day there is no "yesterday", so the last non-outage day stands in.
    The lag weeks' side already skips outage days, as the baseline does.
    """
    parts = dict(blocks(grid))
    day_mean = grid.groupby("date")[TARGET].mean()  # NaN on outage days
    last_good = day_mean.ffill()
    yesterday = grid["date"].map(last_good.shift(1)).astype(float)
    lag_days = pd.concat([day_mean.shift(7 * k) for k in range(1, WEEKS + 1)], axis=1).mean(axis=1)
    drift = np.log1p(yesterday) - np.log1p(grid["date"].map(lag_days).astype(float))
    calendar = parts["holiday + drift"].copy()
    calendar["drift"] = drift
    parts["holiday + drift"] = calendar
    return parts


@dataclass
class FinalRun:
    scored: pd.Series  # the common hours every model is judged on
    preds: dict  # label -> Series of predictions on the scored hours
    primary_by_seed: dict  # seed -> Series


def run_models(grid: pd.DataFrame, fit_mask, eval_mask, recipe) -> FinalRun:
    """Fit every named model on `fit_mask` hours and predict `eval_mask` hours."""
    cache = robust_blocks(grid)
    base = offset(grid)
    X_none = design(grid, None, NO_WEATHER_RUNG, cache)
    X_actual = design(grid, grid, WEATHER_RUNG, cache)
    noisy = {seed: design(grid, perturb(grid, recipe.FORECAST, seed), WEATHER_RUNG, cache) for seed in recipe.SEEDS}

    ok = usable(grid, X_none, base) & usable(grid, X_actual, base)
    for X in noisy.values():
        ok &= usable(grid, X, base)
    fit_rows = fit_mask & ok
    scored = eval_mask & ok

    def fit_predict(X_fit, X_eval):
        model = CorrectionRidge(recipe.LAM).fit(X_fit[fit_rows], grid.loc[fit_rows, TARGET], base[fit_rows])
        return pd.Series(model.predict(X_eval[scored], base[scored]), index=grid.index[scored])

    preds = {
        "4-week baseline": base[scored],
        "no weather": fit_predict(X_none, X_none),
        "weather, actual (upper bound)": fit_predict(X_actual, X_actual),
    }
    # trained on forecasts and scored on forecasts, one season of noise per seed (Day 88)
    by_seed = {seed: fit_predict(X, X) for seed, X in noisy.items()}
    return FinalRun(scored, preds, by_seed)


def day_interval(y, pred, days, n_boot: int, seed: int):
    """MAE with a 95% interval from resampling whole days."""
    y, pred, days = np.asarray(y, dtype=float), np.asarray(pred, dtype=float), np.asarray(days)
    uniq, idx = np.unique(days, return_inverse=True)
    rows_of = [np.flatnonzero(idx == d) for d in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        rows = np.concatenate([rows_of[d] for d in rng.integers(0, len(uniq), size=len(uniq))])
        draws[i] = mae(y[rows], pred[rows])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(mae(y, pred)), float(lo), float(hi)


def card(grid, run: FinalRun, recipe) -> pd.DataFrame:
    """One row per model: MAE with a day-block interval, RMSE, WAPE.

    The primary row is the mean over seeds; its interval comes from seed 0,
    and the seed spread is reported next to it.
    """
    y = grid.loc[run.scored, TARGET]
    days = grid.loc[run.scored, "date"]
    rows = []
    for label, pred in run.preds.items():
        point, lo, hi = day_interval(y, pred, days, recipe.N_BOOT, recipe.BOOT_SEED)
        rows.append({"model": label, "mae": point, "lo": lo, "hi": hi, "rmse": rmse(y, pred), "wape": wape(y, pred), "seed_sd": 0.0})
    per_seed = [mae(y, p) for p in run.primary_by_seed.values()]
    first = run.primary_by_seed[recipe.SEEDS[0]]
    _, lo, hi = day_interval(y, first, days, recipe.N_BOOT, recipe.BOOT_SEED)
    rows.append(
        {
            "model": recipe.PRIMARY,
            "mae": float(np.mean(per_seed)),
            "lo": lo,
            "hi": hi,
            "rmse": float(np.mean([rmse(y, p) for p in run.primary_by_seed.values()])),
            "wape": float(np.mean([wape(y, p) for p in run.primary_by_seed.values()])),
            "seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        }
    )
    return pd.DataFrame(rows)


def primary_gaps(grid, run: FinalRun, recipe, against: str) -> pd.DataFrame:
    """Primary minus one other model, per seed, each with a day-block interval."""
    y = grid.loc[run.scored, TARGET]
    days = grid.loc[run.scored, "date"]
    rows = []
    for seed, pred in run.primary_by_seed.items():
        d = day_block_bootstrap(y, pred, run.preds[against], days, n_boot=recipe.N_BOOT, seed=recipe.BOOT_SEED)
        rows.append({"seed": seed, "diff": d["diff"], "lo": d["lo"], "hi": d["hi"]})
    return pd.DataFrame(rows)
