"""Day 87 — score the feature ladder without leaning on validation alone.

The features in features.py were designed while looking at validation
MAE (Days 86-87 notes say so plainly), which makes those validation
numbers a little optimistic. As a check that the design isn't tuned to
one summer, every rung is also scored by rolling-origin CV inside train:

    fold 2012-01: fit on everything before 2012-01-01, forecast January
    fold 2012-02: fit on everything before 2012-02-01, forecast February
    ...                                                    ... through June

Each fold only ever forecasts forward, like the real task. The penalty is
chosen on these folds, so validation is used only to report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import project_paths  # noqa: F401
from baseline_engine import mae
from bike_data import TARGET
from correction_model import CorrectionRidge
from features import LADDER, blocks, design, offset
from split_engine import scorable

CV_MONTHS = ("2012-01", "2012-02", "2012-03", "2012-04", "2012-05", "2012-06")
LAMS = (1e-3, 1e-2, 1e-1)


def usable(grid: pd.DataFrame, X: pd.DataFrame, base: pd.Series) -> pd.Series:
    """Hours with a full feature row, a baseline, and a count worth scoring."""
    return X.notna().all(axis=1) & base.notna() & scorable(grid)


def rolling_folds(grid: pd.DataFrame, slices: pd.Series, months=CV_MONTHS):
    """(name, fit mask, forecast mask) per month, all inside the training slice."""
    train = slices == "train"
    period = grid["ts"].dt.to_period("M").astype(str)
    folds = []
    for month in months:
        start = pd.Timestamp(month + "-01")
        forecast = train & (period == month)
        if not forecast.any():
            raise ValueError(f"{month} is not inside the training slice")
        # strictly earlier days only: the fold never sees its own month
        folds.append((month, train & (grid["date"] < start), forecast))
    return folds


@dataclass
class RungResult:
    rung: str
    lam: float
    n_features: int
    cv_mae: float
    cv_by_month: dict
    val_mae: float
    model: CorrectionRidge
    val_pred: pd.Series


def _fit_score(grid, X, base, fit_mask, eval_mask, lam):
    model = CorrectionRidge(lam).fit(X[fit_mask], grid.loc[fit_mask, TARGET], base[fit_mask])
    pred = pd.Series(model.predict(X[eval_mask], base[eval_mask]), index=X.index[eval_mask])
    return model, pred, mae(grid.loc[eval_mask, TARGET], pred)


def score_ladder(grid: pd.DataFrame, slices: pd.Series, lams=LAMS):
    """Every rung: pick lam by mean CV MAE, then fit on all of train and score validation."""
    cache = blocks(grid)
    base = offset(grid)
    folds = rolling_folds(grid, slices)
    results = []
    for rung in LADDER:
        X = design(grid, rung, cache)
        ok = usable(grid, X, base)
        by_lam = {}
        for lam in lams:
            months = {name: _fit_score(grid, X, base, fit & ok, fc & ok, lam)[2] for name, fit, fc in folds}
            by_lam[lam] = months
        # ties within 0.1 bikes go to the stronger penalty
        best = min(float(np.mean(list(m.values()))) for m in by_lam.values())
        lam = max(l for l, m in by_lam.items() if np.mean(list(m.values())) <= best + 0.1)
        train_ok = (slices == "train") & ok
        val_ok = (slices == "validation") & ok
        model, pred, val_mae = _fit_score(grid, X, base, train_ok, val_ok, lam)
        results.append(
            RungResult(rung, lam, X.shape[1], float(np.mean(list(by_lam[lam].values()))), by_lam[lam], val_mae, model, pred)
        )
    return results


def baseline_cv(grid: pd.DataFrame, slices: pd.Series) -> dict:
    """The Day 86 4-week baseline on the same CV months, for a like-for-like bar."""
    base = offset(grid)
    ok = base.notna() & scorable(grid)
    return {name: mae(grid.loc[fc & ok, TARGET], base[fc & ok]) for name, _, fc in rolling_folds(grid, slices)}


def error_by(y, pred, groups) -> pd.DataFrame:
    """MAE and hour count for each group (e.g. weather category or hour of day)."""
    df = pd.DataFrame({"err": np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)), "group": np.asarray(groups)})
    return df.groupby("group")["err"].agg(mae="mean", hours="size").reset_index()
