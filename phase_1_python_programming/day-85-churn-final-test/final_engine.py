"""Day 85 — fit the final models and build the score card.

Nothing in here knows which slice it is scoring. The test rows only reach
these functions through holdout_gate.open_test() in main.py, so the unit
tests can run every function on train and validation rows without the
test slice ever being read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import project_paths  # noqa: F401
from baseline_engine import RateTable, add_tenure_band, log_loss, recall_at_top, roc_auc
from churn_data import TARGET
from features import VARIANTS, Encoder, apply_steps
from logit_engine import NewtonLogistic
from policy_engine import bootstrap_profit_diff, call_if_profitable, call_top_frac, evaluate, hindsight_best


@dataclass
class Champion:
    steps: tuple
    encoder: Encoder
    model: NewtonLogistic

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.encoder.transform(apply_steps(df, self.steps)))


def fit_champion(fit_df: pd.DataFrame, variant: str, lam: float) -> Champion:
    """The Day 82 recipe, refit from scratch on whatever rows it is given."""
    steps, numeric = VARIANTS[variant]
    prepared = apply_steps(fit_df, steps)
    # the encoder's means and stds come from the fit rows only, as on every earlier day
    encoder = Encoder(numeric).fit(prepared)
    model = NewtonLogistic(lam=lam).fit(encoder.transform(prepared), prepared[TARGET])
    return Champion(steps, encoder, model)


class Baseline:
    """The Day 81 lookup floor, refit on the same rows as the champion."""

    def __init__(self, keys):
        self.table = RateTable(list(keys))

    def fit(self, fit_df):
        self.table.fit(add_tenure_band(fit_df))
        return self

    def predict(self, df):
        return self.table.predict(add_tenure_band(df))


def interval(metric, y, score, n_boot: int, seed: int):
    """Point value plus a 95% percentile bootstrap interval over customers."""
    y = np.asarray(y).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        take = rng.integers(0, len(y), size=len(y))
        draws[i] = metric(y[take], score[take])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(metric(y, score)), float(lo), float(hi)


def score_card(y, p, n_boot: int, seed: int) -> dict:
    """Headline metric, ranking, and probability quality, each with an interval."""
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    # the same seed for every metric, so all three intervals use the same resamples
    return {
        "top20_recall": interval(recall_at_top, y, p, n_boot, seed),
        "auc": interval(roc_auc, y, p, n_boot, seed),
        "log_loss": interval(log_loss, y, p, n_boot, seed),
        "mean_p": float(p.mean()),
        "churn_rate": float(y.mean()),
    }


def policy_card(p, y, df: pd.DataFrame, econ, top_frac: float, n_boot: int, seed: int) -> dict:
    """Day 84's two call lists in dollars, the paired gap, and the hindsight ceiling."""
    rule = call_if_profitable(p, df, econ)
    fixed = call_top_frac(p, top_frac)
    _, ceiling = hindsight_best(np.asarray(p) * econ.value(df), y, df, econ)
    return {
        "rule": evaluate(rule, y, df, econ),
        "fixed": evaluate(fixed, y, df, econ),
        "gap": bootstrap_profit_diff(rule, fixed, y, df, econ, n_boot=n_boot, seed=seed),
        "ceiling": ceiling,
    }
