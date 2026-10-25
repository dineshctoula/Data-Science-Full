"""Day 84 — turn churn probabilities into a call list and score the list in dollars.

A policy maps (probabilities, customers) to a yes/no call decision. The
break-even policy never looks at outcomes: its cut-off comes from the
economics alone, so scoring it on validation is a fair test, not a fit.
The "hindsight best" list size does look at outcomes; it is shown only
as a ceiling, never as something a team could actually run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import project_paths  # noqa: F401
from churn_data import TARGET
from economics import Economics

CHAMPION_VARIANT = "collapsed, + log tenure"
CHAMPION_LAM = 1e-3


def champion_scores(train: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """Day 82's picked logistic, refit on train, scored on `other`."""
    from select_engine import run_grid

    trial = run_grid(train, other, variants=[CHAMPION_VARIANT], lams=[CHAMPION_LAM])[0]
    return trial.scores(other)


def sharpen(p, factor: float):
    """Multiply the log-odds by `factor`: same ranking, overconfident probabilities (Day 69)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    z = np.log(p / (1 - p)) * factor
    return 1.0 / (1.0 + np.exp(-z))


# --- policies -------------------------------------------------------------


def call_top_frac(score, frac: float) -> np.ndarray:
    """Call the top `frac` of customers by score (the Day 81 headline rule)."""
    if not 0 < frac <= 1:
        raise ValueError("frac should be in (0, 1]")
    score = np.asarray(score, dtype=float).reshape(-1)
    k = int(round(frac * len(score)))
    called = np.zeros(len(score), dtype=bool)
    # stable sort so equal scores keep row order and the list is reproducible
    called[np.argsort(-score, kind="stable")[:k]] = True
    return called


def call_if_profitable(p, df: pd.DataFrame, econ: Economics) -> np.ndarray:
    """Call exactly the customers whose expected profit is above zero."""
    return econ.expected_profit(p, df) > 0


# --- scoring --------------------------------------------------------------


def evaluate(called, y, df: pd.DataFrame, econ: Economics) -> dict:
    called = np.asarray(called, dtype=bool).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    gain = econ.realized_profit(y, df)
    return {
        "calls": int(called.sum()),
        "leavers_reached": int(y[called].sum()),
        "expected_saves": float((y[called] * econ.save_rate).sum()),
        "cost": float(called.sum() * econ.cost),
        "profit": float(gain[called].sum()),
    }


def profit_curve(order, y, df: pd.DataFrame, econ: Economics) -> np.ndarray:
    """Realized profit after calling the first k customers of `order`, for k = 0..n."""
    gain = econ.realized_profit(y, df)
    return np.concatenate([[0.0], np.cumsum(gain[np.asarray(order)])])


def hindsight_best(score, y, df: pd.DataFrame, econ: Economics):
    """The list size that would have been best on these rows. Peeks at y; a ceiling only."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    curve = profit_curve(order, y, df, econ)
    k = int(np.argmax(curve))
    return k, float(curve[k])


def bootstrap_profit_diff(called_a, called_b, y, df, econ, n_boot: int = 2000, seed: int = 84):
    """Paired interval for profit(a) - profit(b) on resampled customers.

    Both lists are fixed; only which customers show up is resampled, so a
    draw heavy in leavers lifts both lists together and the difference stays steady.
    """
    if n_boot < 100:
        raise ValueError("n_boot should be at least 100")
    a = np.asarray(called_a, dtype=float).reshape(-1)
    b = np.asarray(called_b, dtype=float).reshape(-1)
    gain = econ.realized_profit(y, df)
    per_row = gain * (a - b)
    rng = np.random.default_rng(seed)
    n = len(per_row)
    diffs = np.array([per_row[rng.integers(0, n, size=n)].sum() for _ in range(n_boot)])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(per_row.sum()), "lo": float(lo), "hi": float(hi), "share_above_zero": float(np.mean(diffs > 0))}


def sensitivity(p, y, df: pd.DataFrame, econ: Economics, save_rates, top_frac: float = 0.20) -> pd.DataFrame:
    """For each save rate: how many the break-even rule calls and what each list earns."""
    rows = []
    for s in save_rates:
        e = econ.with_save_rate(float(s))
        rule = evaluate(call_if_profitable(p, df, e), y, df, e)
        top = evaluate(call_top_frac(p, top_frac), y, df, e)
        best_k, best_profit = hindsight_best(p * e.value(df), y, df, e)
        rows.append(
            {
                "save_rate": float(s),
                "rule_calls": rule["calls"],
                "rule_profit": rule["profit"],
                "top_profit": top["profit"],
                "hindsight_calls": best_k,
                "hindsight_profit": best_profit,
            }
        )
    return pd.DataFrame(rows)
