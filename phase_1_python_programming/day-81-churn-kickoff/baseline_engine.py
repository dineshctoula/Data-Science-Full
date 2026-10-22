"""Day 81 — the project metric and the bars a real model has to clear.

The question: a retention team can call one customer in five each month.
If they call the riskiest fifth, what share of next month's leavers do
they reach? That share is the headline metric (recall in the top 20%).
Log loss and AUC ride along to keep the probabilities honest.

The baselines are lookup tables: the training churn rate of the cell a
customer falls in. No fitting beyond counting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_data import TARGET

TOP_FRAC = 0.20


def log_loss(y, p) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.clip(np.asarray(p, dtype=float).reshape(-1), 1e-12, 1 - 1e-12)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("y and p should be the same non-empty length")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y, score) -> float:
    """Chance a random leaver outscores a random stayer. Ties count half."""
    y = np.asarray(y).reshape(-1).astype(int)
    score = np.asarray(score, dtype=float).reshape(-1)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need both classes for AUC")
    diff = pos[:, None] - neg[None, :]
    return float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))


def recall_at_top(y, score, frac: float = TOP_FRAC) -> float:
    """Share of all leavers found among the top `frac` of rows by score.

    When the cut-off lands inside a group of tied scores, the slots left
    are shared across that group in proportion, which is what a random
    tie-break gives on average. So a constant score gets exactly `frac`,
    and the result never depends on row order.
    """
    if not 0 < frac <= 1:
        raise ValueError("frac should be in (0, 1]")
    y = np.asarray(y).reshape(-1).astype(int)
    score = np.asarray(score, dtype=float).reshape(-1)
    if len(y) != len(score) or y.sum() == 0:
        raise ValueError("need matching lengths and at least one leaver")
    slots = frac * len(y)
    found = 0.0
    # walk the distinct scores from riskiest down, filling the call list
    for value in np.unique(score)[::-1]:
        in_group = score == value
        size = int(in_group.sum())
        hits = int(y[in_group].sum())
        if slots >= size:
            found += hits
            slots -= size
        else:
            found += hits * slots / size
            break
    return float(found / y.sum())


class RateTable:
    """Training churn rate per cell, shrunk toward the overall rate.

    Shrinkage (an m-estimate) keeps a cell with 3 customers from claiming
    a 0% or 100% rate: the cell's rate is blended with the overall rate as
    if `prior_rows` extra average customers sat in every cell.
    """

    def __init__(self, keys=(), prior_rows: float = 20.0):
        if prior_rows < 0:
            raise ValueError("prior_rows should be zero or more")
        self.keys = list(keys)
        self.prior_rows = prior_rows
        self.overall = None
        self.cells = None

    def fit(self, df: pd.DataFrame):
        self.overall = float(df[TARGET].mean())
        if not self.keys:
            self.cells = None
            return self
        grouped = df.groupby(self.keys, observed=True)[TARGET].agg(["sum", "size"])
        m = self.prior_rows
        self.cells = (grouped["sum"] + m * self.overall) / (grouped["size"] + m)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.overall is None:
            raise RuntimeError("fit first")
        if self.cells is None:
            return np.full(len(df), self.overall)
        index = pd.MultiIndex.from_frame(df[self.keys]) if len(self.keys) > 1 else pd.Index(df[self.keys[0]])
        # a cell that never appeared in training falls back to the overall rate
        return self.cells.reindex(index).fillna(self.overall).to_numpy(dtype=float)


def add_tenure_band(df: pd.DataFrame, edges=(0, 6, 12, 24, 48, 72)) -> pd.DataFrame:
    out = df.copy()
    out["tenure_band"] = pd.cut(out["tenure"], bins=list(edges), include_lowest=True).astype(str)
    return out


def score_all(y, p) -> dict:
    return {"top20_recall": recall_at_top(y, p), "auc": roc_auc(y, p), "log_loss": log_loss(y, p)}


def run_baselines(train: pd.DataFrame, val: pd.DataFrame, prior_rows: float = 20.0):
    """Fit each lookup on train, score it on validation. Test stays closed."""
    train, val = add_tenure_band(train), add_tenure_band(val)
    tables = {
        "train churn rate for everyone": RateTable([], prior_rows),
        "rate by contract": RateTable(["Contract"], prior_rows),
        "rate by contract × tenure band": RateTable(["Contract", "tenure_band"], prior_rows),
        "rate by contract × tenure band × internet": RateTable(["Contract", "tenure_band", "InternetService"], prior_rows),
    }
    results = []
    for name, table in tables.items():
        p = table.fit(train).predict(val)
        results.append({"baseline": name, "cells": 1 if table.cells is None else len(table.cells), **score_all(val[TARGET], p)})
    return pd.DataFrame(results)
