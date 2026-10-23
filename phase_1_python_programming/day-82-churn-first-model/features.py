"""Day 82 — turn the cleaned churn table into model columns.

Two choices come straight out of the Day 81 audit:

- "No internet service" in the six add-on columns says the same thing as
  InternetService == "No", and "No phone service" repeats PhoneService ==
  "No". collapse_mirrors() folds them into plain "No", so the fact is
  stated once by InternetService / PhoneService.
- Churn falls steeply in the first months and slowly after, so
  log1p(tenure) is offered next to raw tenure.

Encoder learns everything (reference levels, means, stds) from train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import project_paths  # noqa: F401  (puts the Day 81 modules on the path)
from churn_data import CATEGORICAL

ADDONS = ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")


def collapse_mirrors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ADDONS:
        out[col] = out[col].replace("No internet service", "No")
    out["MultipleLines"] = out["MultipleLines"].replace("No phone service", "No")
    return out


def add_log_tenure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # log1p, not log: 11 customers have tenure 0
    out["log_tenure"] = np.log1p(out["tenure"])
    return out


class Encoder:
    """Numeric columns as-is, categoricals one-hot minus a reference level, all standardized.

    The reference level is the most common training level. Dropping one
    level per column keeps the columns from summing to the intercept, so
    the fit stays well defined even with a tiny penalty.
    """

    def __init__(self, numeric, categorical=CATEGORICAL):
        if not numeric and not categorical:
            raise ValueError("no columns to encode")
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.levels = None
        self.mean = None
        self.std = None
        self.names = None

    def fit(self, df: pd.DataFrame):
        self.levels = {}
        for col in self.categorical:
            counts = df[col].value_counts()
            # ties in the count go alphabetical, so the reference doesn't depend on row order
            ordered = sorted(counts.index, key=lambda lvl: (-counts[lvl], str(lvl)))
            self.levels[col] = ordered
        raw = self._unscaled(df)
        self.mean = raw.mean(axis=0)
        self.std = raw.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def _unscaled(self, df: pd.DataFrame) -> np.ndarray:
        parts, names = [], []
        for col in self.numeric:
            parts.append(df[col].to_numpy(dtype=float))
            names.append(col)
        for col in self.categorical:
            values = df[col].to_numpy()
            # levels[col][0] is the reference and gets no column of its own
            for level in self.levels[col][1:]:
                parts.append((values == level).astype(float))
                names.append(f"{col}={level}")
        self.names = tuple(names)
        return np.column_stack(parts)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("fit first")
        return (self._unscaled(df) - self.mean) / self.std

    def reference(self, col: str):
        return self.levels[col][0]


# Feature sets compared on validation. Each is (prep steps, numeric columns).
VARIANTS = {
    "as-is, 3 numerics": ((), ("tenure", "MonthlyCharges", "TotalCharges")),
    "collapsed, 3 numerics": ((collapse_mirrors,), ("tenure", "MonthlyCharges", "TotalCharges")),
    "collapsed, drop TotalCharges": ((collapse_mirrors,), ("tenure", "MonthlyCharges")),
    "collapsed, + log tenure": ((collapse_mirrors, add_log_tenure), ("tenure", "log_tenure", "MonthlyCharges")),
}


def apply_steps(df: pd.DataFrame, steps) -> pd.DataFrame:
    for step in steps:
        df = step(df)
    return df
