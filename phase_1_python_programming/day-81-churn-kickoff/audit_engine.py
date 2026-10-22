"""Day 81 — look at the cleaned churn table before modelling anything.

Everything here is description: what each column holds, how churn moves
with it, and which columns quietly repeat each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_data import CATEGORICAL, ID, NUMERIC, TARGET


def column_audit(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: kind, blanks, distinct values, and the most common value."""
    rows = []
    for col in df.columns:
        values = df[col]
        top = values.value_counts().index[0]
        rows.append(
            {
                "column": col,
                "kind": "numeric" if col in NUMERIC else "id" if col == ID else "target" if col == TARGET else "category",
                "missing": int(values.isna().sum()),
                "distinct": int(values.nunique()),
                "top": top,
                "top_share": float((values == top).mean()),
            }
        )
    return pd.DataFrame(rows)


def churn_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Churn rate and row count for each level, highest rate first."""
    out = df.groupby(col, observed=True)[TARGET].agg(rate="mean", rows="size").reset_index()
    return out.sort_values("rate", ascending=False, ignore_index=True)


def tenure_buckets(df: pd.DataFrame, edges=(0, 6, 12, 24, 48, 72)) -> pd.DataFrame:
    """Churn by tenure band. The first band includes the tenure-0 customers."""
    edges = list(edges)
    if edges != sorted(edges) or len(edges) < 2:
        raise ValueError("edges should be increasing")
    labels = [f"{lo}–{hi}" if i == 0 else f"{lo + 1}–{hi}" for i, (lo, hi) in enumerate(zip(edges, edges[1:]))]
    # include_lowest so tenure 0 lands in the first band instead of falling out as NaN
    band = pd.cut(df["tenure"], bins=edges, labels=labels, include_lowest=True)
    out = df.groupby(band, observed=True)[TARGET].agg(rate="mean", rows="size").reset_index()
    return out.rename(columns={"tenure": "band"})


def shadow_levels(df: pd.DataFrame, cols=CATEGORICAL):
    """Levels that pick out exactly the same rows as a level of another column.

    Example: 'No internet service' in OnlineSecurity marks the same 1526
    customers as InternetService == 'No'. A model would see that fact six
    or seven times over, so it's worth knowing before encoding.
    """
    masks = {}
    for col in cols:
        for level in df[col].unique():
            masks[(col, level)] = (df[col] == level).to_numpy()
    groups = {}
    for key, mask in masks.items():
        # identical boolean columns hash to identical bytes
        groups.setdefault(mask.tobytes(), []).append(key)
    return [sorted(keys) for keys in groups.values() if len({c for c, _ in keys}) > 1]


def charges_check(df: pd.DataFrame) -> dict:
    """How closely TotalCharges follows tenure × MonthlyCharges.

    If the two agree almost perfectly, TotalCharges carries little that
    tenure and MonthlyCharges don't already say.
    """
    product = df["tenure"] * df["MonthlyCharges"]
    total = df["TotalCharges"]
    gap = (total - product).abs()
    return {
        "corr": float(np.corrcoef(product, total)[0, 1]),
        "median_gap": float(gap.median()),
        # relative gap, skipping the tenure-0 rows where both sides are 0
        "median_rel_gap": float((gap / product.where(product > 0)).median()),
    }
