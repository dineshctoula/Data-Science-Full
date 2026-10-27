"""Day 86 — split by time, once, and write the cut dates down.

Project 1 split customers at random because a customer's row stood alone.
Hours don't: demand in 2012 runs about 1.4-2.6x the same month in 2011,
and a random split would let a model see October 2012 while "forecasting"
July 2012. So every slice is a block of whole days, in order:

    train       2011-01-01 .. 2012-06-30   (18 months)
    validation  2012-07-01 .. 2012-09-30   (3 months)
    test        2012-10-01 .. 2012-12-31   (3 months, includes Sandy and the holidays)

The ranges are committed in split_dates.tsv; later days read that file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SLICES = ("train", "validation", "test")
DEFAULT_RANGES = (
    ("train", "2011-01-01", "2012-06-30"),
    ("validation", "2012-07-01", "2012-09-30"),
    ("test", "2012-10-01", "2012-12-31"),
)


def write_split_dates(path, ranges=DEFAULT_RANGES) -> Path:
    path = Path(path)
    lines = ["slice\tfirst_day\tlast_day"] + ["\t".join(r) for r in ranges]
    path.write_text("\n".join(lines) + "\n")
    return path


def load_split_dates(path) -> pd.DataFrame:
    """Read the ranges and refuse anything that isn't train, then validation, then test, back to back."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    if list(df.columns) != ["slice", "first_day", "last_day"] or tuple(df["slice"]) != SLICES:
        raise ValueError("split file should list train, validation, test in that order")
    df["first_day"] = pd.to_datetime(df["first_day"])
    df["last_day"] = pd.to_datetime(df["last_day"])
    if (df["last_day"] < df["first_day"]).any():
        raise ValueError("a slice ends before it starts")
    # each slice must begin the day after the previous one ends: no gap, no overlap
    gaps = df["first_day"].iloc[1:].to_numpy() - df["last_day"].iloc[:-1].to_numpy()
    if (gaps != pd.Timedelta(days=1)).any():
        raise ValueError("slices should be back to back in time")
    return df


def assign_slices(grid: pd.DataFrame, ranges: pd.DataFrame) -> pd.Series:
    """Slice name for every hour; an hour outside all ranges is an error."""
    labels = pd.Series(pd.NA, index=grid.index, dtype="object", name="slice")
    for _, r in ranges.iterrows():
        inside = (grid["date"] >= r["first_day"]) & (grid["date"] <= r["last_day"])
        labels[inside] = r["slice"]
    if labels.isna().any():
        raise ValueError("some hours fall outside every slice")
    return labels


def scorable(grid: pd.DataFrame) -> pd.Series:
    """Hours a forecast can be judged on: not an outage day, and a count exists."""
    return ~grid["outage"] & grid["cnt"].notna()
