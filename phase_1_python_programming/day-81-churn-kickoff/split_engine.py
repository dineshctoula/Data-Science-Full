"""Day 81 — cut the churn table once and write the cut down.

The split is stratified on churn and saved as customerID -> slice in a
small TSV that is committed. Later days read that file instead of
re-splitting, so a change of seed, row order, or pandas version can't
move a customer from test into train.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churn_data import ID, TARGET

SLICES = ("train", "validation", "test")


def stratified_split(df: pd.DataFrame, fracs=(0.70, 0.15, 0.15), seed: int = 81) -> pd.Series:
    """Return a slice name per row, keeping the churn rate the same in each slice."""
    if len(fracs) != 3 or min(fracs) <= 0 or abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError("fracs should be three positive numbers that sum to 1")
    rng = np.random.default_rng(seed)
    labels = np.empty(len(df), dtype=object)
    y = df[TARGET].to_numpy()
    # split leavers and stayers separately, then each slice gets the same share of both
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n_train = int(round(len(idx) * fracs[0]))
        n_val = int(round(len(idx) * fracs[1]))
        labels[idx[:n_train]] = "train"
        labels[idx[n_train : n_train + n_val]] = "validation"
        labels[idx[n_train + n_val :]] = "test"
    return pd.Series(labels, index=df.index, name="slice")


def write_split(df: pd.DataFrame, slices: pd.Series, path) -> Path:
    path = Path(path)
    out = pd.DataFrame({ID: df[ID], "slice": slices}).sort_values(ID)
    # sorted by ID so the committed file diffs cleanly if it ever has to change
    out.to_csv(path, sep="\t", index=False, lineterminator="\n")
    return path


def load_split(df: pd.DataFrame, path) -> dict:
    """Read the frozen split and return {slice name: rows of df}.

    Refuses a file that doesn't cover exactly the customers in df, or that
    names a slice other than train / validation / test.
    """
    frozen = pd.read_csv(path, sep="\t", dtype=str)
    if list(frozen.columns) != [ID, "slice"]:
        raise ValueError("split file should have columns customerID and slice")
    if not frozen[ID].is_unique:
        raise ValueError("a customer appears twice in the split file")
    if set(frozen["slice"]) - set(SLICES):
        raise ValueError(f"unknown slice names: {sorted(set(frozen['slice']) - set(SLICES))}")
    if set(frozen[ID]) != set(df[ID]):
        raise ValueError("split file and table cover different customers")
    slice_of = frozen.set_index(ID)["slice"]
    names = df[ID].map(slice_of)
    return {name: df[names == name].reset_index(drop=True) for name in SLICES}
