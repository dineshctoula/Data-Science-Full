"""Day 91 — fit the no-weather model and keep only the hours it may serve.

The fit is Day 87's ridge on the log-ratio to the 4-week baseline, penalty
0.1, the value rolling CV picked and Day 89 froze. It is not retuned here.

Train rows are the only rows that are fit. Validation is what the server
answers with: fitting on August would let a request for August be scored
by a model that has already seen August.

Test rows are never predicted. assemble() drops them even if a caller
hands numbers in, and the frame it builds has no true-count column, so
the object the server holds cannot leak one.

The day after an outage has no "yesterday", so the drift feature is
missing and that day is refused. Day 89's fallback (use the last
non-outage day) belongs to the locked test recipe and is not reused here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import project_paths  # noqa: F401  (puts Days 86 and 87 on the path)
from bike_data import TARGET, load_grid
from contract import parse_date
from correction_model import CorrectionRidge
from evaluate_engine import usable
from features import design, offset
from service import DayRecord, ForecastService
from split_engine import assign_slices, load_split_dates

MODEL_NAME = "no weather"
# Day 87's middle rung: hour shape, holiday, and drift. No weather columns.
RUNG = "+ holiday, drift (no weather)"
LAM = 0.1

_OUTAGE = "that day was an outage, so its counts were never a forecast target"
_SHORT = "the 4-week baseline or the midnight drift feature is missing for at least one hour"


def _day_key(value) -> str:
    return parse_date(pd.Timestamp(value).strftime("%Y-%m-%d"))


def assemble(frame: pd.DataFrame, report: dict | None = None) -> ForecastService:
    """Split a frame of hours into served days, the closed quarter, and refusals.

    Required columns: date, hr, slice, outage, baseline, expected, eligible.
    True counts are not a column. `expected` on a test row is ignored.
    """
    missing = {"date", "hr", "slice", "outage", "baseline", "expected", "eligible"} - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing {sorted(missing)}")

    days, closed, blocked = {}, set(), {}
    keyed = frame.copy()
    keyed["_day"] = keyed["date"].map(_day_key)
    for day, group in keyed.groupby("_day", sort=True):
        labels = set(group["slice"])
        if labels == {"test"}:
            closed.add(day)
            continue
        if "test" in labels:
            raise ValueError(f"{day} mixes the test quarter with another slice")
        if bool(group["outage"].any()):
            # whole day, even if a few hours happened to look eligible
            blocked[day] = _OUTAGE
            continue
        ready = group[group["eligible"].astype(bool)]
        if len(group) != 24 or len(ready) != 24:
            blocked[day] = _SHORT
            continue
        ready = ready.sort_values("hr")
        slice_name = str(ready["slice"].iloc[0])
        if set(ready["slice"]) != {slice_name}:
            raise ValueError(f"{day} has more than one slice")
        hours = tuple(
            {"hr": int(row.hr), "baseline": float(row.baseline), "expected": float(row.expected)}
            for row in ready.itertuples(index=False)
        )
        days[day] = DayRecord(slice_name, hours)
    return ForecastService(days, closed, blocked, model_name=MODEL_NAME, report=report)


def prepare(grid: pd.DataFrame | None = None, slices: pd.Series | None = None):
    """Fit on train and return (service, report). Test hours are not predicted."""
    if grid is None:
        grid = load_grid(project_paths.DATA_DIR)
    if slices is None:
        slices = assign_slices(grid, load_split_dates(project_paths.SPLIT_FILE))
    # masks below are positional; a non-range index would let .loc and
    # numpy disagree about which hour is which
    if not grid.index.equals(pd.RangeIndex(len(grid))):
        raise ValueError("grid index should be a fresh range")

    features = design(grid, RUNG)
    base = offset(grid)
    ok = usable(grid, features, base)
    train = ok & (slices == "train")
    model = CorrectionRidge(LAM).fit(features.loc[train], grid.loc[train, TARGET], base.loc[train])

    # eligible stops at validation; the test mask is not in this index
    eligible = ok & slices.isin(("train", "validation"))
    expected = np.full(len(grid), np.nan)
    expected[eligible.to_numpy()] = model.predict(features.loc[eligible], base.loc[eligible])

    # positional masks: the grid index is a reset range, and expected is
    # built in that same row order, so .loc isn't needed and can't reorder
    val = (eligible & (slices == "validation")).to_numpy()
    y = grid[TARGET].to_numpy()[val]
    report = {
        "model": MODEL_NAME,
        "rung": RUNG,
        "lam": LAM,
        "fit_on": "train",
        "n_fit_hours": int(train.sum()),
        "n_validation_hours": int(val.sum()),
        "validation_mae": float(np.mean(np.abs(y - expected[val]))),
        "validation_baseline_mae": float(np.mean(np.abs(y - base.to_numpy()[val]))),
    }
    frame = pd.DataFrame(
        {
            "date": grid["date"].to_numpy(),
            "hr": grid["hr"].to_numpy(),
            "slice": slices.to_numpy(),
            "outage": grid["outage"].to_numpy(),
            "baseline": base.to_numpy(),
            "expected": expected,
            "eligible": eligible.to_numpy(),
        }
    )
    service = assemble(frame, report)
    # counted after assemble, which is what actually decides a day is servable
    service.report["n_served_days"] = len(service.days)
    service.report["n_closed_days"] = len(service.closed)
    return service
