"""Morning counts that happened, for days the model may already have seen.

The test quarter is dropped before any window is summed. A lookup for
an October date fails because the date is closed, not because the true
count was read and then hidden. Putting the answer in the table and
then refusing to return it would still be reading test.

Only hours 7–9 are kept. This day scores the morning the crew staged.
It does not need the rest of the day's rides, and keeping them would
make it easy to "just also" score evening on a closed date later.
"""

from __future__ import annotations

import paths  # noqa: F401  (Day 92, then Day 91 / Day 86)

from bike_data import TARGET, load_grid
from contract import parse_date
from split_engine import assign_slices, load_split_dates
from week import CLOSED_FROM
from windows import MORNING

import pandas as pd
import project_paths


class ClosedQuarter(ValueError):
    """Asked to score a date on or after 2012-10-01."""


def _day_key(value) -> str:
    return parse_date(pd.Timestamp(value).strftime("%Y-%m-%d"))


def _whole_count(value) -> int:
    """A recorded rental count is a whole number of bikes.

    Quiet gaps were filled with 0 on Day 86. An outage hour is NaN and
    is not a score: that day's morning never happened as demand.
    """
    if value is None or value != value:
        raise ValueError("a morning hour has no count")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("a morning count should be a number")
    number = float(value)
    if number < 0 or number != int(number):
        raise ValueError("a morning count should be a whole number of bikes")
    return int(number)


def open_mornings_from_grid(grid: pd.DataFrame, slices: pd.Series) -> dict:
    """Train and validation mornings only. Test hours are not copied in.

    Required columns: date, hr, and the Day 86 target. Extra columns are
    ignored, including casual/registered, so a leaky column in the grid
    cannot become a field the score prints.
    """
    missing = {"date", "hr", TARGET} - set(grid.columns)
    if missing:
        raise ValueError(f"grid is missing {sorted(missing)}")
    if len(grid) != len(slices):
        raise ValueError("slices should line up with the grid rows")

    # positional: a non-aligned index would keep a test hour whose label
    # sat on a different row than its count
    open_mask = slices.isin(("train", "validation")).to_numpy()
    dates = pd.Series(grid["date"].to_numpy()[open_mask])
    hours = pd.Series(grid["hr"].to_numpy()[open_mask])
    counts = pd.Series(grid[TARGET].to_numpy()[open_mask])
    labels = pd.Series(slices.to_numpy()[open_mask])

    table = {}
    keyed = pd.DataFrame(
        {
            "day": dates.map(_day_key),
            "hr": hours.astype(int),
            "cnt": counts,
            "slice": labels,
        }
    )
    morning = keyed[keyed["hr"].isin(MORNING)]
    for day, group in morning.groupby("day", sort=True):
        if day >= CLOSED_FROM:
            # belt: a mislabeled test hour still does not enter the table
            continue
        labels_in_day = set(group["slice"])
        if labels_in_day == {"test"}:
            continue
        if "test" in labels_in_day:
            raise ValueError(f"{day} mixes the test quarter with another slice")
        by_hour = {}
        for row in group.itertuples(index=False):
            hr = int(row.hr)
            if hr in by_hour:
                raise ValueError(f"{day} hour {hr} appears twice")
            by_hour[hr] = _whole_count(row.cnt)
        if sorted(by_hour) != list(MORNING):
            # a partial morning is not a number the crew can be judged on
            continue
        slice_name = str(group["slice"].iloc[0])
        table[day] = {
            "slice": slice_name,
            "hours": by_hour,
            "morning": sum(by_hour[hr] for hr in MORNING),
        }
    return table


def load_open_mornings(grid: pd.DataFrame | None = None, slices: pd.Series | None = None) -> dict:
    """Read the Day 86 grid, then drop the test quarter before summing."""
    if grid is None:
        grid = load_grid(project_paths.DATA_DIR)
    if slices is None:
        slices = assign_slices(grid, load_split_dates(project_paths.SPLIT_FILE))
    return open_mornings_from_grid(grid, slices)


def morning_actual(table: dict, day: str) -> dict:
    """The three morning hours for an open date.

    The closed-quarter check runs before the dict lookup. A test date
    stuffed into `table` by a caller is still not returned.
    """
    day = parse_date(day)
    if day >= CLOSED_FROM:
        raise ClosedQuarter(f"{day} is in the closed test quarter and is not scored")
    record = table.get(day)
    if record is None:
        raise KeyError(f"no open morning counts for {day}")
    return record
