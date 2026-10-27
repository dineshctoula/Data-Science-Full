"""Day 86 — fetch the UCI Bike Sharing data and rebuild a complete hourly grid.

Source: UCI Machine Learning Repository, dataset 275 (CC BY 4.0).
Citation asked for by the authors: Fanaee-T, H. and Gama, J., "Event
labeling combining ensemble detectors and background knowledge",
Progress in Artificial Intelligence (2013), doi:10.1007/s13748-013-0040-3.

hour.csv has 17379 of the 17544 hours in 2011-2012. No recorded hour has
a count of 0, so the gaps are of two kinds:
- scattered quiet hours (mostly 3-4 a.m.) with no rentals at all, which
  the file simply leaves out -> filled with 0
- whole stretches on days the system was down (snowstorms, Hurricane
  Irene, Hurricane Sandy) -> the entire day is flagged as an outage, its
  counts set to NaN, and it is never scored or used as history
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
ZIP_NAME = "bike_sharing.zip"
TARGET = "cnt"
# casual + registered == cnt on every row: they ARE the target, split two ways
LEAKY = ("casual", "registered")
# a day missing this many hours or more is treated as a shutdown, not a quiet night
OUTAGE_MIN_MISSING = 6

RAW_COLUMNS = (
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt",
)


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(cache_dir="data", url: str = URL, expected_sha: str = SHA256) -> Path:
    """Return the cached zip, downloading it once. A wrong hash is refused."""
    cache_dir = Path(cache_dir)
    path = cache_dir / ZIP_NAME
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        # the repo ignores *.csv but not *.zip, so the cache folder ignores itself
        (cache_dir / ".gitignore").write_text("*\n")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    got = sha256_of(path)
    if got != expected_sha:
        raise ValueError(f"{path} has sha256 {got[:12]}…, expected {expected_sha[:12]}…; delete it and re-fetch")
    return path


def load_raw(zip_path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(io.BytesIO(z.read("hour.csv")))
    if tuple(df.columns) != RAW_COLUMNS:
        raise ValueError("unexpected columns; is this the UCI bike sharing hour.csv?")
    return df


def complete_grid(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per hour from the first to the last day, with gaps labelled.

    Adds `ts` (timestamp), `date`, `missing` (hour absent from the file),
    and `outage` (the whole day is a shutdown day). Calendar columns are
    rebuilt from `ts` for every row, so filled rows aren't left blank.
    """
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    if not df["ts"].is_unique:
        raise ValueError("an hour appears twice")
    full = pd.date_range(df["ts"].min().normalize(), df["ts"].max().normalize() + pd.Timedelta(hours=23), freq="h")
    grid = df.set_index("ts").reindex(full)
    grid.index.name = "ts"
    grid = grid.reset_index()
    grid["missing"] = grid["instant"].isna()
    grid["date"] = grid["ts"].dt.normalize()

    missing_per_day = grid.groupby("date")["missing"].transform("sum")
    grid["outage"] = missing_per_day >= OUTAGE_MIN_MISSING

    # calendar facts follow from the timestamp, so they can be filled for every hour
    grid["hr"] = grid["ts"].dt.hour
    grid["mnth"] = grid["ts"].dt.month
    grid["yr"] = grid["ts"].dt.year - 2011
    # the file's weekday is 0 = Sunday; pandas' is 0 = Monday
    grid["weekday"] = (grid["ts"].dt.dayofweek + 1) % 7
    # holiday / workingday / season are per day: copy them from that day's recorded hours
    for col in ("holiday", "workingday", "season"):
        grid[col] = grid.groupby("date")[col].transform(lambda s: s.ffill().bfill())

    # weather drifts smoothly hour to hour, so a gap takes its neighbours' values:
    # straight lines for the measurements, the previous hour's category for weathersit
    for col in ("temp", "atemp", "hum", "windspeed"):
        grid[col] = grid[col].interpolate(limit_direction="both")
    grid["weathersit"] = grid["weathersit"].ffill().bfill()

    quiet = grid["missing"] & ~grid["outage"]
    for col in ("cnt", "casual", "registered"):
        grid.loc[quiet, col] = 0
        # an outage day's counts describe a shutdown, not demand
        grid.loc[grid["outage"], col] = np.nan
    return grid.drop(columns=["instant", "dteday"])


def load_grid(cache_dir="data") -> pd.DataFrame:
    return complete_grid(load_raw(fetch(cache_dir)))
