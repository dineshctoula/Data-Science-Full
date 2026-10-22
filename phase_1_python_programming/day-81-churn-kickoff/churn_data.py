"""Day 81 — fetch and clean the Telco customer-churn table.

The CSV comes from IBM's public sample repo (Apache-2.0). It is cached
under data/ and never committed (the repo ignores *.csv), so the hash
below is what pins the exact file every later day works on.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
FILENAME = "Telco-Customer-Churn.csv"

NUMERIC = ("tenure", "MonthlyCharges", "TotalCharges")
# SeniorCitizen arrives as 0/1 but is a yes/no flag like Partner, so it sits here
CATEGORICAL = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)
ID, TARGET = "customerID", "Churn"

# the CSV's own header order, so a reordered or renamed file is caught
RAW_COLUMNS = (
    ID, "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", TARGET,
)


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(cache_dir="data", url: str = URL, expected_sha: str = SHA256) -> Path:
    """Return the cached CSV, downloading it first if it isn't there yet.

    A file with the wrong hash is refused rather than silently used: a
    changed upstream file would quietly change every number downstream.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / FILENAME
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        # download to a temp name so a broken transfer never looks like a cached file
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    got = sha256_of(path)
    if got != expected_sha:
        raise ValueError(f"{path} has sha256 {got[:12]}…, expected {expected_sha[:12]}…; delete it and re-fetch")
    return path


def load_raw(path) -> pd.DataFrame:
    """Read every column as text, exactly as stored, and check the header."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(df.columns) != RAW_COLUMNS:
        raise ValueError("unexpected columns; is this the Telco churn file?")
    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Typed copy of the raw table. Every change here is listed in PROJECT.md.

    - TotalCharges is blank for 11 customers, all with tenure 0. They
      haven't been billed yet, so the honest value is 0, not a median.
    - SeniorCitizen 0/1 becomes No/Yes to match the other flags.
    - Churn Yes/No becomes 1/0.
    """
    df = raw.copy()
    for col in ("tenure", "MonthlyCharges"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    total = df["TotalCharges"].str.strip()
    blank = total == ""
    # only a brand-new customer is allowed a blank bill; anything else is a real gap
    if (df.loc[blank, "tenure"] != 0).any():
        raise ValueError("TotalCharges is blank for a customer with tenure > 0")
    df["TotalCharges"] = pd.to_numeric(total.where(~blank, "0"), errors="raise")

    df["SeniorCitizen"] = df["SeniorCitizen"].map({"0": "No", "1": "Yes"})
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    if df[["SeniorCitizen", TARGET]].isna().any().any():
        raise ValueError("SeniorCitizen or Churn has a value outside the expected two")
    if not df[ID].is_unique:
        raise ValueError("customerID repeats")
    df[TARGET] = df[TARGET].astype(int)
    return df


def load_clean(cache_dir="data") -> pd.DataFrame:
    return clean(load_raw(fetch(cache_dir)))
