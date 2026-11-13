"""Day 93 — one staged morning against the rides that happened.

The crew staged `morning_bikes` before 7. The score is how many of
those bikes the commute used. Positive error means bikes sat unused;
negative error means the rush was short.

A refused staging row is not scored. The actual is not looked up first:
looking it up on a closed date would be reading the test quarter in
order to decide not to use it.

The 4-week baseline is recovered from the staging row
(`morning_bikes - morning_extra`). That is the same integer Day 92
printed, so the baseline MAE uses the number the crew would have
staged if they had ignored the model.
"""

from __future__ import annotations

import paths  # noqa: F401

from actuals import ClosedQuarter, morning_actual
from week import CLOSED_FROM


def _blank_score(row: dict, reason: str) -> dict:
    return {
        "date": row["date"],
        "weekday": row.get("weekday"),
        "kind": row.get("kind"),
        "status": "unscored",
        "reason": reason,
        "morning_staged": None,
        "morning_actual": None,
        "morning_error": None,
        "abs_error": None,
        "morning_baseline": None,
        "baseline_abs_error": None,
        "morning_extra": None,
        "slice": row.get("slice"),
    }


def score_row(row: dict, actuals: dict) -> dict:
    """Join one Day 92 staging row to the morning that happened.

    Closed dates raise even when the staging row is already blank. The
    walkthrough still has to prove it did not score October, and a raise
    is louder than a dash.
    """
    day = row.get("date")
    if not isinstance(day, str):
        raise ValueError("a staging row needs a date")
    if day >= CLOSED_FROM:
        raise ClosedQuarter(f"{day} is in the closed test quarter and is not scored")
    if row.get("status") != "served":
        return _blank_score(row, row.get("reason") or "that day was not served")
    staged = row.get("morning_bikes")
    extra = row.get("morning_extra")
    if type(staged) is not int or type(extra) is not int:
        raise ValueError("morning_bikes and morning_extra should be integers on a served day")
    truth = morning_actual(actuals, day)
    actual = truth["morning"]
    baseline = staged - extra
    error = staged - actual
    return {
        "date": day,
        "weekday": row.get("weekday"),
        "kind": row.get("kind"),
        "status": "scored",
        "reason": "",
        "morning_staged": staged,
        "morning_actual": actual,
        "morning_error": error,
        "abs_error": abs(error),
        "morning_baseline": baseline,
        "baseline_abs_error": abs(baseline - actual),
        "morning_extra": extra,
        "slice": truth["slice"],
    }
