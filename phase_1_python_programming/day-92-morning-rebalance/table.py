"""Day 92 — one forecast reply becomes one staging row.

The row is what a morning crew reads: how many bikes to have out for
the commute, how that compares with a typical recent week, and which
wave to staff first. A refused day is a row of blanks. It is not a row
of zeros, because zero would mean "pull the fleet in."

The true count is not a field the server sends. If a reply contains
one anyway, the row is not built: displaying it would put the answer
next to the forecast.
"""

from __future__ import annotations

from datetime import date

from windows import WINDOWS, first_wave, window_baseline, window_bikes

# Day 86 dropped these. casual + registered is the target, and registered
# alone tracks it at 0.97. A reply that carries any of them is a leak.
_LEAK = ("cnt", "casual", "registered")

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def calendar_kind(day: str) -> tuple[str, str]:
    """Weekday name, and whether the calendar date is Monday–Friday.

    The forecast JSON does not carry the holiday flag. A weekday holiday
    is still labeled "working day" here. The staged counts come from the
    model, which did see the holiday, so the numbers are the holiday
    forecast. Only the label is coarse.
    """
    parsed = date.fromisoformat(day)
    name = _WEEKDAYS[parsed.weekday()]
    kind = "weekend" if parsed.weekday() >= 5 else "working day"
    return name, kind


def _hour_map(body: dict) -> dict:
    """Hours 0–23, each with the integer bikes and the baseline float."""
    if not isinstance(body, dict):
        raise ValueError("a forecast reply should be an object")
    leaked = [key for key in _LEAK if key in body]
    if leaked:
        raise ValueError(f"the reply includes {leaked}, which is the target, not a forecast")
    hours = body.get("hours")
    if not isinstance(hours, list) or len(hours) != 24:
        raise ValueError("a forecast reply needs 24 hours")
    by_hour = {}
    for hour in hours:
        if not isinstance(hour, dict):
            raise ValueError("each hour should be an object")
        # bool is an int subclass; True would otherwise stage as one bike
        hr = hour.get("hr")
        bikes = hour.get("bikes")
        baseline = hour.get("baseline")
        if type(hr) is not int or type(bikes) is not int:
            raise ValueError("hr and bikes should be integers")
        if isinstance(baseline, bool) or not isinstance(baseline, (int, float)):
            raise ValueError("baseline should be a number")
        by_hour[hr] = {"bikes": bikes, "baseline": float(baseline)}
    if sorted(by_hour) != list(range(24)):
        raise ValueError("hours should be 0 through 23 exactly once")
    # the day total is the sum of the integers, per Day 91; a mismatch
    # means the reply was edited and the window sums would not add up
    if body.get("bikes") != sum(row["bikes"] for row in by_hour.values()):
        raise ValueError("the day total does not match the 24 hourly bikes")
    return by_hour


def _blank_counts() -> dict:
    return {
        "morning_bikes": None,
        "midday_bikes": None,
        "evening_bikes": None,
        "quiet_bikes": None,
        "day_bikes": None,
        "morning_extra": None,
        "first_wave": None,
        "slice": None,
    }


def staging_row(body: dict) -> dict:
    """One served day. `morning_extra` is forecast minus the 4-week baseline.

    Positive means stage more bikes than the last four weeks of the same
    hours. Negative means some of that usual morning fleet can stay in.
    """
    day = body.get("date") if isinstance(body, dict) else None
    if not isinstance(day, str):
        raise ValueError("the reply has no date")
    by_hour = _hour_map(body)
    weekday, kind = calendar_kind(day)
    counts = {name: window_bikes(by_hour, hours) for name, hours in WINDOWS.items()}
    morning_baseline = window_baseline(by_hour, WINDOWS["morning"])
    return {
        "date": day,
        "weekday": weekday,
        "kind": kind,
        "status": "served",
        "http_status": 200,
        "reason": "",
        "morning_bikes": counts["morning"],
        "midday_bikes": counts["midday"],
        "evening_bikes": counts["evening"],
        "quiet_bikes": counts["quiet"],
        "day_bikes": body["bikes"],
        "morning_extra": counts["morning"] - morning_baseline,
        "first_wave": first_wave(counts),
        "slice": body.get("slice"),
    }


def refused_row(day: str, http_status: int, reason: str) -> dict:
    """A day the server declined. The bike fields stay None, not zero."""
    weekday, kind = calendar_kind(day)
    row = {
        "date": day,
        "weekday": weekday,
        "kind": kind,
        "status": "refused",
        "http_status": http_status,
        "reason": reason,
    }
    row.update(_blank_counts())
    return row
