"""Day 92 — seven dates, asked one at a time, printed as one table.

The week is seven calendar days. It is not seven days the server has
agreed to forecast. A date inside the test quarter is still requested,
and the row stays blank when the server refuses it. Skipping those
dates locally would hide a server that had started answering them.

Refused days are left out of the totals. Adding them as zero would
shrink the morning fleet by the number of days the model declined.
"""

from __future__ import annotations

from datetime import date, timedelta

import server_path  # noqa: F401  (puts Day 91 on the path without taking its module name)
from contract import parse_date
from table import refused_row, staging_row

# Monday of the week that contains Day 91's sample Tuesday, 2012-08-07.
# The whole week sits inside validation (2012-07-01 .. 2012-09-30).
SAMPLE_START = "2012-08-06"
# Friday 28 Sep through Thursday 4 Oct: the last validation days, then
# the closed quarter. Chosen so the walkthrough has to show a refusal.
CROSSING_START = "2012-09-28"
# Day 86 froze the test quarter here. String order matches calendar order
# for these zero-padded dates.
CLOSED_FROM = "2012-10-01"

_COLUMNS = (
    ("date", 12, False),
    ("weekday", 11, False),
    ("kind", 14, False),
    ("http", 6, False),
    ("morning", 8, False),
    ("extra", 7, True),
    ("midday", 8, False),
    ("evening", 8, False),
    ("day", 7, False),
)


def week_dates(start: str) -> list[str]:
    """Seven calendar days from `start`, inclusive."""
    first = date.fromisoformat(parse_date(start))
    return [(first + timedelta(days=offset)).isoformat() for offset in range(7)]


def build_week(client, start: str) -> list[dict]:
    """Ask for each day. A served body whose date is not the one asked for is rejected.

    Cache-Control on the server is no-store, and the client still checks
    the echo. A reused body would stage Tuesday's fleet on Wednesday.
    """
    rows = []
    for day in week_dates(start):
        reply = client.fetch(day)
        if reply["status"] == 200:
            row = staging_row(reply["body"])
            if row["date"] != day:
                raise ValueError(f"asked for {day} and the reply was for {row['date']}")
            rows.append(row)
        else:
            rows.append(refused_row(day, reply["status"], reply["error"]))
    return rows


def closed_quarter_is_blank(rows) -> list[str]:
    """Dates on or after 2012-10-01 that came back with a bike count.

    Empty means the test quarter contributed nothing to the table.
    """
    leaked = []
    for row in rows:
        if row["date"] < CLOSED_FROM:
            continue
        if row["status"] == "served" or row["morning_bikes"] is not None or row["day_bikes"] is not None:
            leaked.append(row["date"])
    return leaked


def summarize(rows) -> dict:
    """Totals over served days only."""
    served = [row for row in rows if row["status"] == "served"]
    return {
        "n_days": len(rows),
        "n_served": len(served),
        "n_refused": len(rows) - len(served),
        "morning_bikes": sum(row["morning_bikes"] for row in served),
        "morning_extra": sum(row["morning_extra"] for row in served),
    }


def _cell(value, width: int, signed: bool) -> str:
    if value is None:
        text = "-"
    elif signed:
        text = f"{value:+d}"
    else:
        text = str(value)
    return f"{text:<{width}}" if width > 8 else f"{text:>{width}}"


def render(rows) -> str:
    """One line per day. A refusal is a dash, and its reason sits underneath.

    The four windows do not add up to the day column: shoulder hours
    (6, 10, and 20–23) are in the day total and in no window.
    """
    header_names = {
        "date": "date",
        "weekday": "weekday",
        "kind": "kind",
        "http": "http",
        "morning": "morning",
        "extra": "extra",
        "midday": "midday",
        "evening": "evening",
        "day": "day",
    }
    header = ""
    for key, width, _signed in _COLUMNS:
        piece = header_names[key]
        header += f"{piece:<{width}}" if width > 8 else f"{piece:>{width}}"
    header += "  first wave"
    lines = [header]
    for row in rows:
        values = {
            "date": row["date"],
            "weekday": row["weekday"],
            "kind": row["kind"],
            "http": row["http_status"],
            "morning": row["morning_bikes"],
            "extra": row["morning_extra"],
            "midday": row["midday_bikes"],
            "evening": row["evening_bikes"],
            "day": row["day_bikes"],
        }
        line = ""
        for key, width, signed in _COLUMNS:
            line += _cell(values[key], width, signed)
        wave = row["first_wave"] if row["first_wave"] else "-"
        line += f"  {wave}"
        lines.append(line)
        if row["status"] == "refused" and row["reason"]:
            lines.append(f"{'':12}{row['reason']}")
    totals = summarize(rows)
    lines.append(
        f"served {totals['n_served']} of {totals['n_days']}; "
        f"morning bikes {totals['morning_bikes']} "
        f"({totals['morning_extra']:+d} vs the 4-week baseline). "
        "Refused days are blank, not zero."
    )
    return "\n".join(lines)
