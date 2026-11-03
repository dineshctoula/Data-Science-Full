"""Day 92 — the hours a morning crew stages for.

Day 86's shapes: a working day peaks near 8 a.m. and 5 p.m., and a
weekend has one mid-afternoon hump. Day 87 called 7–9 and 16–19 the
peak hours. These windows use those same clocks. They are not re-fit
on this week's forecasts, because a window that moved after seeing the
week would no longer be the one the crew planned for.

The UCI file is one citywide count per hour, not a dock. A window says
when the fleet has to be out. It does not say which station a bike
should sit at, and this day does not invent station ids to pretend it can.
"""

from __future__ import annotations

import project_paths  # noqa: F401  (puts Day 91's contract on the path)
from contract import ApiError, whole_bikes

# Inbound commute. Staged before 7, because the rush is already happening at 8.
MORNING = (7, 8, 9)
# Day 87's evening peak runs through 7 p.m., not just the 5 p.m. hour.
EVENING = (16, 17, 18, 19)
# The weekend hump sits between the two commute peaks.
MIDDAY = (11, 12, 13, 14, 15)
# Overnight. These bikes can stay in the yard; dawn is not spent staging them.
QUIET = (0, 1, 2, 3, 4, 5)

# Insertion order is the tie-break order in first_wave: earlier clock wins.
WINDOWS = {
    "morning": MORNING,
    "midday": MIDDAY,
    "evening": EVENING,
    "quiet": QUIET,
}

# 6 is the ramp, 10 is the lull between peaks, 20–23 are the tail of the
# ride home. They belong in the day total and in none of the windows, so
# adding the four window columns will not reproduce the day total.
SHOULDERS = (6, 10, 20, 21, 22, 23)


def window_bikes(by_hour: dict, hours: tuple) -> int:
    """Sum of the server's integer bikes over `hours`.

    The floats are not rounded again here. Day 91 already turned each
    hour's expected count into `bikes`, and a second rounding would not
    match the column a caller of that API adds up.
    """
    return sum(by_hour[hr]["bikes"] for hr in hours)


def window_baseline(by_hour: dict, hours: tuple) -> int:
    """The 4-week baseline, staged the same way as the forecast.

    Half a bike rounds up on each hour, then the hours are added. Rounding
    the window's float sum instead would disagree with Day 91's day total
    by up to one bike per hour in the window.
    """
    try:
        return sum(whole_bikes(by_hour[hr]["baseline"]) for hr in hours)
    except ApiError as exc:
        raise ValueError(str(exc)) from None


def first_wave(counts: dict) -> str:
    """Which staffed window has the most bikes per hour.

    Midday is five hours and morning is three, so the raw sums would
    crown midday on a flat day. The rate is the rush. Quiet hours are
    not a candidate: a full yard at 3 a.m. is not a wave to staff.

    A tie goes to the earlier window, the one that has to be ready first.
    """
    best = None
    best_rate = None
    for name in ("morning", "midday", "evening"):
        rate = counts[name] / len(WINDOWS[name])
        if best_rate is None or rate > best_rate:
            best = name
            best_rate = rate
    return best
