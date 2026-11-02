"""Day 91 — turn one calendar day into the 24-hour reply, or refuse it.

The service holds three disjoint sets:

- days it can forecast (train and validation only)
- the closed test quarter, which is refused without a number attached
- days that are on record but not forecastable (outages, missing history)

A date is never in more than one of them. Sandy sits in the test quarter
and was also an outage; the closed-quarter answer wins, so the reply does
not describe test days at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from contract import ApiError, build_response, parse_date

# Day 88: a model trained on actual weather and fed a forecast lost to no
# weather at all. This server's model was trained without weather, so a
# weather body is a client error rather than something to ignore.
_WEATHER_REFUSAL = (
    "this model was trained without weather; a forecast sent here would be ignored, "
    "and feeding weather to a model that was not trained on it can lose to no weather at all"
)


@dataclass(frozen=True)
class DayRecord:
    slice_name: str
    hours: tuple


def _require_hours(hours) -> tuple:
    """Hours 0 through 23 in order, with finite non-negative counts."""
    rows = tuple(hours)
    if [row["hr"] for row in rows] != list(range(24)):
        raise ValueError("a served day needs hours 0 through 23 in order")
    for row in rows:
        for key in ("baseline", "expected"):
            value = float(row[key])
            if value < 0 or value != value or value in (float("inf"), float("-inf")):
                raise ValueError("baseline and expected counts have to be finite and non-negative")
    return rows


class ForecastService:
    def __init__(self, days: dict, closed, blocked: dict, model_name: str = "no weather", accepts_weather: bool = False, report: dict | None = None):
        self.model_name = model_name
        self.accepts_weather = accepts_weather
        self.report = dict(report or {})
        self.days = {}
        for day, record in days.items():
            parse_date(day)
            if record.slice_name not in ("train", "validation"):
                raise ValueError("only train and validation days can be served")
            self.days[day] = DayRecord(record.slice_name, _require_hours(record.hours))
        self.closed = frozenset(closed)
        self.blocked = dict(blocked)
        for day in self.closed | set(self.blocked):
            parse_date(day)
        overlap = set(self.days) & self.closed or set(self.days) & set(self.blocked) or self.closed & set(self.blocked)
        if overlap:
            raise ValueError("a date can't be served, closed, and blocked at once")

    def health(self) -> dict:
        return {
            "status": "ok",
            "model": self.model_name,
            "accepts_weather": self.accepts_weather,
            "served_days": len(self.days),
        }

    def forecast(self, day: str, weather):
        """Reply for one day.

        A bad weather body is rejected first, including on a closed date:
        that mistake is about the request, and it should not come back
        looking like a test-quarter refusal.
        """
        if weather is not None and not self.accepts_weather:
            raise ApiError(400, _WEATHER_REFUSAL)
        if weather is not None:
            # the flag is the only switch; there is no second model hiding behind it
            raise ApiError(500, "weather was accepted but no weather model is wired up")
        if day in self.closed:
            raise ApiError(403, f"{day} is in the closed test quarter and is not served")
        if day in self.blocked:
            raise ApiError(409, f"{day}: {self.blocked[day]}")
        record = self.days.get(day)
        if record is None:
            raise ApiError(404, f"no hours on record for {day}")
        return build_response(day, self.model_name, record.slice_name, record.hours)
