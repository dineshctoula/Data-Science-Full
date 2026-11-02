"""Day 91 — what a forecast request may say, and what the reply looks like.

The server is a thin shell over this module. A bad date or a 23-hour
weather list fails here, before any model runs, and the JSON shape is
fixed so a caller can rely on the keys.

Weather is optional on purpose. Omitting it means "use the no-weather
model". Sending null is not the same thing: that is a mistake, and it
is refused rather than quietly treated as no weather.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from urllib.parse import parse_qs

# temp is degrees C / 41 and humidity is a fraction, the UCI scaling.
# A Celsius temperature fails this check instead of being read as impossible heat.
_UNIT = ("temp", "hum")
_WEATHER_KEYS = ("hr", "temp", "hum", "weathersit")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ApiError(Exception):
    """A refusal the client is allowed to see: status code plus one sentence."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def parse_date(value) -> str:
    """YYYY-MM-DD, zero-padded, and a real calendar day.

    fromisoformat on this Python also accepts '2012-8-7'. The reply echoes
    the date, so the spelling has to be the one callers will match on.
    """
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        raise ApiError(400, "date should be YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ApiError(400, "date should be a real calendar day") from None
    return value


def _unit_interval(value, name: str) -> float:
    # bool is an int subclass; True would otherwise become 1.0 and look like a real humidity
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(400, f"{name} should be a number from 0 to 1 on the UCI scale")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ApiError(400, f"{name} should be a number from 0 to 1 on the UCI scale")
    return number


def parse_weather(rows) -> list:
    """24 hours, each {hr, temp, hum, weathersit}, hours 0 through 23 once."""
    if not isinstance(rows, list) or len(rows) != 24:
        raise ApiError(400, "weather needs one row for each hour from 0 through 23")
    parsed = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(_WEATHER_KEYS):
            raise ApiError(400, "each weather row needs hr, temp, hum, and weathersit, and nothing else")
        hr = row["hr"]
        if isinstance(hr, bool) or type(hr) is not int or not 0 <= hr <= 23:
            raise ApiError(400, "hr should be an integer from 0 through 23")
        sit = row["weathersit"]
        if isinstance(sit, bool) or type(sit) is not int or sit not in (1, 2, 3, 4):
            raise ApiError(400, "weathersit should be 1, 2, 3, or 4")
        parsed.append(
            {
                "hr": hr,
                "temp": _unit_interval(row["temp"], "temp"),
                "hum": _unit_interval(row["hum"], "hum"),
                "weathersit": sit,
            }
        )
    if sorted(row["hr"] for row in parsed) != list(range(24)):
        raise ApiError(400, "weather hours should be 0 through 23 exactly once")
    parsed.sort(key=lambda row: row["hr"])
    return parsed


def parse_query(query: str) -> str:
    """The GET form: /forecast?date=YYYY-MM-DD and no other field."""
    parts = parse_qs(query, keep_blank_values=True)
    if "date" not in parts:
        raise ApiError(400, "date is required (YYYY-MM-DD)")
    if set(parts) != {"date"}:
        raise ApiError(400, "the only query field is date")
    if len(parts["date"]) != 1:
        raise ApiError(400, "pass date once")
    return parse_date(parts["date"][0])


def parse_body(raw: bytes):
    """The POST form. Returns (date, weather or None)."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ApiError(400, "body should be UTF-8 JSON") from None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise ApiError(400, "body should be a JSON object") from None
    if not isinstance(payload, dict):
        raise ApiError(400, "body should be a JSON object")
    if set(payload) - {"date", "weather"}:
        raise ApiError(400, "the only fields are date and weather")
    if "date" not in payload:
        raise ApiError(400, "date is required (YYYY-MM-DD)")
    if "weather" in payload and payload["weather"] is None:
        raise ApiError(400, "omit weather to use the no-weather model; null is not a forecast")
    weather = parse_weather(payload["weather"]) if "weather" in payload else None
    return parse_date(payload["date"]), weather


def whole_bikes(expected: float) -> int:
    """Bikes to stage. Half a bike rounds up (0.5 -> 1, 2.5 -> 3).

    The builtin round() goes to even, so 0.5 and 2.5 would become 0 and 2.
    Quiet hours land near those halves more often than rush hour does, and
    they would be shorted.
    """
    if not math.isfinite(expected) or expected < 0:
        raise ApiError(500, "a forecast came back negative or non-numeric")
    return int(expected + 0.5)


def build_response(date: str, model: str, slice_name: str, hours: list) -> dict:
    """One day, 24 hours, knowable at midnight at the start of `date`.

    as_of has no timezone. The UCI file is local time with no zone, and
    attaching UTC would slide the morning peak by several hours.

    expected is rounded to 4 decimals (finer than the 0.1 bike the MAE was
    reported at) so the JSON doesn't carry binary float noise. bikes is
    whole_bikes of that rounded value, so a client who recomputes it gets
    the same integer. The day total is the sum of those integers, which is
    what a dispatcher would add up; it can differ from rounding the sum
    of the floats by up to 12 bikes.
    """
    if [h["hr"] for h in hours] != list(range(24)):
        raise ApiError(500, "the forecast did not cover hours 0 through 23")
    out_hours = []
    for hour in hours:
        expected = round(float(hour["expected"]), 4)
        out_hours.append(
            {
                "hr": hour["hr"],
                "baseline": round(float(hour["baseline"]), 4),
                "expected": expected,
                "bikes": whole_bikes(expected),
            }
        )
    return {
        "date": date,
        "as_of": f"{date}T00:00:00",
        "model": model,
        "slice": slice_name,
        "hours": out_hours,
        "expected_total": round(sum(hour["expected"] for hour in out_hours), 4),
        "bikes": sum(hour["bikes"] for hour in out_hours),
    }
