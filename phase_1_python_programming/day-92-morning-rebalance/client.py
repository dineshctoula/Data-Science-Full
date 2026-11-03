"""Day 92 — ask Day 91 for one day. This module does not forecast.

The only request is GET /forecast?date=YYYY-MM-DD. A POST is how a
weather body would be sent, and this client has no weather to send.
Day 91 would refuse it anyway; not sending it keeps the refusal from
being the way the week learns it asked for the wrong thing.

A 403, 404, or 409 is an answer about that date: closed, unknown, or
blocked. It comes back as a result so the week table can leave the row
blank. A 400 means the request itself was wrong. A 5xx or a dead
server is not a property of the date, so those raise instead of
becoming a blank row.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import project_paths  # noqa: F401  (puts Day 91's contract on the path)
from contract import parse_date

# dates the server has decided not to forecast; the week still prints them
_REFUSALS = frozenset({403, 404, 409})


class ClientError(Exception):
    """The server could not be asked, or the failure was not about one date."""


class ForecastClient:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        """GET /health. A failure here raises: there is no week to print."""
        payload, status = self._read("/health")
        if status != 200:
            raise ClientError(f"HTTP {status} from /health")
        return payload

    def fetch(self, day: str) -> dict:
        """One day. Returns status, error, and the JSON body or None.

        The date is checked locally first, so a bad spelling fails before
        a socket opens. The server checks it again.
        """
        day = parse_date(day)
        payload, status = self._read(f"/forecast?date={day}")
        if status == 200:
            return {"date": day, "status": 200, "error": "", "body": payload}
        if status in _REFUSALS:
            error = payload.get("error", "") if isinstance(payload, dict) else ""
            return {"date": day, "status": status, "error": error, "body": None}
        message = payload.get("error", "") if isinstance(payload, dict) else ""
        raise ClientError(f"{day}: HTTP {status}: {message}")

    def _read(self, path: str):
        # method is set so a library default cannot turn this into a POST
        request = urllib.request.Request(self.base_url + path, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return self._decode(response.read()), response.status
        except urllib.error.HTTPError as exc:
            return self._decode(exc.read()), exc.code
        except urllib.error.URLError as exc:
            raise ClientError(f"the forecast server is not reachable at {self.base_url}") from exc

    @staticmethod
    def _decode(raw: bytes) -> dict:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ClientError("the forecast server did not return JSON") from None
        if not isinstance(payload, dict):
            raise ClientError("the forecast server did not return a JSON object")
        return payload
