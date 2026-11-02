"""Day 91 — the request, the refusals, and one live HTTP round trip.

The fitted-model checks load the bike grid once. Everything else uses a
three-day table so a contract bug fails in milliseconds.
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

import pandas as pd

from contract import ApiError, build_response, parse_body, parse_date, parse_query, parse_weather, whole_bikes
from fitted import _OUTAGE, _SHORT, assemble, prepare
from http_api import MAX_BODY, bind
from service import DayRecord, ForecastService


def _hours(scale=1.0):
    return [{"hr": hr, "baseline": 10.0 + hr, "expected": scale * (12.0 + hr)} for hr in range(24)]


def _weather():
    return [{"hr": hr, "temp": 0.5, "hum": 0.4, "weathersit": 1} for hr in range(24)]


def _service():
    return ForecastService(
        days={"2012-08-07": DayRecord("validation", _hours())},
        closed=["2012-10-01"],
        blocked={"2011-08-28": _OUTAGE, "2011-01-01": _SHORT},
        model_name="no weather",
    )


def _frame(days):
    """days: list of (date, slice, outage, eligible, expected). One flag covers all 24 hours."""
    rows = []
    for day, slice_name, outage, eligible, expected in days:
        for hr in range(24):
            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "hr": hr,
                    "slice": slice_name,
                    "outage": outage,
                    "baseline": 10.0 + hr,
                    "expected": expected,
                    "eligible": eligible,
                }
            )
    return pd.DataFrame(rows)


def _request(method, url, body=None, headers=None):
    data = None if body is None else body if isinstance(body, bytes) else body.encode()
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode())
            return resp.status, payload, resp.headers
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode())
        return exc.code, payload, exc.headers


class ContractTests(unittest.TestCase):
    def test_dates_are_real_and_zero_padded(self):
        self.assertEqual(parse_date("2012-08-07"), "2012-08-07")
        for bad in ("2012-8-7", "2012-02-31", "08-07-2012", 20120807, "2012-08-07T00:00:00"):
            with self.assertRaises(ApiError) as caught:
                parse_date(bad)
            self.assertEqual(caught.exception.status, 400)

    def test_half_a_bike_rounds_up(self):
        self.assertEqual(whole_bikes(0.5), 1)
        self.assertEqual(whole_bikes(2.5), 3)
        self.assertEqual(whole_bikes(2.49), 2)
        with self.assertRaises(ApiError):
            whole_bikes(float("nan"))

    def test_the_day_total_is_the_sum_of_the_staged_bikes(self):
        # 0.5 at every hour rounds up to 1, so the integer total is 24
        # and the float total stays 12; a dispatcher adding the column gets 24
        body = build_response(
            "2012-08-07",
            "no weather",
            "validation",
            [{"hr": hr, "baseline": 1.0, "expected": 0.5} for hr in range(24)],
        )
        self.assertEqual(body["as_of"], "2012-08-07T00:00:00")
        self.assertEqual([hour["bikes"] for hour in body["hours"]], [1] * 24)
        self.assertEqual(body["bikes"], 24)
        self.assertEqual(body["expected_total"], 12.0)
        self.assertNotIn("cnt", json.dumps(body))

    def test_weather_is_twenty_four_uci_scaled_hours_or_absent(self):
        self.assertEqual(parse_weather(_weather())[0]["hr"], 0)
        self.assertIsNone(parse_body(b'{"date": "2012-08-07"}')[1])
        bad_bodies = [
            b'{"date": "2012-08-07", "weather": null}',
            b'{"date": "2012-08-07", "weather": []}',
            b'{"when": "2012-08-07"}',
            b'"2012-08-07"',
            b"\xff",
        ]
        for raw in bad_bodies:
            with self.assertRaises(ApiError):
                parse_body(raw)
        swapped = _weather()
        swapped[0]["hr"] = 1  # two hour-1 rows, no hour 0
        with self.assertRaises(ApiError):
            parse_weather(swapped)
        hot = _weather()
        hot[3]["temp"] = 25  # Celsius, not the 0-1 UCI scale
        with self.assertRaises(ApiError):
            parse_weather(hot)

    def test_the_query_carries_date_once(self):
        self.assertEqual(parse_query("date=2012-08-07"), "2012-08-07")
        for query in ("", "date=2012-08-07&hour=8", "date=2012-08-07&date=2012-08-08", "date="):
            with self.assertRaises(ApiError):
                parse_query(query)


class ServiceTests(unittest.TestCase):
    def test_a_validation_day_is_served_and_the_gates_hold(self):
        service = _service()
        body = service.forecast("2012-08-07", None)
        self.assertEqual(body["slice"], "validation")
        self.assertEqual(len(body["hours"]), 24)
        self.assertEqual(body["model"], "no weather")
        # rush hour is above the small hours on this table: expected grows with hr
        self.assertGreater(body["hours"][8]["bikes"], body["hours"][3]["bikes"])

        with self.assertRaises(ApiError) as closed:
            service.forecast("2012-10-01", None)
        self.assertEqual(closed.exception.status, 403)

        with self.assertRaises(ApiError) as outage:
            service.forecast("2011-08-28", None)
        self.assertEqual(outage.exception.status, 409)
        self.assertIn("outage", str(outage.exception))

        with self.assertRaises(ApiError) as missing:
            service.forecast("2013-01-01", None)
        self.assertEqual(missing.exception.status, 404)

    def test_weather_is_refused_even_for_a_closed_date(self):
        service = _service()
        # the request is wrong before the date is: a test day with weather is still a 400
        with self.assertRaises(ApiError) as caught:
            service.forecast("2012-10-01", _weather())
        self.assertEqual(caught.exception.status, 400)
        self.assertIn("trained without weather", str(caught.exception))

    def test_accepting_weather_without_a_model_fails_loudly(self):
        service = ForecastService(
            days={"2012-08-07": DayRecord("validation", _hours())},
            closed=[],
            blocked={},
            accepts_weather=True,
        )
        with self.assertRaises(ApiError) as caught:
            service.forecast("2012-08-07", _weather())
        self.assertEqual(caught.exception.status, 500)

    def test_a_date_cannot_be_served_and_closed(self):
        with self.assertRaises(ValueError):
            ForecastService(
                days={"2012-10-01": DayRecord("validation", _hours())},
                closed=["2012-10-01"],
                blocked={},
            )


class AssembleTests(unittest.TestCase):
    def test_test_numbers_are_dropped_and_outages_are_not_described_as_forecasts(self):
        frame = _frame(
            [
                ("2012-08-07", "validation", False, True, 40.0),
                ("2012-10-01", "test", False, True, 999.0),
                ("2012-10-29", "test", True, False, 999.0),
                ("2011-08-28", "train", True, False, 15.0),
                ("2011-01-01", "train", False, False, 15.0),
            ]
        )
        service = assemble(frame)
        self.assertEqual(set(service.days), {"2012-08-07"})
        self.assertIn("2012-10-01", service.closed)
        self.assertIn("2012-10-29", service.closed)
        self.assertNotIn("2012-10-29", service.blocked)
        body = service.forecast("2012-08-07", None)
        self.assertNotIn("999", json.dumps(body))
        with self.assertRaises(ApiError) as sandy:
            service.forecast("2012-10-29", None)
        self.assertEqual(sandy.exception.status, 403)
        self.assertNotIn("outage", str(sandy.exception))
        self.assertNotIn("999", str(sandy.exception))
        with self.assertRaises(ApiError) as early:
            service.forecast("2011-01-01", None)
        self.assertIn("baseline", str(early.exception))


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = bind(_service(), port=0)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.httpd.server_address
        cls.base = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_get_returns_the_day_and_does_not_cache_it(self):
        status, body, headers = _request("GET", self.base + "/forecast?date=2012-08-07")
        self.assertEqual(status, 200)
        self.assertEqual(body["date"], "2012-08-07")
        self.assertEqual(len(body["hours"]), 24)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertTrue(headers["Content-Type"].startswith("application/json"))

    def test_post_without_weather_matches_get(self):
        _, by_get, _ = _request("GET", self.base + "/forecast?date=2012-08-07")
        status, by_post, _ = _request(
            "POST",
            self.base + "/forecast",
            json.dumps({"date": "2012-08-07"}),
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(by_post, by_get)

    def test_refusals_keep_their_status_codes(self):
        cases = [
            ("GET", "/forecast?date=2012-10-01", None, None, 403),
            ("GET", "/forecast?date=2011-08-28", None, None, 409),
            ("GET", "/nope", None, None, 404),
            ("GET", "/health?date=2012-08-07", None, None, 400),
            ("POST", "/forecast", json.dumps({"date": "2012-08-07", "weather": _weather()}), {"Content-Type": "application/json"}, 400),
            ("POST", "/forecast", "date=2012-08-07", {"Content-Type": "application/x-www-form-urlencoded"}, 415),
            ("POST", "/health", "{}", {"Content-Type": "application/json"}, 405),
            ("PUT", "/forecast", None, None, 405),
            ("POST", "/forecast", "x" * (MAX_BODY + 1), {"Content-Type": "application/json"}, 413),
        ]
        for method, path, body, headers, code in cases:
            status, payload, hdrs = _request(method, self.base + path, body, headers)
            self.assertEqual(status, code, payload)
            self.assertIn("error", payload)
            if code == 405:
                self.assertIn("GET", hdrs["Allow"])

    def test_health_names_the_model(self):
        status, body, _ = _request("GET", self.base + "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["model"], "no weather")
        self.assertFalse(body["accepts_weather"])
        self.assertEqual(body["served_days"], 1)


class FittedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = prepare()

    def test_validation_matches_the_day_87_no_weather_score(self):
        # Day 87 reported 44.7 for this rung on validation, and 47.1 for the baseline
        report = self.service.report
        self.assertEqual(report["fit_on"], "train")
        self.assertAlmostEqual(report["validation_mae"], 44.7, delta=0.05)
        self.assertAlmostEqual(report["validation_baseline_mae"], 47.1, delta=0.05)
        self.assertGreater(report["n_served_days"], 300)
        self.assertNotIn("cnt", self.service.report)

    def test_a_real_validation_day_is_served_and_october_is_closed(self):
        body = self.service.forecast("2012-08-07", None)
        self.assertEqual(body["slice"], "validation")
        self.assertEqual([hour["hr"] for hour in body["hours"]], list(range(24)))
        self.assertGreater(body["hours"][8]["expected"], body["hours"][3]["expected"])
        self.assertEqual(body["bikes"], sum(hour["bikes"] for hour in body["hours"]))
        with self.assertRaises(ApiError) as closed:
            self.service.forecast("2012-10-01", None)
        self.assertEqual(closed.exception.status, 403)
        with self.assertRaises(ApiError) as early:
            self.service.forecast("2011-01-01", None)
        self.assertEqual(early.exception.status, 409)


if __name__ == "__main__":
    unittest.main()
