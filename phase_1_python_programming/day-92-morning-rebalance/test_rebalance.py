"""Day 92 — the windows, a refused day, and one live week.

The row and week checks use a reply built in the test. The HTTP checks
use Day 91's three-day table. One test fits the real model, asks for the
validation week and for the week that enters October, and checks that
October contributes no bikes.
"""

from __future__ import annotations

import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import server_path  # noqa: F401
from client import ClientError, ForecastClient
from contract import ApiError
from http_api import bind
from service import DayRecord, ForecastService
from table import calendar_kind, refused_row, staging_row
from week import (
    CLOSED_FROM,
    CROSSING_START,
    SAMPLE_START,
    build_week,
    closed_quarter_is_blank,
    render,
    summarize,
    week_dates,
)
from windows import SHOULDERS, WINDOWS, first_wave, window_baseline, window_bikes


def _hours(scale=1.0):
    return [{"hr": hr, "baseline": 10.0 + hr, "expected": scale * (12.0 + hr)} for hr in range(24)]


def _toy_service():
    return ForecastService(
        days={
            "2012-08-06": DayRecord("validation", _hours()),
            "2012-08-07": DayRecord("validation", _hours(2)),
            "2012-09-28": DayRecord("validation", _hours()),
            "2012-09-29": DayRecord("validation", _hours()),
            "2012-09-30": DayRecord("validation", _hours()),
        },
        closed=["2012-10-01", "2012-10-02", "2012-10-03", "2012-10-04"],
        blocked={"2011-01-01": "the 4-week baseline or the midnight drift feature is missing"},
        model_name="no weather",
    )


def _reply(day="2012-08-07", bikes_at=None, baseline_at=None):
    hours = []
    for hr in range(24):
        bikes = 1 if bikes_at is None else bikes_at(hr)
        baseline = 1.0 if baseline_at is None else baseline_at(hr)
        hours.append({"hr": hr, "baseline": baseline, "expected": float(bikes), "bikes": bikes})
    total = sum(hour["bikes"] for hour in hours)
    return {
        "date": day,
        "as_of": f"{day}T00:00:00",
        "model": "no weather",
        "slice": "validation",
        "hours": hours,
        "expected_total": float(total),
        "bikes": total,
    }


def _by_hour(body):
    return {hour["hr"]: {"bikes": hour["bikes"], "baseline": hour["baseline"]} for hour in body["hours"]}


class WindowTests(unittest.TestCase):
    def test_the_windows_cover_the_day_except_the_shoulders(self):
        seen = []
        for hours in WINDOWS.values():
            for hr in hours:
                self.assertNotIn(hr, seen)
                seen.append(hr)
        self.assertEqual(sorted(seen + list(SHOULDERS)), list(range(24)))

    def test_the_wave_is_bikes_per_hour_and_a_tie_goes_to_morning(self):
        # equal bikes per hour: midday's extra hours must not win on the sum
        flat = {"morning": 30, "midday": 50, "evening": 40}
        self.assertEqual(first_wave(flat), "morning")
        self.assertEqual(first_wave({"morning": 3, "midday": 50, "evening": 4}), "midday")
        self.assertEqual(first_wave({"morning": 3, "midday": 5, "evening": 40}), "evening")

    def test_half_a_baseline_bike_rounds_up_inside_the_window(self):
        body = _reply(bikes_at=lambda hr: 2, baseline_at=lambda hr: 0.5)
        by_hour = _by_hour(body)
        # three morning hours, each baseline 0.5 -> 1, so the window baseline is 3
        self.assertEqual(window_baseline(by_hour, WINDOWS["morning"]), 3)
        self.assertEqual(window_bikes(by_hour, WINDOWS["morning"]), 6)


class TableTests(unittest.TestCase):
    def test_a_weekday_and_a_weekend_are_labeled_from_the_calendar(self):
        # 2012-08-07 is the Tuesday Day 91 prints; the 11th is the Saturday of that week
        self.assertEqual(calendar_kind("2012-08-07"), ("Tuesday", "working day"))
        self.assertEqual(calendar_kind("2012-08-11"), ("Saturday", "weekend"))

    def test_morning_extra_is_forecast_minus_the_staged_baseline(self):
        body = _reply(bikes_at=lambda hr: 10 if hr in (7, 8, 9) else 0, baseline_at=lambda hr: 0.5)
        row = staging_row(body)
        self.assertEqual(row["morning_bikes"], 30)
        self.assertEqual(row["morning_extra"], 27)
        self.assertEqual(row["status"], "served")
        # hour 6 is a shoulder: it is in the day total and in no window
        body = _reply(bikes_at=lambda hr: 9 if hr == 6 else 0, baseline_at=lambda hr: 0.0)
        row = staging_row(body)
        self.assertEqual(row["day_bikes"], 9)
        self.assertEqual(row["morning_bikes"] + row["midday_bikes"] + row["evening_bikes"] + row["quiet_bikes"], 0)

    def test_a_reply_that_carries_the_target_is_not_staged(self):
        body = _reply()
        body["cnt"] = 400
        with self.assertRaises(ValueError):
            staging_row(body)

    def test_a_day_total_that_disagrees_with_the_hours_is_not_staged(self):
        body = _reply()
        body["bikes"] = body["bikes"] + 1
        with self.assertRaises(ValueError):
            staging_row(body)

    def test_a_refusal_is_blank_rather_than_zero(self):
        row = refused_row("2012-10-01", 403, "closed")
        self.assertEqual(row["status"], "refused")
        self.assertIsNone(row["morning_bikes"])
        self.assertIsNone(row["day_bikes"])
        self.assertIsNone(row["morning_extra"])


class WeekTests(unittest.TestCase):
    def test_the_sample_week_holds_the_day_91_tuesday_and_stops_before_october(self):
        days = week_dates(SAMPLE_START)
        self.assertEqual(days[0], "2012-08-06")
        self.assertEqual(len(days), 7)
        self.assertIn("2012-08-07", days)
        self.assertTrue(all(day < CLOSED_FROM for day in days))
        crossing = week_dates(CROSSING_START)
        self.assertEqual(crossing[0], "2012-09-28")
        self.assertIn("2012-10-01", crossing)

    def test_a_mismatched_reply_date_is_not_staged_on_the_asked_day(self):
        body = _reply("2012-08-08")

        class _Wrong:
            def fetch(self, day):
                return {"date": day, "status": 200, "error": "", "body": body}

        with self.assertRaises(ValueError):
            build_week(_Wrong(), "2012-08-06")

    def test_refused_days_are_left_out_of_the_morning_total(self):
        served = staging_row(_reply("2012-09-28", bikes_at=lambda hr: 4))
        refused = refused_row("2012-10-01", 403, "2012-10-01 is in the closed test quarter and is not served")
        rows = [served, refused]
        totals = summarize(rows)
        self.assertEqual(totals["n_served"], 1)
        self.assertEqual(totals["n_refused"], 1)
        self.assertEqual(totals["morning_bikes"], served["morning_bikes"])
        self.assertEqual(closed_quarter_is_blank(rows), [])
        text = render(rows)
        self.assertIn("403", text)
        self.assertIn("closed test quarter", text)
        # a blank cell, not a staged zero, on the refused line
        refused_line = next(line for line in text.splitlines() if line.startswith("2012-10-01"))
        self.assertIn("-", refused_line)
        self.assertNotIn(" 0", refused_line)

    def test_a_served_october_day_is_reported(self):
        row = staging_row(_reply("2012-10-01"))
        self.assertEqual(closed_quarter_is_blank([row]), ["2012-10-01"])


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = bind(_toy_service(), port=0)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.httpd.server_address
        cls.client = ForecastClient(f"http://{host}:{port}")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_a_served_day_and_a_closed_day_come_back_over_http(self):
        reply = self.client.fetch("2012-08-07")
        self.assertEqual(reply["status"], 200)
        row = staging_row(reply["body"])
        self.assertEqual(row["date"], "2012-08-07")
        self.assertGreater(row["morning_bikes"], 0)
        self.assertNotIn("cnt", reply["body"])

        closed = self.client.fetch("2012-10-01")
        self.assertEqual(closed["status"], 403)
        self.assertIsNone(closed["body"])
        blank = refused_row(closed["date"], closed["status"], closed["error"])
        self.assertIsNone(blank["day_bikes"])
        self.assertIn("closed test quarter", blank["reason"])

    def test_the_client_does_not_turn_a_400_into_a_blank_day(self):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                body = b'{"error":"bad request"}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = httpd.server_address
            client = ForecastClient(f"http://{host}:{port}")
            with self.assertRaises(ClientError):
                client.fetch("2012-08-07")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_a_dead_server_raises(self):
        client = ForecastClient("http://127.0.0.1:9", timeout=1.0)
        with self.assertRaises(ClientError):
            client.fetch("2012-08-07")

    def test_a_bad_spelling_fails_before_the_request(self):
        with self.assertRaises(ApiError):
            self.client.fetch("2012-8-7")


class FittedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fitted import prepare

        cls.service = prepare()
        cls.httpd = bind(cls.service, port=0)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.httpd.server_address
        cls.client = ForecastClient(f"http://{host}:{port}")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_the_validation_week_is_served_and_october_is_blank(self):
        sample = build_week(self.client, SAMPLE_START)
        self.assertEqual(len(sample), 7)
        self.assertTrue(all(row["status"] == "served" for row in sample))
        self.assertTrue(all(row["slice"] == "validation" for row in sample))
        self.assertEqual(closed_quarter_is_blank(sample), [])
        tuesday = next(row for row in sample if row["date"] == "2012-08-07")
        self.assertGreater(tuesday["morning_bikes"], tuesday["quiet_bikes"])
        totals = summarize(sample)
        self.assertEqual(totals["morning_bikes"], sum(row["morning_bikes"] for row in sample))

        crossing = build_week(self.client, CROSSING_START)
        self.assertEqual(closed_quarter_is_blank(crossing), [])
        october = [row for row in crossing if row["date"] >= CLOSED_FROM]
        september = [row for row in crossing if row["date"] < CLOSED_FROM]
        self.assertTrue(all(row["status"] == "refused" and row["http_status"] == 403 for row in october))
        self.assertTrue(all(row["morning_bikes"] is None for row in october))
        self.assertTrue(all(row["status"] == "served" for row in september))
        # the morning total does not treat the four closed days as zero bikes staged
        self.assertEqual(summarize(crossing)["morning_bikes"], sum(row["morning_bikes"] for row in september))
        self.assertNotIn("cnt", render(crossing))


if __name__ == "__main__":
    unittest.main()
