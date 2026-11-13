"""Ask Day 91 for a week, stage it as Day 92 did, and score the mornings.

    python3 main.py                  the validation week of 2012-08-06, then a week that enters October
    python3 main.py --start 2012-08-06
"""

from __future__ import annotations

import argparse
import threading

import paths  # noqa: F401

from actuals import load_open_mornings
from client import ForecastClient
from fitted import prepare
from http_api import bind
from week import CROSSING_START, SAMPLE_START, build_week, closed_quarter_is_blank
from week_score import render, score_week, summarize


def _print_scored(client, actuals, start: str, title: str) -> list[dict]:
    print(f"\n{title}")
    rows = build_week(client, start)
    scored = score_week(rows, actuals)
    print(render(scored))
    return scored


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=SAMPLE_START, help="first day of the week to score (default: a validation Monday)")
    args = parser.parse_args(argv)

    print("=" * 72)
    print("DAY 93: MORNING STAGING SCORE")
    print("=" * 72)
    print("\n1) fit Day 91's no-weather model on train, and load mornings that are not test")
    service = prepare()
    actuals = load_open_mornings()
    # a test date in this table would mean the drop in actuals.py failed
    if any(day >= "2012-10-01" for day in actuals):
        print("   error: the test quarter is in the morning table")
        return 1
    # port 0 so this walkthrough does not occupy 8765, which Day 91's notes use
    httpd = bind(service, 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address
        client = ForecastClient(f"http://{host}:{port}")
        health = client.health()
        report = service.report
        print(
            f"   {health['model']!r}  accepts_weather={health['accepts_weather']}  "
            f"served_days={health['served_days']}"
            f"\n   hourly validation MAE {report['validation_mae']:.1f}  "
            f"(baseline {report['validation_baseline_mae']:.1f})"
            f"\n   open mornings kept: {len(actuals)}  (train + validation; October is absent)"
        )

        print("\n2) one validation week: staged bikes vs the rides that happened, 7–9 a.m.")
        sample = _print_scored(client, actuals, args.start, f"   week of {args.start}")
        if any(row["status"] != "scored" for row in sample):
            print("   error: the sample week was not fully scored")
            return 1
        totals = summarize(sample)
        if totals["mae"] is None or totals["mae"] >= totals["baseline_mae"]:
            # the model is supposed to beat the baseline on this week too, but
            # a tie or a loss is still a result: print it, don't hide it
            print("   note: morning MAE did not beat the 4-week baseline this week")

        print("\n3) a week that crosses into the closed quarter")
        crossing = _print_scored(client, actuals, CROSSING_START, f"   week of {CROSSING_START}")
        leaked = closed_quarter_is_blank(
            [
                {
                    "date": row["date"],
                    "status": "served" if row["status"] == "scored" else "refused",
                    "morning_bikes": row["morning_actual"],
                    "day_bikes": row["morning_actual"],
                }
                for row in crossing
            ]
        )
        if leaked:
            print(f"   error: the test quarter was scored: {', '.join(leaked)}")
            return 1
        october = [row for row in crossing if row["date"] >= "2012-10-01"]
        if not october or any(row["status"] == "scored" or row["morning_actual"] is not None for row in october):
            print("   error: October was scored")
            return 1
        print("   October rows carry no actual and no error")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
