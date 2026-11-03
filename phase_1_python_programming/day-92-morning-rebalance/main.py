"""Ask Day 91 for a week and print the morning rebalancing table.

    python3 main.py                  the validation week of 2012-08-06, then a week that enters October
    python3 main.py --start 2012-08-06
"""

from __future__ import annotations

import argparse
import threading

import server_path  # noqa: F401  (puts Day 91 on the path without taking its module name)
from client import ForecastClient
from fitted import prepare
from http_api import bind
from week import CROSSING_START, SAMPLE_START, build_week, closed_quarter_is_blank, render


def _print_week(client, start: str, title: str) -> list[dict]:
    print(f"\n{title}")
    rows = build_week(client, start)
    print(render(rows))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=SAMPLE_START, help="first day of the week to stage (default: a validation Monday)")
    args = parser.parse_args(argv)

    print("=" * 72)
    print("DAY 92: MORNING REBALANCING TABLE")
    print("=" * 72)
    print("\n1) fit Day 91's no-weather model on train, and listen on 127.0.0.1 (an ephemeral port)")
    service = prepare()
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
            f"\n   validation MAE {report['validation_mae']:.1f}  "
            f"(baseline {report['validation_baseline_mae']:.1f})"
        )

        print("\n2) one validation week, knowable at each midnight, no weather sent")
        sample = _print_week(client, args.start, f"   week of {args.start}")
        if any(row["status"] != "served" for row in sample):
            print("   error: the sample week was not fully served")
            return 1
        if any(row["slice"] == "test" for row in sample):
            print("   error: a test day was staged")
            return 1

        print("\n3) a week that crosses into the closed quarter")
        crossing = _print_week(client, CROSSING_START, f"   week of {CROSSING_START}")
        leaked = closed_quarter_is_blank(crossing)
        if leaked:
            print(f"   error: the test quarter was staged: {', '.join(leaked)}")
            return 1
        if not any(row["status"] == "refused" and row["date"] >= "2012-10-01" for row in crossing):
            print("   error: October was not refused")
            return 1
        print("   October rows carry no bike count")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
