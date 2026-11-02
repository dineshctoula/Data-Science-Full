"""Run one forecast, and optionally leave the server up.

    python3 main.py                 one validation day as JSON, then a closed-quarter refusal
    python3 main.py --date 2012-07-04
    python3 main.py --serve         same, then listen on http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json

from contract import ApiError
from fitted import prepare
from http_api import bind

# a Tuesday in the validation quarter, chosen here so the walkthrough
# isn't a date picked after seeing which day scores well
SAMPLE_DAY = "2012-08-07"
CLOSED_DAY = "2012-10-01"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default=SAMPLE_DAY, help="YYYY-MM-DD to forecast (default: a validation Tuesday)")
    parser.add_argument("--serve", action="store_true", help="keep listening on 127.0.0.1 after printing the sample")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    print("=" * 72)
    print("DAY 91: BIKE FORECAST OVER HTTP")
    print("=" * 72)
    service = prepare()
    report = service.report
    print(
        f"\n1) {report['model']!r}  rung {report['rung']!r}  lam {report['lam']:g}  fit on {report['fit_on']}"
        f"\n   {report['n_fit_hours']} train hours -> validation MAE {report['validation_mae']:.1f}"
        f"  (baseline {report['validation_baseline_mae']:.1f})"
        f"\n   serving {report['n_served_days']} days; test quarter is closed"
    )

    print(f"\n2) forecast {args.date} (knowable at midnight, no weather)")
    try:
        body = service.forecast(args.date, None)
    except ApiError as exc:
        print(f"   refused: {exc.status} {exc}")
        return 1
    print(json.dumps(body, indent=2))

    print("\n3) the test quarter is not a route")
    try:
        service.forecast(CLOSED_DAY, None)
    except ApiError as exc:
        print(f"   {CLOSED_DAY}: {exc.status} {exc}")
    else:
        print("   error: a test day was served")
        return 1

    if args.serve:
        httpd = bind(service, args.port)
        host, port = httpd.server_address
        print(f"\n4) listening on http://{host}:{port}/forecast?date={args.date}")
        print("   Ctrl-C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n   stopped")
        finally:
            httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
