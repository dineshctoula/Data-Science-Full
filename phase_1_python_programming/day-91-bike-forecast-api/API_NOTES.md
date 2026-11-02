# Day 91 — Forecast HTTP API

Project 2's no-weather model, the one Day 87 called the honest day-ahead forecast, behind a local HTTP server. No new demand model today. The new piece is the contract: what a caller may send, what comes back, and which days are refused.

The fit is Day 87's ridge on the log-ratio to the 4-week baseline, λ = 0.1, train hours only (12,720). Validation is not in the fit, so a request for an August day is a real forecast. Validation MAE is **44.7**, against **47.1** for the baseline: the same pair Day 87 reported.

## Which days answer

| | days |
|---|---:|
| served (train and validation) | 622 |
| closed test quarter, never predicted | 92 |
| outage, refused | 6 |
| baseline or drift missing | 11 |

622 + 92 + 17 = 731, every day of 2011–2012. The test quarter is 2012-10-01 through 2012-12-31. Those rows are not predicted and then hidden: `prepare` never calls the model on them, and a frame that already contains test predictions still has them dropped before the server sees it.

Sandy (2012-10-29) is both an outage and inside the test quarter. The closed-quarter answer wins, so the reply does not describe test days. The six outages that are refused as outages are all in train, including Hurricane Irene on 2011-08-27 and 2011-08-28.

The 11 short days are the first week of 2011, before the 4-week baseline exists, and the days after an outage. Drift reads yesterday's whole-day mean, and an outage has no yesterday. Day 89's fallback (use the last non-outage day) is part of the locked test recipe and is not reused here.

## One day

`2012-08-07`, a validation Tuesday, knowable at midnight, no weather sent:

| | 03:00 | 08:00 | day |
|---|---:|---:|---:|
| 4-week baseline | 4.5 | 703 | — |
| forecast | 4.2 | 725.6 | 7068.7 expected |
| bikes to stage | 4 | 726 | **7071** |

Half a bike rounds up (0.5 → 1), not to even, because quiet hours land near those halves more often than rush hour. The day total is the sum of the 24 integers, which is what someone adding the column would get. It is 2 bikes above the rounded-off float total on this day, and it can differ by up to 12.

`as_of` is `2012-08-07T00:00:00` with no timezone. The UCI file is local time with no zone, and attaching UTC would move the morning peak.

## The contract

Standard library only (`http.server`, `json`). Listens on `127.0.0.1`.

| | |
|---|---|
| `GET /health` | model name, and that this server does not accept weather |
| `GET /forecast?date=YYYY-MM-DD` | the 24 hours |
| `POST /forecast` | `{"date": "..."}` or the same plus `"weather": [24 rows]` |

Weather is parsed fully (hours 0–23 once, temp and humidity on the UCI 0–1 scale, `weathersit` in 1–4) and then refused with **400**. This model was trained without weather. Day 88 showed that feeding a forecast to a model trained on actual weather can lose to no weather at all; ignoring a weather body would be the same kind of quiet mix-up. A weather body on a test date is still a 400: the bad request is reported before the closed quarter.

| status | when |
|---:|---|
| 400 | bad date, bad weather, or weather sent to this model |
| 403 | the test quarter |
| 404 | unknown path, or a date that is not in 2011–2012 |
| 409 | outage, or the baseline or drift is missing |
| 413 | body over 8 KB |
| 415 | not JSON |

The true count is not a field. Replies send `Cache-Control: no-store`.

## Tests

16 tests in `test_api.py`. The contract, the refusals, and the HTTP round trip use a three-day table. One test fits the real model, checks the Day 87 validation MAE, serves 2012-08-07, and checks that 2012-10-01 comes back 403.

```bash
python3 tools/run_day.py 91                 # the sample day as JSON, then the 403
python3 tools/run_day.py 91 -- --serve      # then listen on http://127.0.0.1:8765
python3 tools/run_day.py 91 --tests
```
