# Day 92 — Morning rebalancing table

Day 91 answers one date. A morning crew needs the week. This day is a client: it asks that server for seven days and turns the replies into a staging table. No new demand model. The fit is still the no-weather ridge, train only, validation MAE **44.7** against **47.1** for the baseline.

The UCI file is one citywide count per hour, not a dock. The table says when the fleet has to be out. It does not say which station a bike should sit at.

## The windows

The clocks are Day 86's shapes and Day 87's peak hours, not a fit on this week.

| window | hours | what it is for |
|---|---|---|
| morning | 7–9 | inbound commute, staged before 7 |
| midday | 11–15 | the weekend hump |
| evening | 16–19 | Day 87's evening peak |
| quiet | 0–5 | stays in the yard; not a wave to staff |

Hours 6, 10, and 20–23 are shoulders. They are in the day total and in no window, so the four window columns do not add up to the day. The day column is the server's integer total. Tuesday 7 August is **7071**, the same total Day 91 printed for that date.

`extra` is the morning window minus the 4-week baseline, both staged as whole bikes (half a bike rounds up, then the hours are added). Positive means stage more than a typical recent week of those hours.

The first wave is the staffed window with the most bikes per hour. Midday is five hours and morning is three, so the raw sums would crown midday on a flat day. A tie goes to the earlier window.

The weekday label is the calendar only. The reply does not carry the holiday flag, so a weekday holiday is still labeled "working day". The counts come from the model, which did see the holiday. Only the label is coarse.

## The week of 6 August

All seven days are in validation. Each one is a GET, with no weather body.

| date | kind | morning | extra | midday | evening | day | first wave |
|---|---|---:|---:|---:|---:|---:|---|
| 2012-08-06 Mon | working day | 1299 | −92 | 1073 | 2461 | 6303 | evening |
| 2012-08-07 Tue | working day | 1602 | +49 | 1175 | 2689 | 7071 | evening |
| 2012-08-08 Wed | working day | 1584 | +52 | 1239 | 2632 | 7303 | evening |
| 2012-08-09 Thu | working day | 1619 | +91 | 1247 | 2668 | 7327 | evening |
| 2012-08-10 Fri | working day | 1545 | +78 | 1496 | 2433 | 7089 | evening |
| 2012-08-11 Sat | weekend | 450 | −35 | 2342 | 1462 | 5948 | midday |
| 2012-08-12 Sun | weekend | 392 | −20 | 2586 | 1619 | 6257 | midday |

**8491** morning bikes across the week, **+123** against the 4-week baseline. Monday is the only weekday under that baseline. Every working day staffs the evening wave first; both weekend days staff midday. That is Day 86's two shapes, read off the forecast rather than off the history.

## A week that enters October

Friday 28 September through Thursday 4 October. The first three days are served. 1–4 October are the closed test quarter: HTTP 403, and the bike columns are blank. They are not zeros. A zero would tell the crew to pull the fleet in. The morning total is the three served days only: **2657** (−4 vs the baseline).

## Tests

17 tests in `test_rebalance.py`. The windows, the half-up baseline, a leaked target, and a refused row use replies built in the test. One HTTP round trip uses Day 91's three-day table, including a 400 that must not become a blank day. One fit asks for both weeks above and checks that October contributes no bikes.

```bash
python3 tools/run_day.py 92
python3 tools/run_day.py 92 --tests
```
