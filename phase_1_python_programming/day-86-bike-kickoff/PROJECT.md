# Project 2 — How many bikes will be rented each hour tomorrow? (Day 86: kickoff)

Project 1 was classification on independent customers. Project 2 is **regression over time**: rows are hours, they come in order, and tomorrow depends on today. Today sets up the question, the data, the split, the metric, and the bar to beat. There is no model yet.

## The question

At midnight, forecast all 24 hours of the next day's rentals, so bikes can be rebalanced across stations before the morning rush.

- **Allowed inputs** for day D: the calendar (hour, weekday, holiday), the weather for day D, and every count up to the end of D−1. Using actual weather in place of a weather forecast is a simplification. Later results will be a little optimistic because of it, and the notes on later days will say so.
- **Headline metric:** MAE in bikes per hour, scored on every non-outage hour. **WAPE** (total absolute error ÷ total rentals) rides along, because demand grows and MAE alone would make busy months look worse.
- **Intervals** resample whole days (92 in validation), not hours. The 24 hours of a day share weather and events. Resampling single hours gave a 2.4× narrower interval for the same comparison, which would overstate the evidence.

## The data

[UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Capital Bikeshare in Washington D.C., 2011–2012, hourly (CC BY 4.0). Citation requested by the authors: Fanaee-T and Gama, *Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence (2013).

`bike_data.fetch()` caches the zip under `data/` (git-ignored) and refuses any file whose sha256 isn't `b70182d0…d401`.

### Rebuilding the hourly grid

`hour.csv` has 17,379 of the 17,544 hours. **No recorded hour has a count of 0**, which explains the gaps:

| gap | hours | treatment |
|---|---:|---|
| scattered quiet hours, mostly 3–4 a.m. | 74 | filled with **0 rentals** |
| days missing 6+ hours: shutdowns | 91 on 8 days | the whole day is flagged as an **outage**: counts blanked, never scored, never used as history |

The outage days are 2011-01-18, 2011-01-26/27 (snowstorm), 2011-02-22, 2011-08-27/28 (Hurricane Irene), and 2012-10-29/30 (Hurricane Sandy, which falls in the test slice). Weather for the filled hours comes from the neighbouring hours.

### What the audit found

- **A leakage trap.** `casual + registered == cnt` on all 17,278 usable hours, and `registered` alone correlates with the target at 0.97. Both columns are the target split two ways, so both are dropped.
- **Strong growth.** The same month runs **1.41× to 2.57×** higher in 2012 than in 2011. A random split would let a model peek at the future level, which is why the split follows time.
- **Two kinds of day.** Working days peak at 8 a.m. (479 an hour) and 5 p.m. (529). Weekends and holidays have one mid-afternoon hump (about 374).
- **Which history is usable.** The previous hour correlates best (0.844), but a day-ahead forecast can't see it for most hours. Of the usable lags, **same hour last week (0.880)** beats same hour yesterday (0.823), because the weekday/weekend pattern repeats weekly.
- **Weather matters.** By weather category 1–4, rentals per hour fall from 204 to 175, 112 and 94.

![timeline](output/timeline.png)

![daily shapes](output/daily_shapes.png)

## The split (frozen)

Whole days in time order, committed in **`split_dates.tsv`**:

| slice | dates | scorable hours | mean rentals / hour |
|---|---|---:|---:|
| train | 2011-01-01 .. 2012-06-30 | 12,984 | 167.3 |
| validation | 2012-07-01 .. 2012-09-30 | 2,208 | 288.4 |
| test | 2012-10-01 .. 2012-12-31 | 2,160 | 219.5 |

Validation has no outage days. Test includes Sandy and the holiday season, and it is a different season from validation. That's deliberate: it's a harder, more honest check.

## Baselines (day-ahead, scored on validation)

Every baseline is built from shifts of 24 hours or more, and `lag()` refuses anything shorter.

| baseline | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| yesterday, same hour | 83.7 | 139.9 | 29.0% |
| last week, same hour | 56.0 | 97.9 | 19.4% |
| **mean of the last 4 weeks, same hour** | **47.1** | **79.6** | **16.3%** |
| training-period profile (by working day × hour) | 124.9 | 167.6 | 43.3% |

- Averaging 4 weeks beats copying 1 week by 8.9 bikes per hour, with a day-block 95% interval of [−14.7, −3.8].
- The frozen training profile is the worst by far. It has the right *shape* but the wrong *level*, because demand kept growing after training ended.
- "Yesterday" suffers on every Monday and Saturday, when it copies the wrong kind of day.

![sample week](output/sample_week.png)

The sample week shows what lags can't do. On **Friday 10 August** it rained at rush hour (weather category 3), and rentals at 8 a.m. were 119 instead of the usual ~700. A forecast built from past weeks can't see that coming. The day's weather can, and that's the opening for the first model.

**The bar:** the first model has to beat **MAE 47.1** on validation.

## Files

| file | what it does |
|---|---|
| `bike_data.py` | fetch with hash check, rebuild the hourly grid, label quiet gaps and outage days |
| `audit_engine.py` | gaps, growth, daily shapes, leak check, lag correlations, weather effect |
| `split_engine.py` | write and load `split_dates.tsv`, assign slices, scorable hours |
| `baseline_engine.py` | MAE, RMSE, WAPE, day-ahead lag baselines, day-block bootstrap |
| `visualizer.py` | timeline, daily shapes, sample week |
| `main.py` | the walkthrough (`python3 main.py`) |
| `test_bike_kickoff.py` | 13 tests; the real-data test skips until the zip is cached |
