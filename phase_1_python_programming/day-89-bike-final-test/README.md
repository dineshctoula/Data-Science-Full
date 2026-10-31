# Project 2 — How many bikes will be rented each hour tomorrow?

A day-ahead forecast of hourly bike rentals for [Capital Bikeshare](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (Washington D.C., 2011–2012, CC BY 4.0), built over Days 86–89 with NumPy and pandas only. At midnight, the forecast covers all 24 hours of the next day, using only what's known by then.

## Result on the held-out test slice (Oct–Dec 2012, opened once)

| model | MAE (bikes/hour) | 95% interval (days) | WAPE |
|---|---:|---:|---:|
| 4-week same-hour mean (floor) | 60.6 | [51.8, 70.1] | 27.6% |
| no weather | 47.8 | [41.7, 54.5] | 21.8% |
| **weather from typical forecasts** (primary) | **46.9** ± 0.9 over 10 forecast draws | [41.3, 51.6] | 21.4% |
| *actual weather (upper bound, not a real forecast)* | *43.5* | [38.9, 48.3] | 19.8% |

Scored on 2160 hours over 90 days. The two Hurricane Sandy outage days are excluded.

- **The model clearly beats the floor:** −13.7 MAE, and every one of the 10 forecast draws gives an interval wholly below zero. In autumn, demand fell month after month, and the 4-week mean lagged behind it, over-forecasting total rentals by 15.5% against 4% for the model.
- **Weather from a forecast helps only a little:** −0.9 against no weather, ranging from −2.2 to +0.3 across draws, and no draw's interval excludes zero. It helped in November (49.4 vs 53.0) and hurt in December (44.7 vs 42.1).

![cards](output/cards.png)

![seed gaps](output/seed_gaps.png)

## How it was built

| day | folder | what it settled |
|---|---|---|
| 86 | [`day-86-bike-kickoff`](../day-86-bike-kickoff) | Rebuilt the full hourly grid: quiet hours filled with 0, 8 shutdown days flagged as outages. Dropped the `casual` + `registered` leak. Froze a time split. Set the 4-week same-hour baseline (validation MAE 47.1) |
| 87 | [`day-87-bike-first-model`](../day-87-bike-first-model) | Ridge **correction** to the baseline on the log scale. Every feature is a difference from the four lag weeks, with hinge-curve weather. Confirmed by rolling-origin CV inside train |
| 88 | [`day-88-bike-forecast-weather`](../day-88-bike-forecast-weather) | Simulated forecast-grade weather. A model trained on actual weather lost to *no weather*, so the model is trained on forecast-grade weather instead |
| 89 | this folder | The recipe was committed first, the drift fallback for the day after Sandy was set before test opened, and the gate opened test once (`test_opened.lock`) |

The rules followed throughout:

- The split follows time, because demand grew 1.4–2.6× year over year.
- No forecast uses anything from the day it's forecasting.
- Intervals resample whole days, not hours. Hourly resampling gave intervals 2.4× too narrow.
- Choices were made on validation or rolling CV, and test was opened once.

## What the four evidence sources said about weather

| | no weather | weather, typical forecasts | gap |
|---|---:|---:|---:|
| rolling CV, Jan–Jun 2012 | 45.3 | 40.5 | −4.8 |
| validation, Jul–Sep 2012 | 44.7 | 44.8 | +0.1 |
| **test, Oct–Dec 2012** | **47.8** | **46.9** | **−0.9** |

The spring folds made weather from forecasts look valuable, and summer made it look useless. Autumn landed in between, closer to summer. The honest conclusion: with a typical day-ahead forecast, weather is worth roughly a bike an hour, and that isn't proven at this sample size. What reliably beats the floor is the correction model's sense of level and calendar (drift, holidays, hour shape).

## Where it still misses

![Thanksgiving week](output/sample_week.png)

The sample week was picked in the recipe, before test opened, so it couldn't be chosen to flatter. Thanksgiving itself (Thursday 22 November) is flagged as a holiday, and every model predicts a weekend-shaped day. **The Friday after** is a working day in the data. Every model predicts commute peaks of about 600, but people were off, and rentals came in at roughly 250–450. The cold weekend that followed was over-forecast by about 2×. Bridge days and late-autumn weekends are the obvious next feature.

## Limits

- **Simulated forecasts.** The dataset has only actual weather. The forecast errors are stated assumptions (2.1 °C temperature error, 53% of rain hours caught for "typical"), not real forecasts.
- **One city, two years.** Demand was still growing quickly, and a mature system would behave differently.
- **Test is one quarter.** 90 days gives MAE intervals about ±5 wide, so differences of a bike or two can't be told apart.
- **The lock's `rows` field reads 17544:** the gate was handed a mask over the whole grid. Test is 2208 hours, and 2160 of them were scored.

## Run it

```bash
cd phase_1_python_programming/day-86-bike-kickoff && python3 main.py     # fetches and caches the zip
cd ../day-89-bike-final-test && python3 main.py --rehearse              # full pipeline on validation
python3 main.py                                                         # the recorded test run (gate checks the recipe)
python3 -m unittest test_final                                          # never reads the test slice
```
