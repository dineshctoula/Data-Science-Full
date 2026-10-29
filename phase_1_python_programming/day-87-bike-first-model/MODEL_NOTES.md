# Day 87 — First bike-demand model

Project 2, step 2. The frozen Day 86 split, metric, and baselines are imported from `day-86-bike-kickoff`. Test is still closed.

**The bar from Day 86:** the mean of the same hour over the last 4 weeks scores validation **MAE 47.1** bikes per hour (WAPE 16.3%).

## Two false starts, on purpose

1. **Ridge on the lags themselves**, in log or raw counts, lost to the baseline: MAE 48–50 on the log scale and about 90–111 raw. A model asked to learn "use the 4-week mean" from scratch mostly re-learns it badly. Least squares also fits quiet 3 a.m. hours as hard as rush hour.
2. **Adding the day's weather made it worse.** Both problems came from the same place: the baseline already contains the weather of the four weeks it averages.

## The model that worked: correct the baseline, don't replace it

```
log1p(forecast) = log1p(4-week same-hour mean) + b0 + X @ beta        (ridge, intercept free)
```

Every feature describes how **today differs from the four lag weeks**: f(today) − mean of f over those weeks. A wet Tuesday after four dry ones gets a big rain difference; a wet Tuesday after four wet ones gets none.

| block | what it adds |
|---|---|
| hour shape | 48 one-hots: hour × (working day or not) |
| holiday + drift | holiday vs the lag weeks, and yesterday's whole-day level vs the lag weeks' days (known at midnight) |
| weather | hinge curves in temperature (knots ≈ 12, 20, 29 °C) and humidity (60%, 80%), plus rain and mist flags and rain at rush hour, all taken as differences from the lag weeks |

**Why hinges:** a straight line in "warmer than the last few weeks" was learned mostly on cool months (correlation +0.30 with the correction in training). In July it means almost nothing (+0.04), and it pushed validation MAE to 51.6. With hinges, warming helps in spring and stops helping in summer heat.

## Scored two ways

The features were designed while looking at validation, so validation numbers are a little optimistic. As a check, every rung is also scored by **rolling-origin CV inside train**: each month from January to June 2012 is forecast by a model fit only on earlier days. The penalty (λ = 0.1 for every rung) was picked on those folds.

| rung | cols | rolling CV MAE | validation MAE | validation WAPE |
|---|---:|---:|---:|---:|
| 4-week baseline | — | 48.7 | 47.1 | 16.3% |
| baseline × hour shape | 48 | 48.4 | 46.1 | 16.0% |
| **+ holiday, drift** (no weather: the honest day-ahead forecast) | 50 | **45.3** | **44.7** | 15.5% |
| **+ weather** (actual weather: an upper bound) | 60 | **35.4** | **41.2** | 14.3% |

![ladder](output/ladder.png)

![cv months](output/cv_months.png)

The ordering holds under CV, and weather improves every one of the six months. So the design isn't tuned to one summer. Weather helps more in spring (CV) than in summer (validation): spring weather swings more from week to week.

### How sure? (day-block bootstrap, 92 validation days)

| gap | MAE | 95% interval |
|---|---:|---:|
| no-weather model − baseline | −2.4 | [−4.6, +0.2] |
| weather model − baseline | −5.9 | [−9.2, −2.8] |
| weather model − no-weather model | −3.5 | [−6.3, −1.1] |

On validation alone, the no-weather gain is **borderline**: the interval just touches zero. Its CV gain (48.7 → 45.3, five of six months better) is steadier. The weather gain is clear on both.

## Where the gain comes from

| weather | hours | baseline MAE | with weather |
|---|---:|---:|---:|
| clear | 1594 | 43.2 | 37.8 |
| mist / cloudy | 484 | 46.6 | 41.5 |
| rain / snow | 130 | 96.8 | 81.9 |

Rainy hours are still the hardest: an MAE of 82 on roughly 6% of the hours.

![sample week](output/sample_week.png)

On the rainy Friday (10 August), 8 a.m. had 119 rentals. The baseline and the no-weather model expected about 720, and the weather model about 350: much closer, but still well above what happened.

## What the model learned about weather

![weather curves](output/weather_curves.png)

Each value is the multiplier on rentals against the same hour at 20 °C and 60% humidity:

- **Temperature:** × 0.64 at 5 °C, × 0.73 at 12 °C, a peak of **× 1.21 around 29 °C**, then down to × 1.07 at 35 °C.
- **Humidity:** flat up to 60%, × 0.94 at 80%, × 0.81 at 95%.
- **Rain:** × 0.64 against dry, and a further × 0.95 at a working-day rush hour.

## Caveat that stays open

The weather rung uses the day's **actual** weather. A real day-ahead forecast would use a weather *forecast*, which is worse, especially for the timing of rain. So the 41.2 is an upper bound, and the honest day-ahead number is 44.7. How much of the weather gain survives forecast error is the next thing to measure.

**New bar on validation:** MAE **44.7** without weather, **41.2** with actual weather.
