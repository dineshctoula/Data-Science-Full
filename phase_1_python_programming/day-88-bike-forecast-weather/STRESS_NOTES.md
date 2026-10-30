# Day 88 — Does the weather gain survive a forecast?

Project 2, step 3. Day 87's best model used the day's **actual** weather (validation MAE 41.2), which a midnight forecast can't have. The honest no-weather model scored 44.7. Today measures how much of that 3.5-bike gap survives when today's weather comes from a forecast. Test is still closed.

## Simulating a forecast

The data has no forecasts, so `weather_noise.py` makes forecast-grade weather from the actual weather. **The error sizes are assumptions** in the range of day-ahead forecasts, not measurements. Three scenarios bracket them:

| scenario | temperature error (avg) | humidity error | rain hours caught | false rain alarms |
|---|---:|---:|---:|---:|
| good | 1.3 °C | 4.7 pp | 67% | 2.9% |
| typical | 2.1 °C | 7.8 pp | 53% | 4.8% |
| poor | 3.4 °C | 11.4 pp | 38% | 7.9% |

Temperature error is mostly a **per-day bias** plus a small hourly wobble: a forecast that's too warm at 9 a.m. is usually too warm at 3 p.m. Only today's weather is perturbed. The four lag weeks are history and stay actual. Each scenario is drawn 10 times (one draw ≈ one season of forecasts), and the ridge penalty stays at Day 87's CV pick, so nothing new is tuned.

## Two ways to train

- **Trained on actual:** fit on what the weather really was, then forecast from a forecast. That's what happens if you build on historical observations and deploy.
- **Trained on forecasts:** fit on forecast-grade weather too, so the model learns how far to trust a forecast.

## Results (mean of 10 draws ± sd)

"Kept" = the share of the no-weather → perfect-weather gain that survives.

| scenario | trained on | validation MAE | kept | rolling CV MAE | kept | rain × |
|---|---|---:|---:|---:|---:|---:|
| *no weather* | — | *44.7* | — | *45.3* | — | — |
| *perfect* | — | *41.2* | — | *35.4* | — | *0.64* |
| good | actual | 44.5 ± 0.6 | 6% | 38.7 ± 0.3 | 66% | 0.64 |
| good | **forecasts** | **43.2 ± 0.6** | **43%** | **38.7 ± 0.3** | **67%** | 0.75 |
| typical | actual | 47.8 ± 1.1 | −87% | 41.4 ± 0.5 | 39% | 0.64 |
| typical | **forecasts** | **44.8 ± 0.8** | **−2%** | **40.5 ± 0.4** | **48%** | 0.79 |
| poor | actual | 53.6 ± 1.1 | −251% | 45.9 ± 0.9 | −6% | 0.64 |
| poor | **forecasts** | **46.1 ± 0.5** | **−40%** | **42.5 ± 0.5** | **28%** | 0.86 |

![survival](output/survival.png)

### 1. A model trained on actual weather can be worse than no weather

With typical forecasts, the actual-trained model scores 47.8 on validation, against 44.7 with no weather at all. On one draw, its gap to the no-weather model is +2.9, with a day-block interval of [+0.1, +5.3]. It applies the full weather effect it learned (rain × 0.64) to forecasts that are wrong about half the rainy hours, so it overreacts. A model is only as good as the inputs it will actually get.

### 2. Train on what you'll have at forecast time

Training on forecast-grade weather is better in every scenario, under both CV and validation. The model learns to trust a forecast less: the rain multiplier moves from × 0.64 to × 0.75, 0.79 and 0.86 as forecasts get worse. Noisy inputs pull fitted slopes toward zero (errors-in-variables), and here that's exactly right, because the noise at fit time matches the noise at forecast time.

![rain trust](output/rain_trust.png)

### 3. How much survives depends on the season

- **Spring (rolling CV):** weather swings week to week, and even typical forecasts keep about half the gain (45.3 → 40.5).
- **Summer (validation):** weather is steadier, and the gain was small to begin with. Good forecasts keep 43%. Typical forecasts keep none: 44.8 vs 44.7, and on one draw a gap of −0.1 [−2.2, +2.0]. Poor forecasts lose ground.

![sample week](output/sample_week.png)

## What this means for the project

The honest model has to be one trained on forecast-grade weather, and scored only with forecast-grade weather. "Actual weather" numbers are an upper bound and nothing more.

The two evidence sources disagree on whether weather helps at typical forecast quality. Six months of rolling CV say yes, clearly (−4.8 MAE). Three summer months of validation say it's a tie. Neither says it hurts. The test slice (October–December) is autumn, with weather that swings more like spring than summer. That's an argument, not a measurement, so the recipe for the one-time test should name **both** models up front: the typical-forecast-trained weather model as primary, and the no-weather model as the comparison.
