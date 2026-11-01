# What the two projects taught

[Project 1](phase_1_python_programming/day-85-churn-final-test/README.md) (Telco churn, Days 81–85) and [Project 2](phase_1_python_programming/day-89-bike-final-test/README.md) (day-ahead bike demand, Days 86–89) were different problems: independent customers vs ordered hours, and classification vs regression. The same habits mattered in both. Each lesson below comes with the numbers that taught it.

## 1. Split first, and write the split down

Both projects froze their split on day one and committed it: customer IDs in `split_ids.tsv`, cut dates in `split_dates.tsv`. Later days read the file instead of re-splitting, so no change of seed or row order could move a row into test.

**The split must match how the model will be used.** Customers stand alone, so a stratified random split was right. Hours don't: 2012 demand ran 1.4–2.6× the same month in 2011, and a random split would have let the bike model see October while "forecasting" July. → [Day 81](phase_1_python_programming/day-81-churn-kickoff/PROJECT.md), [Day 86](phase_1_python_programming/day-86-bike-kickoff/PROJECT.md)

## 2. Build the dumb baseline before the model, and expect it to be strong

| project | baseline | its score | first real model |
|---|---|---:|---:|
| churn | churn-rate lookup: contract × tenure band × internet | top-20% recall 0.503 | 0.554 (+0.051) |
| bikes | mean of the same hour over the last 4 weeks | MAE 47.1 | 44.7 (−2.4) |

Counting and averaging got most of the way in both. Without the baseline, "AUC 0.885" or "MAE 44.7" sound impressive. With it, they're a modest, honest step.

## 3. Correct the baseline instead of replacing it

The first bike model regressed on the lags themselves and lost to the baseline (MAE 48–50 against 47.1). Keeping the baseline as a log-scale offset, and learning only how today differs from it, is what worked. The same idea fixed weather: describe how today's weather differs from the weeks the baseline averaged, not today's weather alone. → [Day 87](phase_1_python_programming/day-87-bike-first-model/MODEL_NOTES.md)

## 4. A flexible model has to earn its place

Gradient-boosted trees, tuned by CV inside train, chose depth-1 stumps. Stumps are an additive model, the same family as logistic regression. They tied the logistic on ranking (top-20% recall −0.004, interval [−0.019, +0.020]) and lost slightly on log loss. The logistic stayed. → [Day 83](phase_1_python_programming/day-83-churn-boosting/BOOST_NOTES.md)

## 5. Choose on a steady metric, report the headline

Top-20% recall moves in steps of 1 leaver in 280 and jumps around between settings, so the churn model was chosen on log loss and the headline was reported, not chased. Ties within 0.001 went to the simpler model in both projects.

## 6. One small validation slice is a good referee and a poor forecaster

| | validation | CV inside train | test (opened once) |
|---|---:|---:|---:|
| churn log loss | 0.372 | 0.418 | **0.420** |
| bike weather gain (typical forecasts) | +0.1 | −4.8 | **−0.9** |

Validation picked between models well, and the comparisons held on test (churn: +0.051 over the lookup on both). Its *levels* were luckier than the truth. CV over more rows forecast the churn test almost exactly.

## 7. Calibrated probabilities turn into decisions

Once the costs were written down ($30 a call, a 25% save rate, a year of margin), the right call list was everyone with positive expected profit. That rule needs no tuning, and it reached 97% of the hindsight ceiling on validation and 98% on test, earning $3,927 and then $4,181 more than a fixed top 20%. It only works because the probabilities were honest: mean predicted 0.274 against an actual 0.266 on test. → [Day 84](phase_1_python_programming/day-84-churn-call-list/CALL_NOTES.md)

## 8. Train on what you'll have when you predict

A bike model trained on actual weather and fed typical forecasts scored MAE 47.8, worse than using no weather at all (44.7). Training on forecast-grade weather fixed that in every scenario, and the model learned to trust rain calls less (rain multiplier ×0.64 → ×0.79). → [Day 88](phase_1_python_programming/day-88-bike-forecast-weather/STRESS_NOTES.md)

## 9. Resample whatever is actually independent

The 24 hours of a day share weather and events. Resampling single hours gave an interval 2.4× narrower than resampling whole days for the same comparison. Every bike interval resamples days, and every churn interval resamples customers.

## 10. Commit the recipe, then open test once

Both final days committed the recipe before any code that reads test rows, rehearsed the whole pipeline on validation, and opened test through a fingerprint gate that refuses a changed recipe (`test_opened.lock`). The last open decision for Project 2 was a rule for the day after Hurricane Sandy, and it was made from the outage flags before test was opened. → [Day 85](phase_1_python_programming/day-85-churn-final-test/README.md), [Day 89](phase_1_python_programming/day-89-bike-final-test/README.md)

## 11. When a number looks good, find out why

Several of the bugs were only caught because a result looked off:

- A by-month table averaged ten simulated forecasts into one, an ensemble nobody has at midnight, and flattered the primary model (46.2 instead of a tie).
- Column names were dropped before fitting, so the learned weather curves couldn't be read.
- `NaN != NaN` made a no-leak test fail for the right code.
- A plot's y-axis said "validation" on the test chart.

Each fix got its own commit.
