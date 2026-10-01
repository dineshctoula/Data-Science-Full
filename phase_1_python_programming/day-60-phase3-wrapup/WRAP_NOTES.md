# Day 60 — Phase 3 wrap-up

One exam-score dataset, not a new topic. Hours, sleep, and a prior score.
The exam was generated as `22 + 4.2*hours + 1.5*sleep + 0.30*prior` plus noise.

## What showed up (seed=60, n=180)

| column | mean | corr with score |
|---|---:|---:|
| hours | 5.67 | 0.730 |
| sleep | 7.06 | 0.196 |
| prior | 72.43 | 0.153 |

Holdout OLS: hours slope **4.40** (truth 4.2), test R² **0.516**, test MSE **32.2**.
Predicting the training mean instead was MSE **67.8**, so the fit is doing real work.

Bootstrap 95% interval for the hours slope, full sample, 300 resamples:
**[3.68, 4.87]**. The true 4.2 sits inside it.

## What this phase was

Days 36–59 were the pieces: linear algebra, calculus and gradients, distributions,
tests and intervals, regression and classification, trees and ensembles, clustering,
PCA, cross-validation, penalties, and the bias-variance split. Today was just
putting a summary, a fit, a holdout, and an interval on the same table.
