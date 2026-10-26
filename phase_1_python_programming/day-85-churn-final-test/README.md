# Project 1 — Who should a retention team call?

A churn model and call-list rule for the [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) data (7043 customers, 26.5% churn), built over Days 81–85 with NumPy and pandas only. The logistic regression, the boosted trees, the metrics, and the bootstrap are all written from scratch.

## Result on the held-out test slice (opened once)

| | top-20% recall | AUC | log loss |
|---|---:|---:|---:|
| **final logistic** (fit on train + validation) | **0.471** [0.426, 0.523] | **0.841** [0.816, 0.864] | **0.420** [0.390, 0.452] |
| Day 81 lookup floor (contract × tenure × internet) | 0.420 [0.381, 0.467] | 0.817 [0.791, 0.843] | 0.446 [0.417, 0.475] |

Brackets are 95% bootstrap intervals over the 1057 test customers (281 leavers). Mean predicted churn was 0.274 against an actual 0.266.

**In dollars**, under placeholder economics ($30 per call, 25% of called leavers saved, a save worth 12 months × 50% margin × the monthly bill):

| call list | calls | leavers reached | profit |
|---|---:|---:|---:|
| **break-even rule**: call when expected profit > 0 | 422 | 211 | **$13,627** |
| top 20% by risk | 211 | 132 | $9,446 |

The rule earned **$4,181 more** (paired 95% interval [$2,325, $6,131]). It reached **98%** of the best list size that hindsight could have picked ($13,879).

![test vs validation](output/val_vs_test.png)

## Validation was lucky; cross-validation wasn't

Test scores came in well below validation: top-20% recall 0.471 against 0.554, and log loss 0.420 against 0.372. That's not the model breaking. Day 83 had already scored the same logistic by 5-fold CV inside the training rows, and it got a log loss of **0.4177**. The one test opening landed at **0.420**. The 1056-row validation slice was simply an easy draw, and the CV estimate from 5 × ~990 rows was the honest one.

The comparisons held up even though the levels moved. On validation the model beat the lookup by 0.051 in top-20% recall, and on test by 0.051. On validation the break-even rule beat the fixed list by $3,927, and on test by $4,181. **A single small validation slice is a good referee between models, but a poor forecaster of the final number.**

## How it was built

| day | folder | what it settled |
|---|---|---|
| 81 | [`day-81-churn-kickoff`](../day-81-churn-kickoff) | hash-checked data, cleaning, audit, a **frozen stratified split** committed as IDs, the top-20% recall metric, lookup baselines |
| 82 | [`day-82-churn-first-model`](../day-82-churn-first-model) | L2 logistic by Newton's method. Collapsing mirrored levels and adding log1p(tenure), picked on validation log loss, beat the lookup by +0.051 |
| 83 | [`day-83-churn-boosting`](../day-83-churn-boosting) | Newton-boosted trees with histogram splits, tuned by CV inside train. Stumps won, tied the logistic on ranking, and lost on log loss, so the logistic stayed |
| 84 | [`day-84-churn-call-list`](../day-84-churn-call-list) | economics as explicit assumptions; the break-even rule beat a fixed 20% and sat near the hindsight ceiling |
| 85 | this folder | the recipe was committed first, then a fingerprint gate opened test once (`test_opened.lock`) |

The rules followed throughout:

- Every statistic the model learns comes from the fit rows only.
- Every choice was made on validation, or on CV inside train when there were many settings to tune.
- The test slice was opened once, under a recipe committed beforehand. `holdout_gate.py` refuses to reopen it under any other recipe.

## What the model says

The strongest effects, as odds of churning against the most common level (Day 82):

| level | odds |
|---|---:|
| two-year contract (vs month-to-month) | × 0.18 |
| no internet service (vs fiber) | × 0.29 |
| one-year contract | × 0.48 |
| DSL (vs fiber) | × 0.54 |
| online security | × 0.69 |

Tenure enters as a log curve. Without any hint, boosted stumps learned the same shape on Day 83.

## Limits

- **The economics are placeholders.** The save rate matters most and nobody has measured it. At 5%, no call pays at all (Day 84). A small randomized trial would settle it before any real campaign.
- **The data is a single snapshot.** "Churn" means left within the last month, with no dates, so the model can't be checked on a later period, and there's no guarantee it holds as prices or products change.
- **The test slice is small.** 281 leavers give a top-20% recall interval about ±0.05 wide.

## Run it

```bash
cd phase_1_python_programming/day-81-churn-kickoff && python3 main.py      # fetches and caches the CSV
cd ../day-85-churn-final-test && python3 main.py --rehearse               # full pipeline on validation
python3 main.py                                                           # the recorded test run (gate checks the recipe)
python3 -m unittest test_final                                            # never reads the test slice
```
