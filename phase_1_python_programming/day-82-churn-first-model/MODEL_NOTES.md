# Day 82 — First churn model

Project 1, step 2. The frozen Day 81 split, metric, and baseline are imported from `day-81-churn-kickoff` (through `project_paths.py`), not copied. Test is still closed.

**The bar from Day 81:** a churn-rate lookup on contract × tenure band × internet type reaches **0.503** top-20% recall on validation (AUC 0.858, log loss 0.4094).

## The model

L2-penalized logistic regression, fit by Newton's method. With 23 columns the Hessian is 24 × 24, so each step is cheap and exact. It converges in 8 steps, with no learning rate or epoch count to tune. The intercept isn't penalized.

Four feature sets, each tried with penalties 1e-4, 1e-3, 1e-2, and 0.1:

| feature set | cols | best val log loss | AUC | top-20% |
|---|---:|---:|---:|---:|
| as-is, 3 numerics | 30 | 0.3835 | 0.879 | 0.561 |
| collapsed mirrors, 3 numerics | 23 | 0.3836 | 0.879 | 0.561 |
| collapsed, drop TotalCharges | 22 | 0.3816 | 0.879 | 0.569 |
| **collapsed, + log1p(tenure)** | 23 | **0.3717** | **0.884** | 0.554 |

- **Collapsing** the mirrored "No internet/phone service" levels removed 7 columns and changed nothing: log loss moved by 0.0001.
- **Dropping TotalCharges** helped slightly, as the audit suggested: it's almost exactly tenure × monthly charges.
- **log1p(tenure)** was the one real gain. Churn falls fast in the first months and slowly after, and a straight line in tenure can't bend like that.

## How the pick was made

The pick goes to the lowest validation log loss, with anything within 0.001 of the best counted as a tie. A tie goes to fewer columns, then to the stronger penalty. That picks **collapsed + log tenure, lam 0.001** (0.3720 against the best 0.3717).

The pick uses log loss, not top-20% recall. With 280 validation leavers, top-20% recall moves in steps of 1/280 and jumps around between settings. By that column alone, "drop TotalCharges" looks best (0.569 vs 0.554), but the gap is 4 leavers. Log loss uses every row's probability and is much steadier, so it makes the choice, and the headline is reported rather than chased.

## Is the win real?

A paired bootstrap resamples the validation rows 2000 times and scores both lists on each draw:

| | model | lookup | difference | 95% interval |
|---|---:|---:|---:|---:|
| top-20% recall | 0.554 | 0.503 | **+0.051** | [+0.009, +0.083] |
| AUC | 0.885 | 0.858 | +0.026 | [+0.015, +0.039] |

The top-20% gain is above zero in 99.2% of resamples. That's about **14 more of the 280 leavers** on the same-sized call list. It's real, but it's modest, and the lower end of the interval is close to zero. Of the 0.754 ceiling, the lookup already reached 0.503 using only counting.

![gains](output/gains.png)

![bootstrap](output/bootstrap_diff.png)

## Are the probabilities honest?

Mean predicted churn on validation is 0.268, against 0.265 actual. In ten equal-count bins the points track the diagonal. The top bin runs a little under (predicted 0.76, actual 0.83).

![calibration](output/calibration.png)

## What it leans on

| level (vs the most common one) | odds of churning |
|---|---:|
| Contract = two year (vs month-to-month) | × 0.18 |
| InternetService = none (vs fiber) | × 0.29 |
| Contract = one year | × 0.48 |
| InternetService = DSL (vs fiber) | × 0.54 |
| OnlineSecurity = yes | × 0.69 |
| PhoneService = no | × 1.88 |

![coefficients](output/coefficients.png)

These coefficients are *conditional*: each one holds every other column fixed. The odd one out is PhoneService = no at ×1.88. On its own, no phone service churns at 24.9% against 26.7% with phone service. Holding monthly charges fixed, though, a customer without phone service is paying that whole bill for internet alone, so this isn't "phone service keeps people".

## Takeaway

The first real model beats counting by about 5 points of top-20% recall. It also gives honest probabilities. The new bar on validation is **0.554 top-20% recall, log loss 0.3720**.
