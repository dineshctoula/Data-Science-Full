# Day 83 — Gradient-boosted trees vs the logistic

Project 1, step 3. Can a flexible tree model beat Day 82's logistic (validation top-20% recall 0.554, log loss 0.3720)? Short answer: **no**. It ties on ranking and is slightly worse on probabilities.

## The booster

The booster is written from scratch in `boost_engine.py`. Day 52's version fit squared-error trees and tried every threshold. This one is built for a 0/1 target:

- **Newton boosting on log loss.** Per row, the slope is g = p − y and the curvature is h = p(1 − p). A leaf's value is −G / (H + λ), and a split is worth ½(G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)). This is the XGBoost recipe, and the same Newton step as Day 82, taken per leaf.
- **Histogram splits.** Each column is binned once on train (at most 32 bins). A node's split search is then one `bincount` plus a running sum, so 400 trees on 4930 rows take about 2 seconds.

## Tuning without spending validation

The booster has three settings: depth, learning rate, and tree count, and tree count alone has hundreds of options. Tuning them all on the 1056 validation rows would leave validation too worn for a fair head-to-head. So the booster was tuned by **5-fold CV inside train**, and validation was opened once at the end.

| setting | best trees | CV log loss | at 500 trees |
|---|---:|---:|---:|
| depth 1, lr 0.05 | 500 | 0.4220 | 0.4220 |
| **depth 1, lr 0.1** | 416 | **0.4204** | 0.4206 |
| depth 2, lr 0.05 | 238 | 0.4217 | 0.4242 |
| depth 2, lr 0.1 | 116 | 0.4216 | 0.4302 |
| depth 3, lr 0.05 | 135 | 0.4232 | 0.4345 |
| depth 3, lr 0.1 | 60 | 0.4231 | 0.4487 |
| depth 4, lr 0.05 | 94 | 0.4244 | 0.4482 |
| depth 4, lr 0.1 | 43 | 0.4250 | 0.4773 |
| *Day 82 logistic, same folds* | — | *0.4177* | — |

![cv curves](output/cv_curves.png)

**Depth 1 wins.** Stumps can't combine columns. If letting trees combine two, three, or four columns at once doesn't help, the churn signal here is mostly additive. Deeper trees also overfit sooner: the depth-4 curve turns up after about 40 trees.

Using the same tie rule as Day 82 (within 0.001 of the best means a tie, and the simpler model wins), the pick is **depth 1, lr 0.1, 276 trees**.

The CV losses (~0.42) are higher than the validation losses (~0.37) because those are different rows. The validation slice happens to be easier than the average training fold. That's why the logistic was also scored on the same folds: only numbers from the same rows can be compared.

## Head-to-head on validation (opened once)

| model | log loss | AUC | top-20% |
|---|---:|---:|---:|
| Day 81 lookup | 0.4094 | 0.858 | 0.503 |
| **Day 82 logistic** | **0.3720** | **0.885** | **0.554** |
| boosted stumps (276) | 0.3775 | 0.884 | 0.551 |

Booster minus logistic, paired bootstrap on the validation rows:

| metric | difference | 95% interval |
|---|---:|---:|
| top-20% recall | −0.004 | [−0.019, +0.020] |
| AUC | −0.001 | [−0.004, +0.003] |
| −log loss | −0.0055 | [−0.0093, −0.0010] |

On ranking, the two models are indistinguishable: both intervals sit squarely on zero, and their probabilities correlate at 0.993. On probability quality, the logistic is slightly but clearly better. So the **logistic stays the project model**. It does as well, has 23 coefficients instead of 276 trees, and its probabilities are a little more honest.

![gains](output/gains.png)

## What the stumps learned

A stump ensemble is a sum of one-column step functions, so each column's curve can be read off exactly (`additive_parts`).

![shapes](output/shapes.png)

- **Tenure:** with no hint, the stumps drew almost exactly the log curve Day 82 added by hand. That's an independent check that `log1p(tenure)` was the right feature.
- **Monthly charges:** the stumps find a step near $70 where the logistic draws a straight line. The step is real in the training data, but it doesn't change who lands on the call list.
- **Total charges:** a tiny, noisy curve, roughly ±0.05 log-odds. This matches the audit: given tenure and monthly charges, it adds almost nothing.

![gain importance](output/gain_importance.png)

Tenure (31%) and a two-year contract (26%) take more than half of all split gain. Monthly charges, no internet service, and a one-year contract cover most of the rest.

## Takeaway

A more flexible model is only worth it if the data has structure the simple one can't express. Here the best trees were stumps, and stumps are an additive model, the same family as a logistic regression. The bar stays at **0.554 top-20% recall, log loss 0.3720**, held by the Day 82 logistic.
