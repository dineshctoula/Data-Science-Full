# Day 61 — train, validate, compare

Signup data, n=260, seed=61. About 66% sign up, so "always say yes" is already
a decent guess. Features: visits, spend, and a noise column that shouldn't help.

## Same split, three models

| model | train | validation | gap |
|---|---:|---:|---:|
| majority | 0.626 | 0.744 | -0.117 |
| nearest centroid | 0.720 | 0.705 | +0.015 |
| logistic | 0.747 | 0.731 | +0.016 |

Validation picked **majority**. The held-out slice was 74% signups, so the
dumb rule looked great. Logistic had the best training accuracy and still
lost the decision.

## Second split (seed=62)

| model | train | validation |
|---|---:|---:|
| majority | 0.665 | 0.654 |
| nearest centroid | 0.621 | 0.474 |
| logistic | 0.747 | 0.718 |

This time **logistic** wins. Centroid fell apart on the new slice.

## Takeaways

- Score every model on the same validation rows.
- Train accuracy is allowed to look better. Don't pick with it.
- One split can crown the wrong model. That's why the second seed is in the script.
- Logistic standardizes columns first. Spend is on a much bigger scale than visits.
