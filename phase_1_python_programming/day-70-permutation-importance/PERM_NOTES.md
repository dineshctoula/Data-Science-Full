# Day 70 — Permutation importance

240 rows, 4 columns. `y = 2.2 x0 - 1.4 x1` plus noise. Columns 2 and 3 are junk. Fit OLS on 168 training rows. Shuffle one validation column at a time, 20 times, and record how much the validation MSE rises. The coefficients are not refit.

Validation MSE before any shuffle: **0.176**.

Slopes (intercept first): -0.009, **2.240**, **-1.369**, -0.008, -0.079.

| column | mean MSE rise | std |
|---|---:|---:|
| x0 | 11.620 | 1.473 |
| x1 | 3.785 | 0.460 |
| x2 | 0.001 | 0.001 |
| x3 | 0.022 | 0.007 |

x0 hurts the most, then x1. The noise columns barely move. x3's rise of 0.022 is shuffle noise, not a third signal. The sign of the slope is not in this table: x1 is negative and still important, because the shuffle destroys the column either way.

## Takeaway

Drop or keep a column by how much the validation error moves when that column is scrambled. A near-zero coefficient and a near-zero rise are the same story here. Do the shuffling on rows the fit did not train on.
