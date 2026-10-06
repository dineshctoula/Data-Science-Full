# Day 65 — Prep-step leakage

80 rows, 42 columns. Columns 0 and 1 actually drive y. The other 40 are noise.
Both pipelines use the same train/test split and then keep the 3 columns with
the biggest |correlation| with y.

Honest: correlations from the training rows only.
Leaky: correlations from every row, so the test answers help pick the columns.

## Seed 65

| | test MSE | columns |
|---|---:|---|
| honest | 0.589 | 0, 32, 1 |
| leaky | 0.460 | 0, 1, 8 |

Gap of **0.129**. The leaked fit looks better. Column 32 was a noise column that happened to line up in the training slice; the leak swapped it for another noise column that lined up on the whole table.

## 24 seeds

Mean gap **0.099**. Median gap **0.000**. Only **11 of 24** seeds made the leak look better.

That's the honest version of this toy. With k=3, both methods usually grab the two real columns, so the third slot is a coin flip and the average cheat is small. One flattering split is still enough to fool you if you only look at that split.

## Takeaway

Split first. Any step that looks at y — feature ranking, target encoding, filling values — belongs on the training rows only.
