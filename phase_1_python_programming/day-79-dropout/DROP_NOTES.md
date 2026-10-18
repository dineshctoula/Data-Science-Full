# Day 79 — Dropout

16 noisy points of `sin(1.6 x)` (noise 0.35) and one tanh layer of 64 units, trained for 8000 full-batch steps at lr 0.04. On each training step, dropout switches off a random share of the hidden units and scales up the ones that remain. At prediction time every unit is on. Every number below is a mean over 5 starting seeds. Noise alone sets the best reachable MSE near 0.35² ≈ 0.123.

## 40 validation rows

| dropout | train | validation | test |
|---:|---:|---:|---:|
| 0 | 0.109 | **0.216** | 0.220 |
| 0.1 | 0.149 | 0.217 | **0.169** |
| 0.2 | 0.155 | 0.233 | 0.185 |
| 0.3 | 0.170 | 0.256 | 0.212 |
| 0.5 | 0.172 | 0.270 | 0.222 |

Validation picks **no dropout**, by 0.001. The 300-row test slice clearly prefers 0.1. Forty rows can't separate a gap that small, so this pick is close to a coin flip.

## Same 16 training rows, 200 validation rows

| dropout | train | validation | test |
|---:|---:|---:|---:|
| 0 | 0.109 | 0.206 | 0.208 |
| 0.1 | 0.149 | **0.165** | **0.180** |
| 0.2 | 0.155 | 0.179 | 0.197 |
| 0.3 | 0.170 | 0.203 | 0.225 |
| 0.5 | 0.172 | 0.216 | 0.231 |

With more validation rows the pick is **0.1**, and test MSE drops from **0.208** to **0.180**. The train column is identical to the first table because the training rows are identical. The test numbers shift a little because the second run draws a different test set.

## What the plot shows

![fits](output/fits.png)

Without dropout the curve overshoots to about 1.5 near x = 1 while chasing the one high point at x ≈ 0.5. It also bends through a small bump near x = 0. With dropout 0.1 those wiggles are gone, but the peaks come in short: about 0.73 against a true 1.0. Dropout trades wiggle for flatness. Past 0.1 the flattening costs more than it saves, and 0.5 ends up worse than no dropout.

## Takeaway

Dropout keeps any one hidden unit from carrying a fit on its own. A light rate helped here. A heavy one did not. The tuning lesson from Day 68 still applies: a validation slice that is too small can pick the wrong setting even when a real difference exists.
