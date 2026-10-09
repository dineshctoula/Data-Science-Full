# Day 68 — Hyperparameter search

280 rows of `sin(1.4 x)` plus noise 0.35. Half the rows train the neighbors, a quarter picks `k`, a quarter is scored after the pick. Neighbors are plain averages of the nearest training rows.

| k | train | validation | test |
|---:|---:|---:|---:|
| 1 | 0.000 | 0.253 | 0.272 |
| 3 | 0.079 | 0.165 | 0.207 |
| 5 | 0.084 | 0.134 | 0.195 |
| 9 | 0.094 | 0.121 | 0.189 |
| 15 | 0.109 | 0.117 | 0.175 |
| 25 | 0.113 | 0.119 | 0.190 |
| 41 | 0.115 | 0.130 | 0.205 |

`k = 1` wins the training column because each row is its own neighbor, so that error is exactly 0. On the test slice that choice scores **0.272**.

The validation column picks **k = 15** (validation MSE 0.117). That setting scores **0.175** on the test slice. `k = 9` was close behind on validation (0.121), so this is a flat valley, not a sharp winner.

Peeking at the test column also lands on k = 15. On this seed the cheat and the honest pick are the same row. The damage was in trusting the training score.

## Takeaway

Pick the setting from the validation column. The training error for `k = 1` is a lookup, not a forecast. The test column in the table is there so the notes can show the cost. The choice itself only compared validation errors.
