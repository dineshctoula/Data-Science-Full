# Day 73 — Learning rate

240 rows. `y = 3 x0 - 1.5 x1` plus noise of standard deviation 1. Both columns are already standard normal, so this is the step size and not a scaling problem. Same 70/30 split, 40 full-batch steps, three rates.

## lr 0.02

| epoch | train MSE | test MSE |
|---:|---:|---:|
| 1 | 12.096 | 12.294 |
| 10 | 8.472 | 8.608 |
| 40 | 2.944 | 2.937 |

Still coming down at the end. The budget ran out first.

## lr 0.2

| epoch | train MSE | test MSE |
|---:|---:|---:|
| 1 | 8.107 | 8.235 |
| 10 | 0.914 | 0.812 |
| 40 | 0.798 | 0.695 |

Settled by epoch 20. Test MSE **0.695**, near the noise.

## lr 1.9

| epoch | train MSE | test MSE |
|---:|---:|---:|
| 1 | 13.849 | 13.113 |
| 10 | 479.303 | 465.021 |
| 40 | 208833616.733 | 200858205.879 |

Worse after the first step, then it runs away. Same update rule as the other two.

## Takeaway

Too small and you stop before the fit arrives. Too big and the loss grows. 0.2 on this table is the one that lands. The stable range moves when the columns are not on the same scale, which is a different problem from the step itself.
