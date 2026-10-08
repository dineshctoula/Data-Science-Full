# Day 67 — Learning curves

500 rows of `sin(1.4 x)` plus noise 0.4, x between -2 and 2. 100 rows stay as the test slice. Each point below is the mean MSE over 16 random training subsets of that size.

## Degree 1

| rows | train | test | gap |
|---:|---:|---:|---:|
| 40 | 0.273 | 0.344 | 0.072 |
| 80 | 0.265 | 0.327 | 0.063 |
| 160 | 0.280 | 0.323 | 0.043 |
| 320 | 0.284 | 0.322 | 0.038 |

The test error only moved from **0.344** to **0.322**. The line was already as good as a line gets.

## Degree 4

| rows | train | test | gap |
|---:|---:|---:|---:|
| 40 | 0.153 | 0.192 | 0.040 |
| 80 | 0.154 | 0.175 | 0.021 |
| 160 | 0.163 | 0.166 | 0.003 |
| 320 | 0.166 | 0.164 | -0.002 |

The test error fell from **0.192** to **0.164** and the gap closed. The last gap is slightly negative, which is just this holdout, not a model that beats itself.

At 320 rows the line's test MSE is still **0.322**, about twice the degree-4 number.

## Takeaway

If train and test error are both high and flat, collect a different model, not more of the same rows. If test error is still dropping toward the train error, more rows are doing real work.
