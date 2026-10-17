# Day 78 — Weight decay

80 rows, 42 columns. `y = 2 x0 - 1.2 x1` plus noise with standard deviation 1. The other 40 columns are junk. 36 rows train, 20 pick the decay, 24 are the test slice. Gradient descent with lr 0.1 for 600 epochs. Each step also pulls every slope toward zero by `decay * weight`. The intercept is never pulled.

| decay | train | validation | test | junk weight size | x0 | x1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.004 | 4.838 | 4.697 | 2.273 | +1.27 | -0.82 |
| 0.03 | 0.038 | 3.492 | 3.455 | 1.799 | +1.11 | -0.78 |
| 0.1 | 0.138 | 2.951 | **3.071** | 1.425 | +0.93 | -0.67 |
| 0.3 | 0.375 | 3.175 | 3.162 | 1.115 | +0.73 | -0.51 |
| 1 | 1.014 | 4.023 | 3.521 | 0.753 | +0.47 | -0.33 |
| 3 | 2.056 | 5.160 | 4.130 | 0.433 | +0.26 | -0.18 |

With no decay the training error is 0.004. 36 rows and 43 coefficients is enough room to memorize. The test error is 4.697.

Validation picks **0.1**. Its test MSE is **3.071**, down from 4.697. The junk weights shrink steadily as decay goes up, 2.273 down to 0.433.

The pull doesn't know which columns are real. At decay 3 the x0 slope is 0.26 against a true 2.0, and the test error climbs back to 4.130.

Two things this table does not show:

- Even at decay 0 the x0 slope is 1.27, not 2.0. With fewer rows than coefficients, some of the signal leaks into the junk columns.
- The best test MSE, 3.071, is still about three times the noise variance of 1.0. Decay makes the overfit smaller. It does not stand in for more rows.

## Takeaway

Weight decay is a pull toward zero applied at every step. Choose its strength on validation rows. Too little leaves the memorized noise in. Too much shrinks the slopes you wanted to keep. Leave the intercept out of it.
