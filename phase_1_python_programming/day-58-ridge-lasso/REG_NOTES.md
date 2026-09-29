# Day 58 — Ridge and Lasso

Sparse regression toy (seed=58). True signal is `3*x0 - 2*x1`. `x0_copy` is
almost the same as `x0`. The other three columns are noise.

Intercept is not penalized. Ridge uses the normal equations. Lasso is
coordinate descent with soft-thresholding.

## Numbers (30% holdout)

OLS test MSE **0.268**. Noise coefs were already tiny, so there wasn't much
junk to punish. Pushing λ hard mostly hurts.

| λ | ridge test MSE | lasso nonzero | lasso test MSE |
|--:|---------------:|--------------:|---------------:|
| 0.5 | 0.272 | 6/6 | 0.267 |
| 2 | 0.280 | 6/6 | 0.265 |
| 8 | 0.331 | 5/6 | 0.267 |
| 25 | 0.587 | 3/6 | 0.375 |
| 80 | 1.825 | 2/6 | 1.310 |

At λ=8 the split is the interesting part, not the MSE:

| | x0 | x1 | x0_copy | noise |
|---|---:|---:|---:|---|
| ols | 2.15 | -2.02 | 0.77 | small |
| ridge | 1.40 | -1.89 | 1.41 | still there |
| lasso | 2.01 | -1.97 | 0.86 | one of them is 0 |

Ridge shared the `x0` weight almost evenly with the copy. Lasso kept more
of it on `x0` and had started deleting noise. By λ=25 only 3 coefficients
survive, and the test error is already worse than OLS.

## Takeaways

- λ=0 ridge is just least squares.
- Ridge shrinks, lasso can zero.
- If the noise columns are weak, a big λ overshrinks a fit that was fine.
