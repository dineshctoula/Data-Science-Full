# Day 56 — PCA

Covariance eigendecomposition. Center first, then keep the loud directions.

## Recipe

1. subtract column means
2. sample covariance `(XᵀX) / (n-1)`
3. `eigh`, sort eigenvalues descending
4. project: `(X - mean) @ Wᵀ`
5. reconstruct: `Z @ W + mean`

## Numbers (seed=56)

**Stretched cloud**
- PC1 holds **98.5%** of the variance, PC2 the leftover 1.5%
- full rank recon MSE ≈ 0
- keeping only PC1 → recon MSE **0.071** (the thin axis gets flattened)

**6-D data, real signal is ~2-D**
| k | recon MSE | cumulative variance |
|--:|----------:|--------------------:|
| 1 | 0.226 | 82.5% |
| 2 | 0.015 | 98.9% |
| 3 | 0.010 | 99.2% |

Two components basically tell the story. The other four are jitter.

## Takeaways

- PCA is a rotation + a ranking, not a clustering method.
- "How many components?" → look at the cumulative scree, not a magic k.
- Reconstruction error is the honest price of dropping dimensions.
