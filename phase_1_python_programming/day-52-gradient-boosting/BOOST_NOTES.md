# Day 52 — Gradient Boosting

Numbers from `main.py` (seed=52).

## Core loop

1. Start with `F0 = mean(y)`
2. Residual `r = y - F`
3. Fit a shallow tree to `r`
4. `F ← F + lr * tree(x)`
5. Repeat

Squared loss on purpose so the "gradient" is just the residual.

## Results

**Noisy sine regression**
- train MSE 1.36 → **0.040** over 60 rounds
- test MSE ≈ **0.137**

**Learning rates** (40 rounds):
| lr | final train MSE |
|---:|----------------:|
| 0.05 | 0.096 |
| 0.10 | 0.051 |
| 0.30 | 0.022 |

Bigger steps fit train faster here; on messier data they'd overfit sooner.

**Blob classification** (L2 boost, threshold 0.5): test accuracy **1.0** on this easy pair.

## Takeaways

- Boosting is sequential; forests are parallel votes.
- `learning_rate` is the shrink knob — don't skip it.
- Binary clf via regressing {0,1} is a hack, but the loss curve still tells the story.
