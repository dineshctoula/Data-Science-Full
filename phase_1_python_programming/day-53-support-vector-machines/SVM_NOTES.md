# Day 53 — Linear SVM

Soft-margin, primal, SGD. No kernels this round.

## Setup

Minimize roughly `0.5||w||² + C * mean(hinge)`.  
Hinge is `max(0, 1 - y·(w·x+b))` with labels mapped to ±1.

## Numbers (seed=53)

**Separable blobs:** test accuracy **1.0**, ||w|| ≈ 0.63.

**Overlapping blobs — C sweep:**
| C | test acc | ~margin pts (train) |
|--:|---------:|--------------------:|
| 0.1 | 0.455 | 77 |
| 0.5 | 0.455 | 73 |
| 1.0 | 0.667 | 73 |
| 5.0 | 0.818 | 59 |
| 20 | 0.848 | 40 |

Small C basically shrugs at mistakes (wide soft street, weak fit on this
overlap). Bigger C cares more about hinge → fewer margin points, better
accuracy here.

## Takeaways

- Margin lines at ±1 are the "street"; points on/inside it are the soft
  support set (approximate — this is SGD, not exact dual SVs).
- C is the main knob. Too small → underfit; too big → can chase outliers.
- Kernels / RBF next time if we want bent boundaries.
