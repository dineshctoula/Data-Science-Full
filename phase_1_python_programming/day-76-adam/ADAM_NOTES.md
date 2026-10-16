# Day 76 — Adam

Same bowl as Day 75: loss `0.5 * (flat² + 18 * steep²)`, start at (2, 1), 80 steps. GD and momentum keep lr **0.05**. Adam uses lr **0.1** — that knob is not the same as yesterday's.

## First step

| | flat | steep |
|---|---:|---:|
| GD | 0.100 | 0.900 |
| momentum | 0.015 | 0.135 |
| Adam | 0.100 | 0.100 |

GD follows the raw gradient, so the steep axis gets a step nine times bigger. Adam's bias-corrected first update is the learning rate on both axes. That is the adaptive bit.

## Loss along the way

| step | GD | momentum | Adam |
|---:|---:|---:|---:|
| 1 | 1.895 | 8.704 | 9.095 |
| 10 | 0.717 | 2.320 | 0.577 |
| 40 | 0.033 | 0.020 | 0.119 |
| 80 | 0.0005 | ~5e-6 | 0.0003 |

Momentum still wins this bowl. Adam finishes between GD and momentum, and it overshoots the steep axis around step 20 (weight -0.271). The takeaway is the equalized first step, not a blanket "Adam is best."

## Takeaway

Adam keeps a mean of the gradient and of its square, then divides. Axes with a large gradient get a smaller effective step. On a tiny bowl with a hand-picked momentum beta, momentum can still beat it. Different optimizers want different learning rates.
