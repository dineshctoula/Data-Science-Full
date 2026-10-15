# Day 75 — Momentum

A two-number bowl. Loss is `0.5 * (flat² + 18 * steep²)`, started at (2, 1). Learning rate 0.05 for 80 steps. The velocity is an average of the gradient: `v = beta * v + (1 - beta) * grad`, then the step is `lr * v`. Beta 0 is ordinary gradient descent.

| step | beta 0 | beta 0.85 | beta 0.9 |
|---:|---:|---:|---:|
| 1 | 1.8950 | 8.7041 | 9.4329 |
| 10 | 0.7170 | 2.3201 | 4.3857 |
| 20 | 0.2570 | 0.3846 | 1.5139 |
| 40 | 0.0330 | 0.0199 | 0.1204 |
| 60 | 0.0042 | 0.0004 | 0.0138 |
| 80 | 0.0005 | 0.0000 | 0.0019 |

The 0.0000 for beta 0.85 is rounding. The value is about **5.3e-6**. Beta 0 is **5.5e-4**. Beta 0.9 is **1.9e-3**, which is worse than no momentum.

Plain gradient descent zeros the steep weight by step 10 (it prints as 0.000) and then creeps along the flat axis: 1.197, 0.717, 0.257, 0.033. Beta 0.85 is behind at step 10 because the first updates are shrunk by `(1 - beta)`, and the steep weight swings to -0.367. By step 40 it has passed the plain run. Beta 0.9 is still swinging at step 80 (steep weight 0.011, flat weight -0.042).

## Takeaway

Momentum helps the slow direction. It does not help on the first step, and beta 0.9 on this bowl is too much: the path rings and finishes above plain gradient descent.
