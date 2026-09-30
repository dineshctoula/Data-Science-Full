# Day 59 — Bias vs variance

Truth is `sin(1.4x) + 0.25x`, plus noise of 0.45. Each degree is refit on 40
fresh samples of 35 points. Error on a fixed grid splits as

`mse ≈ bias² + variance + noise²`

noise² is 0.45² = 0.203 and doesn't depend on the model.

## Numbers (seed=59)

| degree | bias² | variance | mse |
|-------:|------:|---------:|----:|
| 1 | 0.384 | 0.042 | 0.629 |
| 2 | 0.360 | 0.063 | 0.626 |
| 3 | 0.035 | 0.023 | 0.260 |
| 5 | 0.001 | 0.030 | 0.233 |
| 8 | 0.001 | 0.067 | 0.270 |

A line can't follow the sine, so bias stays high. Degree 5 is about as
faithful as the noise allows. Degree 8 doesn't cut bias any further and
the fits start disagreeing with each other (variance 0.067, mse back up).

Fifteen spaghetti curves: average disagreement was 0.17 for degree 1 and
0.32 for degree 8.

## Takeaways

- Bias is "wrong on average." Variance is "different every sample."
- The noise term is a floor. No polynomial removes it.
- The sweet spot here was degree 5, not the most flexible model.
