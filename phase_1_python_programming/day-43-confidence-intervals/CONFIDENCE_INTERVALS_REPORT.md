# Day 43 — Confidence Intervals

## Deliverables

- `confidence_engine.py`: z/t critical-value solvers, mean intervals (σ known/unknown), Wald proportion intervals, Welch mean differences, proportion differences, and a coverage simulation.
- `visualizer.py`: a single-interval error-bar chart, a repeated-interval coverage plot, and a confidence-vs-width curve.
- `main.py`: a runnable lesson that builds the intervals and writes the charts.
- Two test modules covering critical values, interval geometry, coverage, and plot output.

## Key results

| Check | Result |
| --- | ---: |
| z* for 95% | ≈ 1.960 |
| t*(df = 9) for 95% | ≈ 2.262 |
| Exam mean (n = 10), z-interval (σ = 5) | [67.40, 73.60] |
| Exam mean (n = 10), t-interval | [69.37, 71.63] |
| Click-through Wald CI | p̂ = 11.25%, [0.0906, 0.1344] |
| Welch treated − control | [−2.83, 7.52] (contains 0) |
| Empirical coverage (300 z-intervals) | ≈ 97.7% |

## Reading the formulas

The comments in `confidence_engine.py` mark the ideas to look for:

1. An interval is `estimate ± critical_value × standard_error`.
2. Higher confidence needs a larger critical value, so the interval gets wider.
3. The standard error of a mean shrinks with `√n`; larger samples give narrower intervals.
4. A t-interval replaces unknown σ with the sample standard deviation and uses heavier tails.
5. Coverage is a long-run property: about 95% of 95% intervals contain the true parameter.

## Practical checks

- Confidence must lie strictly between 0 and 1.
- t-intervals reject constant samples (zero sample standard deviation).
- Proportion successes must satisfy `0 ≤ x ≤ n`.
- Critical values are found by binary search on Normal/t CDFs without SciPy.

## Run

```bash
cd phase_1_python_programming/day-43-confidence-intervals
python3 -m unittest -v
python3 main.py
```

The pipeline writes `output/confidence_interval.png`, `output/coverage_simulation.png`, and `output/width_vs_confidence.png`.
