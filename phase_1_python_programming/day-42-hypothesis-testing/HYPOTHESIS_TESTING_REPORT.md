# Day 42 — Hypothesis Testing

## Deliverables

- `hypothesis_engine.py`: one-sample z/t tests, pooled and Welch two-sample t-tests, one-proportion z-test, chi-square goodness-of-fit, and lesson datasets.
- `visualizer.py`: shaded null-reference curves, A/B conversion bars, and chi-square contribution charts.
- `main.py`: a runnable lesson that states H₀/H₁, prints statistics and p-values, and writes the charts.
- Two test modules covering rejection decisions, tail logic, and plot output.

## Key results

| Scenario | Statistic | p-value | Decision at α = 0.05 |
| --- | ---: | ---: | --- |
| Class mean vs 70 (n = 10) | t = 1.00 | 0.343 | Fail to reject |
| Training gain (paired n = 35) | t = 12.60 | ≈ 1.1×10⁻¹⁴ | Reject |
| Before vs after (Welch) | t = −2.52 | 0.0072 | Reject |
| A/B conversion uplift | z = −0.29 | 0.614 | Fail to reject |
| Four-category χ² fit | χ² = 6.50 | 0.090 | Fail to reject |

## Reading the tests

The comments in `hypothesis_engine.py` mark the logic to follow:

1. State H₀ and H₁ before looking at the p-value.
2. The test statistic measures how far the data sit from the null prediction.
3. The p-value is the probability, assuming H₀ is true, of seeing a statistic at least this extreme.
4. Reject H₀ when p < α; otherwise fail to reject (not “accept” H₀).
5. Welch’s t-test does not assume equal variances; the proportion z-test uses p₀ in the standard error.

## Practical checks

- α must lie strictly between 0 and 1; alternatives are `two-sided`, `greater`, or `less`.
- t-tests require positive sample spread; constant columns are rejected.
- Chi-square expected counts must be strictly positive.
- t and χ² tail probabilities are computed by numerical integration without SciPy.

## Run

```bash
cd phase_1_python_programming/day-42-hypothesis-testing
python3 -m unittest -v
python3 main.py
```

The pipeline writes `output/null_distribution.png`, `output/ab_conversion_rates.png`, and `output/chi_square_contributions.png`.
