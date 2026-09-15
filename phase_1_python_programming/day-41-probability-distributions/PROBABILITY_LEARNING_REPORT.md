# Day 41 — Probability Distributions

## Deliverables

- `probability_engine.py`: Bernoulli, Binomial, Poisson, Normal, and Exponential PMF/PDF/CDF formulas with mean/variance identities and Monte Carlo helpers.
- `visualizer.py`: Binomial/Poisson stem charts, a Normal PDF overlay on a sample histogram, and Exponential PDF/CDF panels.
- `main.py`: a runnable lesson that checks textbook identities and writes the charts.
- Two test modules covering PMF totals, critical Normal values, Monte Carlo agreement, and plot output.

## Key results

| Distribution | Check | Result |
| --- | --- | ---: |
| Bernoulli(0.3) | mean / variance | `0.30` / `0.21` |
| Binomial(10, 0.4) | P(X = 4) / P(X ≤ 4) | `0.2508` / `0.6331` |
| Poisson(λ = 3) | mean = variance / P(X ≤ 2) | `3.0` / `0.4232` |
| Normal(70, 10) | P(μ ± 1.96σ) | `≈ 0.950` |
| Exponential(λ = 2) | mean / P(X ≤ 0.5) | `0.50` / `0.6321` |
| Monte Carlo | empirical vs exact Binomial CDF | `0.615` vs `0.623` |

## Reading the formulas

The comments in `probability_engine.py` mark the distinctions to look for:

1. A PMF gives probability on discrete points; a PDF gives density whose area is probability.
2. Binomial counts successes in `n` independent Bernoulli trials; Poisson counts rare events with rate `λ`.
3. The Normal curve is symmetric about `μ`; about 95% of the mass sits inside `μ ± 1.96σ`.
4. The Exponential waiting-time mean is `1/λ`, and its CDF at the mean is `1 − e^(−1) ≈ 0.63`.
5. Monte Carlo estimates should land close to the exact CDF when the sample is large enough.

## Practical checks

- Probabilities outside `[0, 1]` and non-positive rates/scales are rejected.
- Negative counts return PMF `0` or CDF `0` rather than raising on every out-of-support query.
- Continuous Exponential densities are zero for negative waiting times.
- Histogram overlays use `density=True` so their area is comparable to the analytic PDF.

## Run

```bash
cd phase_1_python_programming/day-41-probability-distributions
python3 -m unittest -v
python3 main.py
```

The pipeline writes `output/binomial_pmf.png`, `output/poisson_pmf.png`, `output/normal_pdf_sample.png`, and `output/exponential_pdf_cdf.png`.
