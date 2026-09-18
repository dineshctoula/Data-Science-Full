# Day 44 — Correlation & Simple Linear Regression

## Deliverables

- `regression_engine.py`: sample covariance, Pearson/Spearman correlation, OLS intercept/slope, R²/RMSE/MAE, predictions, and lesson datasets.
- `visualizer.py`: scatter-with-fit, residual diagnostic, and Pearson-vs-Spearman comparison charts.
- `main.py`: a runnable lesson that prints the identities and writes the charts.
- Two test modules covering perfect-line recovery, R² = r², curved-data contrast, and plot output.

## Key results

| Check | Result |
| --- | ---: |
| Perfect line `y = 2x` | Pearson r = 1.0, R² = 1.0 |
| Study hours vs scores (n = 40) | r = 0.941, ŷ = 46.10 + 3.91 x, R² = 0.886 |
| Identity | R² = r² |
| Curved monotone data | Pearson 0.887 vs Spearman 0.977 |
| OLS residuals | sum ≈ 0 |

## Reading the formulas

The comments in `regression_engine.py` mark the ideas to look for:

1. Covariance measures how x and y move together; Pearson scales that by the two standard deviations.
2. Spearman's ρ is Pearson's r computed on ranks, so it tracks monotone curves better.
3. The OLS slope is `Cov(x, y) / Var(x)`; the intercept centers the line on `(x̄, ȳ)`.
4. For one predictor, `R² = r²` — the same linear association, expressed as explained variance.
5. Residuals should scatter around zero with no strong leftover curve if the line is a good summary.

## Practical checks

- Unequal lengths, non-finite values, and zero-variance series are rejected.
- Constant x cannot define a regression slope.
- Predictions reuse the fitted intercept and slope on new x values.
- Charts require the typed result objects so plot helpers stay consistent with the engine.

## Run

```bash
cd phase_1_python_programming/day-44-correlation-simple-regression
python3 -m unittest -v
python3 main.py
```

The pipeline writes `output/scatter_with_fit.png`, `output/residual_plot.png`, and `output/correlation_comparison.png`.
