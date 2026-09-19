# Day 45 — Multiple Linear Regression

## Deliverables

- `multiple_regression_engine.py`: design matrix with intercept, OLS via normal equations, R² / adjusted R² / RMSE / MAE, VIF, and housing/collinearity datasets.
- `visualizer.py`: actual-vs-predicted, residual, coefficient, and VIF charts.
- `main.py`: a runnable lesson that fits models, predicts new homes, and writes the charts.
- Two test modules covering coefficient recovery, diagnostics, VIF, and plot output.

## Key results

| Check | Result |
| --- | ---: |
| Noise-free surface `10 + 2a − 0.5b` | exact coefficient recovery, R² = 1 |
| Housing model (n = 80, p = 3) | R² = 0.875, Adj. R² = 0.870 |
| Slopes | sqft ≈ +77, bedrooms ≈ +17.9k, distance ≈ −2.2k |
| Housing VIFs | all ≈ 1.0–1.1 |
| Collinear toy VIFs | x1/x2 ≈ 1075, x3 ≈ 1.0 |

## Reading the formulas

The comments in `multiple_regression_engine.py` mark the ideas to look for:

1. Append a column of ones so β₀ is an intercept inside the same matrix solve.
2. Solve `(XᵀX) β = Xᵀy` instead of multiplying by an explicit inverse.
3. Adjusted R² penalizes unused predictors so adding noise does not look better.
4. VIF_j = 1 / (1 − R²_j) measures how redundant feature j is with the others.
5. OLS residuals sum to nearly zero because the intercept column is in the fit.

## Practical checks

- Need more observations than predictors plus the intercept.
- Singular / collinear design matrices are rejected before solving.
- VIF needs at least two feature columns.
- Predictions rebuild the intercept column for new rows.

## Run

```bash
cd phase_1_python_programming/day-45-multiple-linear-regression
python3 -m unittest -v
python3 main.py
```

The pipeline writes `output/actual_vs_predicted.png`, `output/residual_plot.png`, `output/coefficient_bars.png`, and `output/vif_bars.png`.
