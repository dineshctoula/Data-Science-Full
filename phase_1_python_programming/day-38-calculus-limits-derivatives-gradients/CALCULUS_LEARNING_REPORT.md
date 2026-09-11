# Day 38 — Calculus: Limits, Derivatives & Gradients

## Deliverables

- `calculus_engine.py`: one-sided limits, central differences, exact polynomial derivatives, and numerical gradients.
- `gradient_descent.py`: scalar and linear-regression optimization with loss histories.
- `visualizer.py`: a tangent-line plot and a gradient-descent loss chart.
- `main.py`: a runnable lesson using the curriculum exercise.
- Three focused test modules covering numerical agreement, validation, optimization convergence, and plot output.

## Key results

For `f(x) = 3x² + 5x − 2`, the derivative is `f'(x) = 6x + 5`; at `x = 2`, both the central-difference estimate and exact derivative are `17`.

For `g(x, y) = x²y + y³`, the gradient is `[2xy, x² + 3y²]`. At `(1, 2)`, it is `[4, 13]`. The gradient points in the direction of greatest increase, so gradient descent moves in its negative direction.

The linear-regression example uses batch gradient descent on mean squared error. It standardizes features during fitting for stable updates, then returns the intercept and coefficients in the original feature units.

## Reading the gradient-descent code

The comments in `gradient_descent.py` mark the three ideas to look for while studying the loop:

1. Record the loss before each update; this is why a run with `n` updates has `n + 1` loss samples.
2. Calculate the derivative (or vector gradient) of the current loss.
3. Subtract `learning_rate × gradient`, which moves parameters downhill.

`OptimizationResult.initial_loss`, `final_loss`, and `steps` make that progress easy to print without indexing the history manually. The runnable lesson now also fits a two-feature relationship, returning `[1.5, 2.0, -0.5]` for its intercept and two coefficients.

`loss_reduction` and `loss_reduction_ratio` turn a loss history into a concise
progress summary. The pipeline compares rates `0.05`, `0.1`, and `0.2` on the
same quadratic problem, then writes `learning_rate_comparison.png`. Because
each trial starts from the same value and uses the same number of updates, the
different curves show the effect of step size directly.

## Practical checks

- Inputs must be finite: `NaN` and infinity are rejected before they can contaminate an update.
- A constant feature is rejected because its standard deviation is zero, so it cannot be standardized safely.
- The returned coefficients work directly with the original, unscaled feature values through `GradientDescent.predict`.
- Non-finite calculation points are rejected before a limit, derivative, or gradient is estimated.
- A logarithmic loss plot is used only when every plotted loss is positive, because zero cannot be shown on a log scale.

## Run

```bash
cd phase_1_python_programming/day-38-calculus-limits-derivatives-gradients
python3 -m unittest -v
python3 main.py
```
