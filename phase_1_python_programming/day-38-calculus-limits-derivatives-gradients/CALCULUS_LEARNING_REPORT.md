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

## Run

```bash
cd phase_1_python_programming/day-38-calculus-limits-derivatives-gradients
python3 -m unittest -v
python3 main.py
```
