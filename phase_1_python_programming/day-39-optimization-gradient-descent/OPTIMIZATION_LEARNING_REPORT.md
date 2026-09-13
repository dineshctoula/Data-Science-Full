# Day 39 — Optimization: Gradient Descent

## Deliverables

- `optimization_engine.py`: a reusable batch optimizer that records loss and gradient-norm histories.
- `optimization_problems.py`: known-minimum helpers for a 1-D bowl, an elongated 2-D bowl, MSE linear regression, and learning-rate comparisons.
- `visualizer.py`: a loss chart, a contour-plus-path plot, and a learning-rate overlay.
- `main.py`: a runnable lesson that prints the recovered minima and writes the charts.
- Three focused test modules covering engine safeguards, problem-helper convergence, and plot output.

## Key results

The one-dimensional bowl `f(w) = (w − 3)²` starts at `w = −5` and recovers `w = 3`. Loss falls from `64` to about `2 × 10⁻¹⁴` in 80 updates.

The elongated bowl `f(x, y) = (x − 2)² + 4(y + 1)²` is four times steeper in `y`. Starting at `[-3, 3]`, descent reaches `[2, −1]`. The uneven curvature bends the path, which is why the contour chart is part of the lesson.

The same update rule fits linear models. A one-feature example recovers `[intercept, slope] = [4.0, 2.5]`. A two-feature example recovers `[1.5, 2.0, −0.5]`. Features are standardized during fitting, then coefficients are converted back to the original units.

## Reading the optimizer

The comments in the engine and problem helpers mark the three ideas to look for:

1. Record the loss before each update; this is why a run with `n` updates has `n + 1` loss samples.
2. Evaluate the gradient of the current loss.
3. Subtract `learning_rate × gradient`, which moves parameters downhill.

`ProblemResult.initial_loss`, `final_loss`, `steps`, and `loss_reduction_ratio` make that progress easy to print without indexing the history manually. The recorded `parameter_history` is what the contour plot uses.

The pipeline compares rates `0.05`, `0.1`, and `0.2` on the same 1-D bowl for 20 updates. Because each trial starts from the same value, the different curves show the effect of step size directly: `0.05` reduces loss by `98.5%`, while `0.1` and `0.2` essentially finish the job.

## Practical checks

- Inputs must be finite: `NaN` and infinity are rejected before they can contaminate an update.
- A constant feature is rejected because its standard deviation is zero, so it cannot be standardized safely.
- The returned coefficients work directly with the original, unscaled feature values through `OptimizationProblems.predict`.
- A logarithmic loss plot is used only when every plotted loss is positive, because zero cannot be shown on a log scale.
- Contour paths are accepted only for two-dimensional parameter histories.

## Run

```bash
cd phase_1_python_programming/day-39-optimization-gradient-descent
python3 -m unittest -v
python3 main.py
```

The pipeline writes `output/gradient_descent_loss.png`, `output/gradient_descent_contour.png`, and `output/learning_rate_comparison.png`.
