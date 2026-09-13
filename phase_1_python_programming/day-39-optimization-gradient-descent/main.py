"""Runnable learning pipeline for Day 39: optimization and gradient descent."""

import numpy as np

from optimization_problems import OptimizationProblems
from visualizer import OptimizationVisualizer


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 39: OPTIMIZATION — GRADIENT DESCENT")
    print("=" * 80)

    bowl = OptimizationProblems.quadratic_bowl(initial_value=-5.0, target=3.0, learning_rate=0.1, iterations=80)
    print("\n1. A 1-D quadratic bowl has a known minimum")
    print(f"  f(w) = (w - 3)² starts at w = -5 and should reach w = 3")
    print(f"  Recovered parameter: {bowl.parameters[0]:.6f}")
    print(f"  Loss: {bowl.initial_loss:.5f} → {bowl.final_loss:.5e} in {bowl.steps} updates")

    elongated = OptimizationProblems.elongated_bowl()
    print("\n2. An elongated 2-D bowl makes the path visible")
    print("  f(x, y) = (x - 2)² + 4(y + 1)² is four times steeper in y")
    print(f"  Start: {np.round(elongated.parameter_history[0], 4)}")
    print(f"  End:   {np.round(elongated.parameters, 6)}  (expected [2, -1])")
    print(f"  Loss: {elongated.initial_loss:.5f} → {elongated.final_loss:.5e}")

    features = np.arange(1.0, 7.0).reshape(-1, 1)
    target = 4.0 + 2.5 * features[:, 0]
    fit = OptimizationProblems.linear_regression(features, target, learning_rate=0.1, iterations=500)
    print("\n3. The same update rule fits a linear model")
    print(f"  Parameters [intercept, slope] = {np.round(fit.parameters, 5)}")
    print(f"  MSE: {fit.initial_loss:.5f} → {fit.final_loss:.5e} in {fit.steps} updates")

    # The same batch update works with multiple columns.  Each returned
    # coefficient stays in the original units of its matching input feature.
    two_feature_inputs = np.array([[1.0, 3.0], [2.0, 1.0], [3.0, 4.0], [4.0, 2.0]])
    two_feature_target = 1.5 + 2.0 * two_feature_inputs[:, 0] - 0.5 * two_feature_inputs[:, 1]
    two_feature_fit = OptimizationProblems.linear_regression(
        two_feature_inputs, two_feature_target, learning_rate=0.1, iterations=500
    )
    print("\n4. The same optimizer handles two features")
    print(f"  Parameters [intercept, x₁, x₂] = {np.round(two_feature_fit.parameters, 5)}")

    learning_rate_trials = OptimizationProblems.compare_quadratic_learning_rates(
        initial_value=-5.0,
        target=3.0,
        learning_rates=np.array([0.05, 0.1, 0.2]),
        iterations=20,
    )
    print("\n5. Learning rate changes the speed of descent")
    for rate, trial in learning_rate_trials.items():
        # The percentage gives a friendlier comparison than tiny final-loss
        # values, especially after a fast-converging run.
        print(f"  rate {rate:.2f}: reduced loss by {trial.loss_reduction_ratio:.1%}")

    visualizer = OptimizationVisualizer()
    loss_image = visualizer.plot_loss_history(fit.loss_history, "Linear regression optimized by gradient descent")
    contour_image = visualizer.plot_contour_path(
        elongated.parameter_history,
        objective=lambda values: float((values[0] - 2.0) ** 2 + 4.0 * (values[1] + 1.0) ** 2),
        target=elongated.target,
        title="Descent path on an elongated 2-D bowl",
    )
    comparison_image = visualizer.plot_learning_rate_comparison(
        {rate: trial.loss_history for rate, trial in learning_rate_trials.items()}
    )
    print("\n6. Generated visual explanations")
    print(f"  Loss chart:         {loss_image}")
    print(f"  Contour path:       {contour_image}")
    print(f"  Rate comparison:    {comparison_image}")
    print("\nDay 39 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
