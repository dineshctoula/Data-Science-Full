"""Runnable learning pipeline for Day 38: limits, derivatives, and gradients."""

import numpy as np

from calculus_engine import CalculusEngine
from gradient_descent import GradientDescent
from visualizer import CalculusVisualizer


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 38: CALCULUS — LIMITS, DERIVATIVES & GRADIENTS")
    print("=" * 80)

    polynomial = np.array([3.0, 5.0, -2.0])  # f(x) = 3x² + 5x - 2
    point = 2.0
    function = lambda x: CalculusEngine.polynomial_value(polynomial, x)
    left, right = CalculusEngine.one_sided_limit(function, point)
    numerical_slope = CalculusEngine.symmetric_derivative(function, point)
    exact_slope = CalculusEngine.polynomial_derivative(polynomial, point)
    print("\n1. Limits and a derivative")
    print(f"  lim x→{point:.0f}⁻ f(x) = {left:.5f}")
    print(f"  lim x→{point:.0f}⁺ f(x) = {right:.5f}")
    print(f"  f'({point:.0f}), numerical = {numerical_slope:.5f}; exact = {exact_slope:.5f}")

    gradient_point = np.array([1.0, 2.0])
    gradient = CalculusEngine.numerical_gradient(
        lambda values: values[0] ** 2 * values[1] + values[1] ** 3, gradient_point
    )
    print("\n2. Gradient of g(x, y) = x²y + y³")
    print(f"  ∇g(1, 2) = {np.round(gradient, 5)}  (expected [4, 13])")

    features = np.arange(1.0, 7.0).reshape(-1, 1)
    target = 4.0 + 2.5 * features[:, 0]
    fit = GradientDescent.linear_regression(features, target, learning_rate=0.1, iterations=500)
    print("\n3. Gradient descent fits a linear model")
    print(f"  Parameters [intercept, slope] = {np.round(fit.parameters, 5)}")
    print(f"  MSE: {fit.initial_loss:.5f} → {fit.final_loss:.5e} in {fit.steps} updates")

    # The same batch update works with multiple columns.  Each returned
    # coefficient stays in the original units of its matching input feature.
    two_feature_inputs = np.array([[1.0, 3.0], [2.0, 1.0], [3.0, 4.0], [4.0, 2.0]])
    two_feature_target = 1.5 + 2.0 * two_feature_inputs[:, 0] - 0.5 * two_feature_inputs[:, 1]
    two_feature_fit = GradientDescent.linear_regression(
        two_feature_inputs, two_feature_target, learning_rate=0.1, iterations=500
    )
    print("\n4. The same optimizer handles two features")
    print(f"  Parameters [intercept, x₁, x₂] = {np.round(two_feature_fit.parameters, 5)}")

    visualizer = CalculusVisualizer()
    x_values = np.linspace(-3, 3, 300)
    tangent_image = visualizer.plot_tangent_line(x_values, function(x_values), point, exact_slope)
    loss_image = visualizer.plot_loss_history(fit.loss_history, "Linear regression optimized by gradient descent")
    print("\n5. Generated visual explanations")
    print(f"  Tangent-line chart: {tangent_image}")
    print(f"  Loss chart:         {loss_image}")
    print("\nDay 38 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
