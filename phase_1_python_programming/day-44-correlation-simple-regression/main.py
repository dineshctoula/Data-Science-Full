"""Runnable learning pipeline for Day 44: correlation and simple regression."""

import numpy as np

from regression_engine import RegressionEngine
from visualizer import RegressionVisualizer


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 44: CORRELATION & SIMPLE LINEAR REGRESSION")
    print("=" * 80)

    # ---- 1. Textbook perfect line ------------------------------------------
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x
    perfect = RegressionEngine.pearson_correlation(x, y)
    perfect_fit = RegressionEngine.fit_simple_regression(x, y)
    print("\n1. A perfect positive line has r = 1 and R² = 1")
    print(f"  {perfect.summary()}")
    print(f"  {perfect_fit.summary()}")

    # ---- 2. Study hours vs exam scores -------------------------------------
    hours, scores = RegressionEngine.generate_study_hours_scores()
    association = RegressionEngine.pearson_correlation(hours, scores)
    fit = RegressionEngine.fit_simple_regression(hours, scores)
    print("\n2. Study hours vs exam scores")
    print(f"  {association.summary()}")
    print(f"  {fit.summary()}")
    # For one predictor, R² equals the square of Pearson's r.
    print(f"  Check: R² ≈ r² → {fit.r_squared:.4f} ≈ {association.coefficient**2:.4f}")

    # ---- 3. Prediction -----------------------------------------------------
    new_hours = np.array([3.0, 6.0, 9.0])
    predicted = fit.predict(new_hours)
    print("\n3. Predictions from the fitted line")
    for hour, score in zip(new_hours, predicted):
        print(f"  {hour:.0f} study hours → predicted score {score:.1f}")

    # ---- 4. Pearson vs Spearman on a curve ---------------------------------
    curved_x, curved_y = RegressionEngine.generate_nonlinear_pair()
    pearson = RegressionEngine.pearson_correlation(curved_x, curved_y)
    spearman = RegressionEngine.spearman_correlation(curved_x, curved_y)
    print("\n4. Curved monotone data: ranks beat raw values")
    print(f"  {pearson.summary()}")
    print(f"  {spearman.summary()}")
    print("  Spearman stays higher because the relationship is monotone but not linear.")

    # ---- 5. Residual check -------------------------------------------------
    print("\n5. Residual diagnostics for the study-hours model")
    print(f"  Residual sum ≈ {fit.residuals.sum():.2e}  (OLS residuals center at zero)")
    print(f"  Residual SD ≈ {fit.residuals.std(ddof=1):.3f}")

    # ---- 6. Charts ---------------------------------------------------------
    visualizer = RegressionVisualizer()
    scatter_image = visualizer.plot_scatter_with_fit(
        fit, title="Exam score vs study hours"
    )
    residual_image = visualizer.plot_residuals(fit, title="Residuals vs fitted scores")
    comparison_image = visualizer.plot_correlation_comparison(
        pearson, spearman, curved_x, curved_y
    )
    print("\n6. Generated visual explanations")
    print(f"  Scatter + fit:         {scatter_image}")
    print(f"  Residual plot:         {residual_image}")
    print(f"  Correlation contrast:  {comparison_image}")
    print("\nDay 44 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
