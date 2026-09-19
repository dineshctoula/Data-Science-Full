"""Runnable learning pipeline for Day 45: multiple linear regression."""

import numpy as np

from multiple_regression_engine import MultipleRegressionEngine
from visualizer import MultipleRegressionVisualizer


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 45: MULTIPLE LINEAR REGRESSION")
    print("=" * 80)

    # ---- 1. Recover a known surface ----------------------------------------
    features = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 2.0],
            [5.0, 3.0],
            [6.0, 1.0],
            [7.0, 5.0],
            [8.0, 2.0],
        ]
    )
    target = 10.0 + 2.0 * features[:, 0] - 0.5 * features[:, 1]
    exact = MultipleRegressionEngine.fit(features, target, ("a", "b"))
    print("\n1. Exact recovery on a noise-free surface")
    print(f"  True model: ŷ = 10 + 2·a − 0.5·b")
    print(f"  {exact.summary()}")

    # ---- 2. Housing prices -------------------------------------------------
    housing_x, housing_y, names = MultipleRegressionEngine.generate_housing_sample()
    housing = MultipleRegressionEngine.fit(housing_x, housing_y, names)
    print("\n2. Simulated housing prices with three predictors")
    print(f"  n = {housing.n_observations}, p = {housing.n_features}")
    print(f"  {housing.summary()}")
    for name, slope in zip(names, housing.slopes):
        print(f"  slope({name}) = {slope:,.2f}")

    # ---- 3. Prediction -----------------------------------------------------
    new_homes = np.array(
        [
            [1_200.0, 2.0, 5.0],
            [1_800.0, 3.0, 12.0],
            [2_400.0, 4.0, 3.0],
        ]
    )
    predicted = housing.predict(new_homes)
    print("\n3. Predictions for three new homes")
    for row, price in zip(new_homes, predicted):
        print(
            f"  {row[0]:.0f} sqft, {row[1]:.0f} beds, {row[2]:.0f} km → "
            f"predicted price ${price:,.0f}"
        )

    # ---- 4. Multicollinearity via VIF --------------------------------------
    healthy_vif = MultipleRegressionEngine.variance_inflation_factors(housing_x, names)
    collinear_x, _, collinear_names = MultipleRegressionEngine.generate_collinear_sample()
    collinear_vif = MultipleRegressionEngine.variance_inflation_factors(collinear_x, collinear_names)
    print("\n4. Variance inflation factors")
    print("  Housing predictors (should stay near 1):")
    for name, value in healthy_vif.items():
        print(f"    VIF({name}) = {value:.2f}")
    print("  Collinear toy data (x2 ≈ 2·x1):")
    for name, value in collinear_vif.items():
        print(f"    VIF({name}) = {value:.1f}")

    # ---- 5. Residual check -------------------------------------------------
    print("\n5. Residual diagnostics for the housing model")
    print(f"  Residual sum ≈ {housing.residuals.sum():.2e}")
    print(f"  Residual SD ≈ {housing.residuals.std(ddof=1):,.1f}")

    # ---- 6. Charts ---------------------------------------------------------
    visualizer = MultipleRegressionVisualizer()
    actual_image = visualizer.plot_actual_vs_predicted(housing, title="Housing prices: actual vs predicted")
    residual_image = visualizer.plot_residuals(housing, title="Housing residuals vs fitted values")
    coef_image = visualizer.plot_coefficients(housing, title="Housing slope coefficients")
    vif_image = visualizer.plot_vif(collinear_vif, title="VIF on collinear predictors")
    print("\n6. Generated visual explanations")
    print(f"  Actual vs predicted: {actual_image}")
    print(f"  Residual plot:       {residual_image}")
    print(f"  Coefficients:        {coef_image}")
    print(f"  VIF bars:            {vif_image}")
    print("\nDay 45 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
