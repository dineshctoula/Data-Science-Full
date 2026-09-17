"""Runnable learning pipeline for Day 43: confidence intervals."""

import numpy as np

from confidence_engine import ConfidenceEngine
from visualizer import ConfidenceVisualizer


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 43: CONFIDENCE INTERVALS")
    print("=" * 80)

    # ---- 1. Critical values ------------------------------------------------
    z_star = ConfidenceEngine.z_critical(0.95)
    t_star = ConfidenceEngine.t_critical(0.95, df=9)
    print("\n1. Critical values for 95% confidence")
    print(f"  z* ≈ {z_star:.4f}  (Normal reference)")
    # With only 9 degrees of freedom the t critical value is larger than z*.
    print(f"  t*(df=9) ≈ {t_star:.4f}  (heavier tails → wider intervals)")

    # ---- 2. Mean intervals -------------------------------------------------
    textbook = np.array([71.0, 72.0, 68.0, 70.0, 69.0, 73.0, 71.0, 70.0, 72.0, 69.0])
    z_interval = ConfidenceEngine.mean_z_interval(textbook, sigma=5.0, confidence=0.95)
    t_interval = ConfidenceEngine.mean_t_interval(textbook, confidence=0.95)
    print("\n2. Mean of a small exam sample")
    print(f"  Sample mean = {textbook.mean():.2f}, n = {textbook.size}")
    print(f"  {z_interval.summary()}")
    print(f"  {t_interval.summary()}")

    # ---- 3. Proportion interval --------------------------------------------
    successes, trials = ConfidenceEngine.generate_conversion_counts()
    prop_interval = ConfidenceEngine.proportion_interval(successes, trials, confidence=0.95)
    print("\n3. Click-through proportion (Wald)")
    print(f"  p̂ = {successes}/{trials} = {prop_interval.estimate:.3%}")
    print(f"  {prop_interval.summary()}")

    # ---- 4. Difference of means --------------------------------------------
    exam = ConfidenceEngine.generate_exam_sample(n=40, seed=43)
    control = exam[:20]
    treated = exam[20:] + 3.5  # Simulate a modest score lift for the treated half.
    diff_interval = ConfidenceEngine.mean_difference_interval(treated, control, equal_var=False)
    print("\n4. Difference of means (Welch)")
    print(f"  Treated mean = {treated.mean():.2f}, control mean = {control.mean():.2f}")
    print(f"  {diff_interval.summary()}")
    # If zero is outside the interval, the difference is distinguishable from no effect.
    print(f"  Contains 0? {diff_interval.contains(0.0)}")

    # ---- 5. Coverage simulation --------------------------------------------
    coverage = ConfidenceEngine.simulate_mean_coverage(
        true_mean=50.0, sigma=10.0, n=30, confidence=0.95, trials=300, seed=43
    )
    print("\n5. Long-run coverage of 95% z-intervals")
    print(f"  Empirical coverage over 300 draws ≈ {coverage:.1%}")
    print("  (Should land near 95% when the Normal sampling model is correct.)")

    # ---- 6. Charts ---------------------------------------------------------
    visualizer = ConfidenceVisualizer()
    interval_image = visualizer.plot_interval(
        t_interval, true_value=70.0, title="95% t-interval for the class mean"
    )
    coverage_image = visualizer.plot_coverage_simulation(trials=40, seed=43)
    width_image = visualizer.plot_width_vs_confidence(textbook)
    print("\n6. Generated visual explanations")
    print(f"  Single interval:   {interval_image}")
    print(f"  Coverage plot:     {coverage_image}")
    print(f"  Width vs confidence: {width_image}")
    print("\nDay 43 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
