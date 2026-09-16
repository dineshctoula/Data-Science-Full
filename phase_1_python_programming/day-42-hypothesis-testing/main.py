"""Runnable learning pipeline for Day 42: hypothesis testing."""

import numpy as np

from hypothesis_engine import HypothesisEngine
from visualizer import HypothesisVisualizer


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 42: HYPOTHESIS TESTING")
    print("=" * 80)

    # ---- 1. One-sample t test ------------------------------------------------
    textbook = np.array([71.0, 72.0, 68.0, 70.0, 69.0, 73.0, 71.0, 70.0, 72.0, 69.0])
    t_result = HypothesisEngine.one_sample_t_test(textbook, mu0=70.0)
    print("\n1. One-sample t-test — is the class mean different from 70?")
    print(f"  Sample mean = {textbook.mean():.2f}, n = {textbook.size}")
    print(f"  t = {t_result.statistic:.3f}, df = {t_result.degrees_of_freedom:g}, p = {t_result.p_value:.4f}")
    print(f"  {t_result.decision}")

    # ---- 2. Paired before/after (use one-sample t on differences) ----------
    before, after = HypothesisEngine.generate_before_after_scores()
    differences = after - before
    paired = HypothesisEngine.one_sample_t_test(differences, mu0=0.0, alternative="greater")
    print("\n2. Paired improvement — did scores rise after training?")
    print(f"  Mean gain = {differences.mean():.2f} points on n = {differences.size}")
    print(f"  t = {paired.statistic:.3f}, p = {paired.p_value:.4g}")
    print(f"  {paired.decision}")

    # ---- 3. Two-sample Welch t test ----------------------------------------
    control = before
    variant = after
    welch = HypothesisEngine.two_sample_t_test(control, variant, equal_var=False, alternative="less")
    print("\n3. Two-sample Welch t-test — before vs after groups")
    print(f"  Means: before = {control.mean():.2f}, after = {variant.mean():.2f}")
    print(f"  t = {welch.statistic:.3f}, df ≈ {welch.degrees_of_freedom:.1f}, p = {welch.p_value:.4g}")
    print(f"  {welch.decision}")

    # ---- 4. One-proportion z test (A/B) ------------------------------------
    succ_a, n_a, succ_b, n_b = HypothesisEngine.generate_ab_conversion()
    prop_test = HypothesisEngine.one_proportion_z_test(
        succ_b, n_b, p0=succ_a / n_a, alternative="greater"
    )
    print("\n4. One-proportion z-test — does variant B beat control A?")
    print(f"  Control rate = {succ_a / n_a:.3%} ({succ_a}/{n_a})")
    print(f"  Variant rate = {succ_b / n_b:.3%} ({succ_b}/{n_b})")
    print(f"  z = {prop_test.statistic:.3f}, p = {prop_test.p_value:.4g}")
    print(f"  {prop_test.decision}")

    # ---- 5. Chi-square goodness-of-fit -------------------------------------
    observed = np.array([120.0, 95.0, 85.0, 100.0])
    expected = np.array([100.0, 100.0, 100.0, 100.0])
    chi = HypothesisEngine.chi_square_goodness_of_fit(observed, expected)
    print("\n5. Chi-square goodness-of-fit — do category counts match 25% each?")
    print(f"  χ² = {chi.statistic:.3f}, df = {chi.degrees_of_freedom:g}, p = {chi.p_value:.4g}")
    print(f"  {chi.decision}")

    # ---- 6. Charts ---------------------------------------------------------
    visualizer = HypothesisVisualizer()
    null_image = visualizer.plot_null_distribution(t_result, title="One-sample t-test reference")
    conversion_image = visualizer.plot_conversion_rates(succ_a, n_a, succ_b, n_b)
    chi_image = visualizer.plot_chi_square_contributions(observed, expected)
    print("\n6. Generated visual explanations")
    print(f"  Null distribution: {null_image}")
    print(f"  A/B conversion:    {conversion_image}")
    print(f"  Chi-square terms:  {chi_image}")
    print("\nDay 42 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
