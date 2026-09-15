"""Runnable learning pipeline for Day 41: probability distributions."""

import numpy as np

from probability_engine import ProbabilityEngine
from visualizer import DistributionVisualizer


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 41: PROBABILITY DISTRIBUTIONS")
    print("=" * 80)

    # ---- 1. Bernoulli and Binomial -----------------------------------------
    p = 0.3
    bern = ProbabilityEngine.bernoulli_summary(p)
    print("\n1. Bernoulli and Binomial — counting successes")
    print(f"  Bernoulli(p={p}): P(X=1) = {ProbabilityEngine.bernoulli_pmf(1, p):.2f}")
    print(f"  Mean = {bern.mean:.2f}, variance = {bern.variance:.2f}  (= p(1−p))")

    n, p_bin = 10, 0.4
    bin_summary = ProbabilityEngine.binomial_summary(n, p_bin)
    # Hand-check: P(X = 4) for Binomial(10, 0.4) is the most likely count near the mean.
    print(f"  Binomial(n={n}, p={p_bin}): P(X=4) = {ProbabilityEngine.binomial_pmf(4, n, p_bin):.4f}")
    print(f"  P(X ≤ 4) = {ProbabilityEngine.binomial_cdf(4, n, p_bin):.4f}")
    print(f"  Mean = np = {bin_summary.mean:.1f}, variance = np(1−p) = {bin_summary.variance:.2f}")

    # ---- 2. Poisson --------------------------------------------------------
    lam = 3.0
    poi = ProbabilityEngine.poisson_summary(lam)
    print("\n2. Poisson — rare events in a fixed window")
    print(f"  λ = {lam}: P(X=2) = {ProbabilityEngine.poisson_pmf(2, lam):.4f}")
    print(f"  P(X ≤ 2) = {ProbabilityEngine.poisson_cdf(2, lam):.4f}")
    # The defining identity of the Poisson: mean and variance share the same λ.
    print(f"  Mean = variance = {poi.mean:.1f}")

    # ---- 3. Normal ---------------------------------------------------------
    mu, sigma = 70.0, 10.0
    print("\n3. Normal — continuous scores with μ and σ")
    print(f"  PDF at the mean: f({mu:g}) = {ProbabilityEngine.normal_pdf(mu, mu, sigma):.5f}")
    # P(X ≤ μ) is always 0.5 for a symmetric Normal curve.
    print(f"  P(X ≤ {mu:g}) = {ProbabilityEngine.normal_cdf(mu, mu, sigma):.2f}")
    # The empirical rule: about 95% of mass sits inside μ ± 1.96σ.
    lower, upper = mu - 1.96 * sigma, mu + 1.96 * sigma
    mass = ProbabilityEngine.normal_cdf(upper, mu, sigma) - ProbabilityEngine.normal_cdf(lower, mu, sigma)
    print(f"  P({lower:.1f} ≤ X ≤ {upper:.1f}) ≈ {mass:.3f}  (≈ 95% empirical rule)")

    # ---- 4. Exponential ----------------------------------------------------
    wait_rate = 2.0
    exp_summary = ProbabilityEngine.exponential_summary(wait_rate)
    print("\n4. Exponential — waiting times with rate λ")
    print(f"  λ = {wait_rate}: mean wait = 1/λ = {exp_summary.mean:.2f}")
    print(f"  P(X ≤ 0.5) = {ProbabilityEngine.exponential_cdf(0.5, wait_rate):.4f}")
    print(f"  Variance = 1/λ² = {exp_summary.variance:.2f}")

    # ---- 5. Monte Carlo check ----------------------------------------------
    samples = ProbabilityEngine.sample_binomial(n=10, p=0.5, size=5_000, seed=41)
    empirical = ProbabilityEngine.empirical_probability(samples, "le", 5)
    exact = ProbabilityEngine.binomial_cdf(5, 10, 0.5)
    print("\n5. Monte Carlo vs exact Binomial CDF")
    print(f"  Exact P(X ≤ 5) for Binomial(10, 0.5) = {exact:.4f}")
    print(f"  Empirical estimate from {samples.size} draws = {empirical:.4f}")

    # ---- 6. Charts ---------------------------------------------------------
    visualizer = DistributionVisualizer()
    binomial_image = visualizer.plot_binomial_pmf(n, p_bin, title=f"Binomial({n}, {p_bin}) PMF")
    poisson_image = visualizer.plot_poisson_pmf(lam, title=f"Poisson(λ={lam:g}) PMF")
    normal_samples = ProbabilityEngine.sample_normal(mu, sigma, size=1_200, seed=41)
    normal_image = visualizer.plot_normal_pdf_with_sample(
        normal_samples, mu, sigma, title=f"Normal(μ={mu:g}, σ={sigma:g}) PDF vs sample"
    )
    exponential_image = visualizer.plot_exponential_pdf_cdf(
        wait_rate, title=f"Exponential(λ={wait_rate:g}) PDF and CDF"
    )
    print("\n6. Generated visual explanations")
    print(f"  Binomial PMF:     {binomial_image}")
    print(f"  Poisson PMF:      {poisson_image}")
    print(f"  Normal overlay:   {normal_image}")
    print(f"  Exponential:      {exponential_image}")
    print("\nDay 41 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
