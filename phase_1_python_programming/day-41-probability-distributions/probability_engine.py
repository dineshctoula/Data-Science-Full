"""Reusable probability-distribution tools for the Day 41 lesson.

Formulas are written out instead of hidden behind SciPy so PMF, PDF, CDF,
and the mean/variance identities stay visible while they are being learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, erfc, factorial

import numpy as np


@dataclass(frozen=True)
class DistributionSummary:
    """Mean, variance, and a short label for one named distribution."""

    name: str
    mean: float
    variance: float
    parameters: dict[str, float]


class ProbabilityEngine:
    """Evaluate common discrete and continuous distributions from first principles."""

    @staticmethod
    def _validate_probability(p: float) -> float:
        """Reject probabilities outside the closed unit interval."""
        value = float(p)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("Probability p must be a finite number between 0 and 1.")
        return value

    @staticmethod
    def _validate_positive(name: str, value: float) -> float:
        """Require a strictly positive finite parameter."""
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be a positive finite number.")
        return number

    @staticmethod
    def _validate_non_negative_integer(name: str, value: int) -> int:
        """Require a whole number that can index a discrete support."""
        if not isinstance(value, (int, np.integer)) or int(value) != value:
            raise ValueError(f"{name} must be an integer.")
        number = int(value)
        if number < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        return number

    # ------------------------------------------------------------------
    # Bernoulli: one trial with success probability p
    # ------------------------------------------------------------------

    @staticmethod
    def bernoulli_pmf(k: int, p: float) -> float:
        """Return P(X = k) for a Bernoulli trial.

        The support is only {0, 1}.  P(X = 1) = p and P(X = 0) = 1 − p.
        """
        success = ProbabilityEngine._validate_probability(p)
        outcome = ProbabilityEngine._validate_non_negative_integer("k", k)
        if outcome == 1:
            return success
        if outcome == 0:
            return 1.0 - success
        # Any other integer is outside the support, so its probability is zero.
        return 0.0

    @staticmethod
    def bernoulli_summary(p: float) -> DistributionSummary:
        """Return E[X] = p and Var(X) = p(1 − p)."""
        success = ProbabilityEngine._validate_probability(p)
        return DistributionSummary(
            name="Bernoulli",
            mean=success,
            # Variance shrinks to zero at the extremes p = 0 and p = 1.
            variance=success * (1.0 - success),
            parameters={"p": success},
        )

    # ------------------------------------------------------------------
    # Binomial: n independent Bernoulli trials
    # ------------------------------------------------------------------

    @staticmethod
    def binomial_pmf(k: int, n: int, p: float) -> float:
        """Return P(X = k) = C(n, k) p^k (1 − p)^(n − k)."""
        trials = ProbabilityEngine._validate_non_negative_integer("n", n)
        success = ProbabilityEngine._validate_probability(p)
        successes = ProbabilityEngine._validate_non_negative_integer("k", k)
        if successes > trials:
            return 0.0
        # comb(n, k) counts the ways to place k successes among n trials.
        return float(comb(trials, successes) * (success**successes) * ((1.0 - success) ** (trials - successes)))

    @staticmethod
    def binomial_cdf(k: int, n: int, p: float) -> float:
        """Return P(X ≤ k) by summing the PMF from 0 through k."""
        trials = ProbabilityEngine._validate_non_negative_integer("n", n)
        success = ProbabilityEngine._validate_probability(p)
        # Flooring negative k to an empty sum keeps the CDF well-defined.
        if k < 0:
            return 0.0
        upper = min(int(k), trials)
        return float(sum(ProbabilityEngine.binomial_pmf(i, trials, success) for i in range(upper + 1)))

    @staticmethod
    def binomial_summary(n: int, p: float) -> DistributionSummary:
        """Return E[X] = np and Var(X) = np(1 − p)."""
        trials = ProbabilityEngine._validate_non_negative_integer("n", n)
        success = ProbabilityEngine._validate_probability(p)
        return DistributionSummary(
            name="Binomial",
            mean=trials * success,
            variance=trials * success * (1.0 - success),
            parameters={"n": float(trials), "p": success},
        )

    # ------------------------------------------------------------------
    # Poisson: counts of rare events in a fixed interval
    # ------------------------------------------------------------------

    @staticmethod
    def poisson_pmf(k: int, lam: float) -> float:
        """Return P(X = k) = e^(−λ) λ^k / k!."""
        rate = ProbabilityEngine._validate_positive("lambda", lam)
        count = ProbabilityEngine._validate_non_negative_integer("k", k)
        # factorial(k) grows quickly; for teaching samples k stays small.
        return float(np.exp(-rate) * (rate**count) / factorial(count))

    @staticmethod
    def poisson_cdf(k: int, lam: float) -> float:
        """Return P(X ≤ k) by summing the Poisson PMF."""
        rate = ProbabilityEngine._validate_positive("lambda", lam)
        if k < 0:
            return 0.0
        return float(sum(ProbabilityEngine.poisson_pmf(i, rate) for i in range(int(k) + 1)))

    @staticmethod
    def poisson_summary(lam: float) -> DistributionSummary:
        """Return the Poisson identity E[X] = Var(X) = λ."""
        rate = ProbabilityEngine._validate_positive("lambda", lam)
        return DistributionSummary(
            name="Poisson",
            mean=rate,
            variance=rate,
            parameters={"lambda": rate},
        )

    # ------------------------------------------------------------------
    # Normal: continuous bell curve with mean μ and std σ
    # ------------------------------------------------------------------

    @staticmethod
    def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Return the Normal density (1 / (σ√(2π))) exp(−(x − μ)² / (2σ²))."""
        mean = float(mu)
        std = ProbabilityEngine._validate_positive("sigma", sigma)
        if not np.isfinite(mean) or not np.isfinite(x):
            raise ValueError("x and mu must be finite numbers.")
        z = (float(x) - mean) / std
        # The constant keeps the total area under the curve equal to one.
        return float(np.exp(-0.5 * z * z) / (std * np.sqrt(2.0 * np.pi)))

    @staticmethod
    def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Return Φ((x − μ) / σ) using the complementary error function.

        ``erfc`` is numerically stable in both tails, so extreme z-scores do
        not collapse to 0 or 1 too early.
        """
        mean = float(mu)
        std = ProbabilityEngine._validate_positive("sigma", sigma)
        if not np.isfinite(mean) or not np.isfinite(x):
            raise ValueError("x and mu must be finite numbers.")
        z = (float(x) - mean) / std
        return float(0.5 * erfc(-z / np.sqrt(2.0)))

    @staticmethod
    def normal_summary(mu: float = 0.0, sigma: float = 1.0) -> DistributionSummary:
        """Return E[X] = μ and Var(X) = σ²."""
        mean = float(mu)
        std = ProbabilityEngine._validate_positive("sigma", sigma)
        if not np.isfinite(mean):
            raise ValueError("mu must be a finite number.")
        return DistributionSummary(
            name="Normal",
            mean=mean,
            variance=std**2,
            parameters={"mu": mean, "sigma": std},
        )

    # ------------------------------------------------------------------
    # Exponential: waiting times with rate λ
    # ------------------------------------------------------------------

    @staticmethod
    def exponential_pdf(x: float, lam: float) -> float:
        """Return λ e^(−λx) for x ≥ 0, and 0 otherwise."""
        rate = ProbabilityEngine._validate_positive("lambda", lam)
        if not np.isfinite(x):
            raise ValueError("x must be a finite number.")
        if x < 0.0:
            return 0.0
        return float(rate * np.exp(-rate * float(x)))

    @staticmethod
    def exponential_cdf(x: float, lam: float) -> float:
        """Return 1 − e^(−λx) for x ≥ 0, and 0 otherwise."""
        rate = ProbabilityEngine._validate_positive("lambda", lam)
        if not np.isfinite(x):
            raise ValueError("x must be a finite number.")
        if x < 0.0:
            return 0.0
        return float(1.0 - np.exp(-rate * float(x)))

    @staticmethod
    def exponential_summary(lam: float) -> DistributionSummary:
        """Return E[X] = 1/λ and Var(X) = 1/λ²."""
        rate = ProbabilityEngine._validate_positive("lambda", lam)
        return DistributionSummary(
            name="Exponential",
            mean=1.0 / rate,
            variance=1.0 / (rate**2),
            parameters={"lambda": rate},
        )

    # ------------------------------------------------------------------
    # Sampling helpers used by the lesson pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def sample_binomial(n: int, p: float, size: int = 1_000, seed: int = 41) -> np.ndarray:
        """Draw ``size`` Binomial(n, p) outcomes for Monte Carlo checks."""
        trials = ProbabilityEngine._validate_non_negative_integer("n", n)
        success = ProbabilityEngine._validate_probability(p)
        if not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("size must be a positive integer.")
        rng = np.random.default_rng(seed)
        return rng.binomial(trials, success, size=int(size)).astype(float)

    @staticmethod
    def sample_normal(mu: float = 0.0, sigma: float = 1.0, size: int = 1_000, seed: int = 41) -> np.ndarray:
        """Draw ``size`` Normal(μ, σ) outcomes for density overlays."""
        summary = ProbabilityEngine.normal_summary(mu, sigma)
        if not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("size must be a positive integer.")
        rng = np.random.default_rng(seed)
        return rng.normal(summary.mean, np.sqrt(summary.variance), size=int(size))

    @staticmethod
    def empirical_probability(samples: np.ndarray, event: str, threshold: float) -> float:
        """Estimate P(X ? threshold) from a Monte Carlo sample.

        ``event`` is one of ``le``, ``ge``, ``eq`` so the same helper can
        check both discrete and continuous lesson statements.
        """
        data = np.asarray(samples, dtype=float)
        if data.ndim != 1 or data.size == 0:
            raise ValueError("samples must be a non-empty one-dimensional array.")
        if not np.isfinite(data).all() or not np.isfinite(threshold):
            raise ValueError("samples and threshold must contain only finite numbers.")
        if event == "le":
            return float(np.mean(data <= threshold))
        if event == "ge":
            return float(np.mean(data >= threshold))
        if event == "eq":
            return float(np.mean(np.isclose(data, threshold)))
        raise ValueError("event must be one of: le, ge, eq.")
