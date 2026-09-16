"""Reusable hypothesis-testing tools for the Day 42 lesson.

Test statistics and p-values are computed from visible formulas instead of a
black-box library call so the null, alternative, and rejection rule stay clear.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, gamma, sqrt

import numpy as np


@dataclass(frozen=True)
class HypothesisResult:
    """One completed test with enough context to interpret the decision."""

    test_name: str
    null_hypothesis: str
    alternative: str
    statistic: float
    p_value: float
    alpha: float
    degrees_of_freedom: float | None
    reject_null: bool

    @property
    def decision(self) -> str:
        """Return a short human-readable verdict at the chosen alpha."""
        if self.reject_null:
            return f"Reject H₀ at α = {self.alpha:g} (p = {self.p_value:.4g})"
        return f"Fail to reject H₀ at α = {self.alpha:g} (p = {self.p_value:.4g})"


class HypothesisEngine:
    """Run common one- and two-sample tests used in introductory data science."""

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        """Require a significance level strictly between 0 and 1."""
        value = float(alpha)
        if not np.isfinite(value) or value <= 0.0 or value >= 1.0:
            raise ValueError("alpha must be a finite number strictly between 0 and 1.")
        return value

    @staticmethod
    def _validate_alternative(alternative: str) -> str:
        """Normalize the direction of the alternative hypothesis."""
        choice = alternative.strip().lower()
        if choice not in {"two-sided", "greater", "less"}:
            raise ValueError("alternative must be one of: two-sided, greater, less.")
        return choice

    @staticmethod
    def _as_sample(values: np.ndarray, minimum_size: int = 1) -> np.ndarray:
        """Validate a one-dimensional finite sample and return a copy."""
        sample = np.asarray(values, dtype=float)
        if sample.ndim != 1 or sample.size < minimum_size:
            raise ValueError(
                f"Values must be a one-dimensional array with at least {minimum_size} observation(s)."
            )
        if not np.isfinite(sample).all():
            raise ValueError("Values must contain only finite numbers.")
        return sample.copy()

    @staticmethod
    def _standard_normal_cdf(z: float) -> float:
        """Return Φ(z) using the complementary error function."""
        return float(0.5 * erfc(-float(z) / sqrt(2.0)))

    @staticmethod
    def _p_value_from_normal(z: float, alternative: str) -> float:
        """Convert a z statistic into a p-value for the chosen tail(s)."""
        alt = HypothesisEngine._validate_alternative(alternative)
        if alt == "two-sided":
            return float(2.0 * min(HypothesisEngine._standard_normal_cdf(z), 1.0 - HypothesisEngine._standard_normal_cdf(z)))
        if alt == "greater":
            return float(1.0 - HypothesisEngine._standard_normal_cdf(z))
        return float(HypothesisEngine._standard_normal_cdf(z))

    @staticmethod
    def _student_t_pdf(t: float, df: float) -> float:
        """Return the Student-t density with ``df`` degrees of freedom."""
        degrees = HypothesisEngine._validate_positive("df", df)
        # The gamma terms normalize the bell-shaped heavy tail.
        coef = gamma((degrees + 1.0) / 2.0) / (sqrt(degrees * np.pi) * gamma(degrees / 2.0))
        return float(coef * (1.0 + (t**2) / degrees) ** (-(degrees + 1.0) / 2.0))

    @staticmethod
    def _student_t_cdf(t: float, df: float) -> float:
        """Return P(T ≤ t) by integrating the t density on a wide grid."""
        degrees = HypothesisEngine._validate_positive("df", df)
        if t <= -40.0:
            return 0.0
        if t >= 40.0:
            return 1.0
        grid = np.linspace(-40.0, float(t), 8_000)
        density = np.array([HypothesisEngine._student_t_pdf(x, degrees) for x in grid])
        # Trapezoidal integration is accurate enough for teaching-sized samples.
        return float(np.trapezoid(density, grid))

    @staticmethod
    def _p_value_from_t(t_stat: float, df: float, alternative: str) -> float:
        """Convert a t statistic into a p-value for the chosen tail(s)."""
        alt = HypothesisEngine._validate_alternative(alternative)
        if alt == "two-sided":
            lower = HypothesisEngine._student_t_cdf(t_stat, df)
            return float(2.0 * min(lower, 1.0 - lower))
        if alt == "greater":
            return float(1.0 - HypothesisEngine._student_t_cdf(t_stat, df))
        return float(HypothesisEngine._student_t_cdf(t_stat, df))

    @staticmethod
    def _chi_square_pdf(x: float, df: float) -> float:
        """Return the χ² density with ``df`` degrees of freedom for x ≥ 0."""
        degrees = HypothesisEngine._validate_positive("df", df)
        if x < 0.0:
            return 0.0
        coef = 1.0 / (2.0 ** (degrees / 2.0) * gamma(degrees / 2.0))
        return float(coef * x ** (degrees / 2.0 - 1.0) * np.exp(-x / 2.0))

    @staticmethod
    def _chi_square_survival(x: float, df: float) -> float:
        """Return P(χ² ≥ x) by integrating the density from x to a large cutoff."""
        degrees = HypothesisEngine._validate_positive("df", df)
        if x <= 0.0:
            return 1.0
        upper = max(200.0, x + 40.0)
        grid = np.linspace(float(x), upper, 8_000)
        density = np.array([HypothesisEngine._chi_square_pdf(point, degrees) for point in grid])
        return float(np.trapezoid(density, grid))

    @staticmethod
    def _validate_positive(name: str, value: float) -> float:
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be a positive finite number.")
        return number

    # ------------------------------------------------------------------
    # One-sample z test (σ known)
    # ------------------------------------------------------------------

    @staticmethod
    def one_sample_z_test(
        sample: np.ndarray,
        mu0: float,
        sigma: float,
        alternative: str = "two-sided",
        alpha: float = 0.05,
    ) -> HypothesisResult:
        """Test H₀: μ = μ₀ when the population standard deviation is known."""
        data = HypothesisEngine._as_sample(sample, minimum_size=2)
        HypothesisEngine._validate_alternative(alternative)
        significance = HypothesisEngine._validate_alpha(alpha)
        population_std = HypothesisEngine._validate_positive("sigma", sigma)
        if not np.isfinite(mu0):
            raise ValueError("mu0 must be a finite number.")

        n = data.size
        x_bar = float(data.mean())
        # Standard error shrinks with √n because the sample mean averages noise.
        z_stat = (x_bar - mu0) / (population_std / sqrt(n))
        p_value = HypothesisEngine._p_value_from_normal(z_stat, alternative)
        return HypothesisResult(
            test_name="One-sample z-test",
            null_hypothesis=f"H₀: μ = {mu0:g}",
            alternative=f"H₁: μ {HypothesisEngine._word_alternative(alternative)} {mu0:g}",
            statistic=z_stat,
            p_value=p_value,
            alpha=significance,
            degrees_of_freedom=None,
            reject_null=p_value < significance,
        )

    # ------------------------------------------------------------------
    # One-sample t test (σ unknown)
    # ------------------------------------------------------------------

    @staticmethod
    def one_sample_t_test(
        sample: np.ndarray,
        mu0: float,
        alternative: str = "two-sided",
        alpha: float = 0.05,
    ) -> HypothesisResult:
        """Test H₀: μ = μ₀ using the sample standard deviation."""
        data = HypothesisEngine._as_sample(sample, minimum_size=2)
        HypothesisEngine._validate_alternative(alternative)
        significance = HypothesisEngine._validate_alpha(alpha)
        if not np.isfinite(mu0):
            raise ValueError("mu0 must be a finite number.")

        n = data.size
        x_bar = float(data.mean())
        sample_std = float(data.std(ddof=1))
        if np.isclose(sample_std, 0.0):
            raise ValueError("Sample standard deviation must be positive for a t-test.")
        t_stat = (x_bar - mu0) / (sample_std / sqrt(n))
        df = float(n - 1)
        p_value = HypothesisEngine._p_value_from_t(t_stat, df, alternative)
        return HypothesisResult(
            test_name="One-sample t-test",
            null_hypothesis=f"H₀: μ = {mu0:g}",
            alternative=f"H₁: μ {HypothesisEngine._word_alternative(alternative)} {mu0:g}",
            statistic=t_stat,
            p_value=p_value,
            alpha=significance,
            degrees_of_freedom=df,
            reject_null=p_value < significance,
        )

    # ------------------------------------------------------------------
    # Two-sample t test (pooled or Welch)
    # ------------------------------------------------------------------

    @staticmethod
    def two_sample_t_test(
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        equal_var: bool = True,
        alternative: str = "two-sided",
        alpha: float = 0.05,
    ) -> HypothesisResult:
        """Test H₀: μ_A = μ_B for two independent samples."""
        a = HypothesisEngine._as_sample(sample_a, minimum_size=2)
        b = HypothesisEngine._as_sample(sample_b, minimum_size=2)
        HypothesisEngine._validate_alternative(alternative)
        significance = HypothesisEngine._validate_alpha(alpha)

        mean_a, mean_b = float(a.mean()), float(b.mean())
        var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
        n_a, n_b = a.size, b.size

        if equal_var:
            # Pooled variance treats both groups as sharing one common spread.
            pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
            if np.isclose(pooled, 0.0):
                raise ValueError("Pooled variance must be positive for a t-test.")
            se = sqrt(pooled * (1.0 / n_a + 1.0 / n_b))
            df = float(n_a + n_b - 2)
            test_label = "Two-sample t-test (pooled)"
        else:
            # Welch uses separate variances and a Satterthwaite df approximation.
            se_a = var_a / n_a
            se_b = var_b / n_b
            se = sqrt(se_a + se_b)
            if np.isclose(se, 0.0):
                raise ValueError("Standard error must be positive for a Welch t-test.")
            df_num = (se_a + se_b) ** 2
            df_den = (se_a**2) / (n_a - 1) + (se_b**2) / (n_b - 1)
            df = float(df_num / df_den)
            test_label = "Two-sample t-test (Welch)"

        t_stat = (mean_a - mean_b) / se
        p_value = HypothesisEngine._p_value_from_t(t_stat, df, alternative)
        return HypothesisResult(
            test_name=test_label,
            null_hypothesis="H₀: μ_A = μ_B",
            alternative=f"H₁: μ_A {HypothesisEngine._word_alternative(alternative)} μ_B",
            statistic=t_stat,
            p_value=p_value,
            alpha=significance,
            degrees_of_freedom=df,
            reject_null=p_value < significance,
        )

    # ------------------------------------------------------------------
    # One-proportion z test
    # ------------------------------------------------------------------

    @staticmethod
    def one_proportion_z_test(
        successes: int,
        n: int,
        p0: float,
        alternative: str = "two-sided",
        alpha: float = 0.05,
    ) -> HypothesisResult:
        """Test H₀: p = p₀ for a Binomial count summarized as a proportion."""
        HypothesisEngine._validate_alternative(alternative)
        significance = HypothesisEngine._validate_alpha(alpha)
        if not isinstance(n, (int, np.integer)) or int(n) <= 0:
            raise ValueError("n must be a positive integer.")
        if not isinstance(successes, (int, np.integer)):
            raise ValueError("successes must be an integer.")
        successes = int(successes)
        trials = int(n)
        if successes < 0 or successes > trials:
            raise ValueError("successes must satisfy 0 ≤ successes ≤ n.")
        null_p = float(p0)
        if not np.isfinite(null_p) or null_p <= 0.0 or null_p >= 1.0:
            raise ValueError("p0 must be a finite probability strictly between 0 and 1.")

        phat = successes / trials
        # The standard error uses p₀ from the null, not the sample proportion.
        se = sqrt(null_p * (1.0 - null_p) / trials)
        z_stat = (phat - null_p) / se
        p_value = HypothesisEngine._p_value_from_normal(z_stat, alternative)
        return HypothesisResult(
            test_name="One-proportion z-test",
            null_hypothesis=f"H₀: p = {null_p:g}",
            alternative=f"H₁: p {HypothesisEngine._word_alternative(alternative)} {null_p:g}",
            statistic=z_stat,
            p_value=p_value,
            alpha=significance,
            degrees_of_freedom=None,
            reject_null=p_value < significance,
        )

    # ------------------------------------------------------------------
    # Chi-square goodness-of-fit
    # ------------------------------------------------------------------

    @staticmethod
    def chi_square_goodness_of_fit(
        observed: np.ndarray,
        expected: np.ndarray,
        alpha: float = 0.05,
    ) -> HypothesisResult:
        """Test whether observed category counts match expected counts."""
        significance = HypothesisEngine._validate_alpha(alpha)
        obs = np.asarray(observed, dtype=float)
        exp = np.asarray(expected, dtype=float)
        if obs.ndim != 1 or exp.ndim != 1 or obs.size != exp.size or obs.size < 2:
            raise ValueError("observed and expected must be equal-length one-dimensional arrays with at least two categories.")
        if not np.isfinite(obs).all() or not np.isfinite(exp).all():
            raise ValueError("observed and expected must contain only finite numbers.")
        if np.any(obs < 0) or np.any(exp <= 0):
            raise ValueError("observed must be non-negative and expected must be strictly positive.")

        # Pearson's statistic adds squared relative deviations across categories.
        chi2_stat = float(np.sum((obs - exp) ** 2 / exp))
        df = float(obs.size - 1)
        p_value = HypothesisEngine._chi_square_survival(chi2_stat, df)
        return HypothesisResult(
            test_name="Chi-square goodness-of-fit",
            null_hypothesis="H₀: observed counts match expected proportions",
            alternative="H₁: at least one category proportion differs",
            statistic=chi2_stat,
            p_value=p_value,
            alpha=significance,
            degrees_of_freedom=df,
            reject_null=p_value < significance,
        )

    @staticmethod
    def _word_alternative(alternative: str) -> str:
        """Map internal alternative codes to readable math symbols."""
        mapping = {"two-sided": "≠", "greater": ">", "less": "<"}
        return mapping[HypothesisEngine._validate_alternative(alternative)]

    # ------------------------------------------------------------------
    # Lesson datasets
    # ------------------------------------------------------------------

    @staticmethod
    def generate_before_after_scores(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a training program that lifts scores by a few points."""
        rng = np.random.default_rng(seed)
        before = rng.normal(68.0, 9.0, size=35)
        # After scores are correlated with before scores plus a small lift.
        after = before + rng.normal(4.5, 3.0, size=35)
        return before, after

    @staticmethod
    def generate_ab_conversion(seed: int = 42) -> tuple[int, int, int, int]:
        """Return (successes_A, n_A, successes_B, n_B) for a click-through A/B test."""
        rng = np.random.default_rng(seed)
        n_a, n_b = 1_200, 1_150
        # Control stays near 8%; variant is engineered slightly higher for the lesson.
        successes_a = int(rng.binomial(n_a, 0.08))
        successes_b = int(rng.binomial(n_b, 0.095))
        return successes_a, n_a, successes_b, n_b
