"""Reusable confidence-interval tools for the Day 43 lesson.

Intervals are built from estimate ± critical_value × standard_error so the
coverage level, margin of error, and sampling variability stay visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, gamma, sqrt

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    """One interval estimate with enough context to interpret the range."""

    name: str
    estimate: float
    lower: float
    upper: float
    confidence: float
    standard_error: float
    critical_value: float
    degrees_of_freedom: float | None = None

    @property
    def margin_of_error(self) -> float:
        """Return half the width of the interval."""
        return float(self.critical_value * self.standard_error)

    @property
    def width(self) -> float:
        """Return the full width upper − lower."""
        return float(self.upper - self.lower)

    def contains(self, value: float) -> bool:
        """Return True when ``value`` lies inside the closed interval."""
        return self.lower <= float(value) <= self.upper

    def summary(self) -> str:
        """Return a short printable description of the interval."""
        pct = 100.0 * self.confidence
        return (
            f"{pct:g}% CI for {self.name}: [{self.lower:.4f}, {self.upper:.4f}] "
            f"(estimate = {self.estimate:.4f}, MoE = {self.margin_of_error:.4f})"
        )


class ConfidenceEngine:
    """Build z- and t-based confidence intervals used in introductory data science."""

    @staticmethod
    def _validate_confidence(confidence: float) -> float:
        """Require a confidence level strictly between 0 and 1."""
        value = float(confidence)
        if not np.isfinite(value) or value <= 0.0 or value >= 1.0:
            raise ValueError("confidence must be a finite number strictly between 0 and 1.")
        return value

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
    def _validate_positive(name: str, value: float) -> float:
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be a positive finite number.")
        return number

    @staticmethod
    def _standard_normal_cdf(z: float) -> float:
        """Return Φ(z) using the complementary error function."""
        return float(0.5 * erfc(-float(z) / sqrt(2.0)))

    @staticmethod
    def _student_t_pdf(t: float, df: float) -> float:
        """Return the Student-t density with ``df`` degrees of freedom."""
        degrees = ConfidenceEngine._validate_positive("df", df)
        coef = gamma((degrees + 1.0) / 2.0) / (sqrt(degrees * np.pi) * gamma(degrees / 2.0))
        return float(coef * (1.0 + (t**2) / degrees) ** (-(degrees + 1.0) / 2.0))

    @staticmethod
    def _student_t_cdf(t: float, df: float) -> float:
        """Return P(T ≤ t) by integrating the t density on a wide grid."""
        degrees = ConfidenceEngine._validate_positive("df", df)
        if t <= -40.0:
            return 0.0
        if t >= 40.0:
            return 1.0
        grid = np.linspace(-40.0, float(t), 6_000)
        density = np.array([ConfidenceEngine._student_t_pdf(x, degrees) for x in grid])
        return float(np.trapezoid(density, grid))

    @staticmethod
    def z_critical(confidence: float) -> float:
        """Return z* such that Φ(z*) − Φ(−z*) = confidence.

        For example, 95% confidence uses z* ≈ 1.96 because each tail holds
        (1 − 0.95) / 2 = 0.025 of the Normal mass.
        """
        level = ConfidenceEngine._validate_confidence(confidence)
        # Binary-search the upper half of the Normal for the right-tail cutoff.
        target = 1.0 - (1.0 - level) / 2.0
        low, high = 0.0, 8.0
        for _ in range(80):
            mid = 0.5 * (low + high)
            if ConfidenceEngine._standard_normal_cdf(mid) < target:
                low = mid
            else:
                high = mid
        return float(0.5 * (low + high))

    @staticmethod
    def t_critical(confidence: float, df: float) -> float:
        """Return t* for a two-sided Student-t interval with ``df`` degrees of freedom."""
        level = ConfidenceEngine._validate_confidence(confidence)
        degrees = ConfidenceEngine._validate_positive("df", df)
        target = 1.0 - (1.0 - level) / 2.0
        low, high = 0.0, 80.0
        for _ in range(80):
            mid = 0.5 * (low + high)
            if ConfidenceEngine._student_t_cdf(mid, degrees) < target:
                low = mid
            else:
                high = mid
        return float(0.5 * (low + high))

    # ------------------------------------------------------------------
    # Mean intervals
    # ------------------------------------------------------------------

    @staticmethod
    def mean_z_interval(
        sample: np.ndarray,
        sigma: float,
        confidence: float = 0.95,
    ) -> ConfidenceInterval:
        """Build a z-interval for μ when the population standard deviation is known."""
        data = ConfidenceEngine._as_sample(sample, minimum_size=2)
        population_std = ConfidenceEngine._validate_positive("sigma", sigma)
        level = ConfidenceEngine._validate_confidence(confidence)
        n = data.size
        estimate = float(data.mean())
        # The standard error of the sample mean shrinks with √n.
        se = population_std / sqrt(n)
        z_star = ConfidenceEngine.z_critical(level)
        return ConfidenceInterval(
            name="μ (σ known)",
            estimate=estimate,
            lower=estimate - z_star * se,
            upper=estimate + z_star * se,
            confidence=level,
            standard_error=se,
            critical_value=z_star,
        )

    @staticmethod
    def mean_t_interval(sample: np.ndarray, confidence: float = 0.95) -> ConfidenceInterval:
        """Build a t-interval for μ when σ is estimated from the sample."""
        data = ConfidenceEngine._as_sample(sample, minimum_size=2)
        level = ConfidenceEngine._validate_confidence(confidence)
        n = data.size
        estimate = float(data.mean())
        sample_std = float(data.std(ddof=1))
        if np.isclose(sample_std, 0.0):
            raise ValueError("Sample standard deviation must be positive for a t-interval.")
        se = sample_std / sqrt(n)
        df = float(n - 1)
        # Small samples need a larger critical value than the Normal z*.
        t_star = ConfidenceEngine.t_critical(level, df)
        return ConfidenceInterval(
            name="μ (σ unknown)",
            estimate=estimate,
            lower=estimate - t_star * se,
            upper=estimate + t_star * se,
            confidence=level,
            standard_error=se,
            critical_value=t_star,
            degrees_of_freedom=df,
        )

    # ------------------------------------------------------------------
    # Proportion interval
    # ------------------------------------------------------------------

    @staticmethod
    def proportion_interval(
        successes: int,
        n: int,
        confidence: float = 0.95,
    ) -> ConfidenceInterval:
        """Build a Wald z-interval for a Binomial proportion p̂ = x / n."""
        level = ConfidenceEngine._validate_confidence(confidence)
        if not isinstance(n, (int, np.integer)) or int(n) <= 0:
            raise ValueError("n must be a positive integer.")
        if not isinstance(successes, (int, np.integer)):
            raise ValueError("successes must be an integer.")
        successes = int(successes)
        trials = int(n)
        if successes < 0 or successes > trials:
            raise ValueError("successes must satisfy 0 ≤ successes ≤ n.")

        phat = successes / trials
        # SE uses the sample proportion; this is the classic Wald formula.
        se = sqrt(phat * (1.0 - phat) / trials)
        z_star = ConfidenceEngine.z_critical(level)
        return ConfidenceInterval(
            name="p (Wald)",
            estimate=phat,
            lower=phat - z_star * se,
            upper=phat + z_star * se,
            confidence=level,
            standard_error=se,
            critical_value=z_star,
        )

    # ------------------------------------------------------------------
    # Difference intervals
    # ------------------------------------------------------------------

    @staticmethod
    def mean_difference_interval(
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        confidence: float = 0.95,
        equal_var: bool = False,
    ) -> ConfidenceInterval:
        """Build a t-interval for μ_A − μ_B from two independent samples."""
        a = ConfidenceEngine._as_sample(sample_a, minimum_size=2)
        b = ConfidenceEngine._as_sample(sample_b, minimum_size=2)
        level = ConfidenceEngine._validate_confidence(confidence)

        mean_a, mean_b = float(a.mean()), float(b.mean())
        var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
        n_a, n_b = a.size, b.size
        estimate = mean_a - mean_b

        if equal_var:
            pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
            if np.isclose(pooled, 0.0):
                raise ValueError("Pooled variance must be positive.")
            se = sqrt(pooled * (1.0 / n_a + 1.0 / n_b))
            df = float(n_a + n_b - 2)
            label = "μ_A − μ_B (pooled)"
        else:
            # Welch SE treats each group's variance separately.
            se_a = var_a / n_a
            se_b = var_b / n_b
            se = sqrt(se_a + se_b)
            if np.isclose(se, 0.0):
                raise ValueError("Standard error must be positive.")
            df = float((se_a + se_b) ** 2 / ((se_a**2) / (n_a - 1) + (se_b**2) / (n_b - 1)))
            label = "μ_A − μ_B (Welch)"

        t_star = ConfidenceEngine.t_critical(level, df)
        return ConfidenceInterval(
            name=label,
            estimate=estimate,
            lower=estimate - t_star * se,
            upper=estimate + t_star * se,
            confidence=level,
            standard_error=se,
            critical_value=t_star,
            degrees_of_freedom=df,
        )

    @staticmethod
    def proportion_difference_interval(
        successes_a: int,
        n_a: int,
        successes_b: int,
        n_b: int,
        confidence: float = 0.95,
    ) -> ConfidenceInterval:
        """Build a Wald z-interval for p̂_A − p̂_B."""
        level = ConfidenceEngine._validate_confidence(confidence)
        for label, value in (
            ("successes_A", successes_a),
            ("n_A", n_a),
            ("successes_B", successes_b),
            ("n_B", n_b),
        ):
            if not isinstance(value, (int, np.integer)) or int(value) < 0:
                raise ValueError(f"{label} must be a non-negative integer.")
        if successes_a > n_a or successes_b > n_b or n_a == 0 or n_b == 0:
            raise ValueError("Trial counts must be positive and successes cannot exceed them.")

        p_a = successes_a / n_a
        p_b = successes_b / n_b
        estimate = p_a - p_b
        # Independent samples: variances of the two proportions add.
        se = sqrt(p_a * (1.0 - p_a) / n_a + p_b * (1.0 - p_b) / n_b)
        z_star = ConfidenceEngine.z_critical(level)
        return ConfidenceInterval(
            name="p_A − p_B (Wald)",
            estimate=estimate,
            lower=estimate - z_star * se,
            upper=estimate + z_star * se,
            confidence=level,
            standard_error=se,
            critical_value=z_star,
        )

    # ------------------------------------------------------------------
    # Coverage simulation and lesson data
    # ------------------------------------------------------------------

    @staticmethod
    def simulate_mean_coverage(
        true_mean: float = 50.0,
        sigma: float = 10.0,
        n: int = 30,
        confidence: float = 0.95,
        trials: int = 400,
        seed: int = 43,
    ) -> float:
        """Estimate the long-run coverage of the z-interval for μ.

        About ``confidence`` of the random intervals should contain the true
        mean when the Normal sampling model is correct.
        """
        level = ConfidenceEngine._validate_confidence(confidence)
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer of at least 2.")
        if not isinstance(trials, (int, np.integer)) or trials < 1:
            raise ValueError("trials must be a positive integer.")
        population_std = ConfidenceEngine._validate_positive("sigma", sigma)
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(int(trials)):
            sample = rng.normal(true_mean, population_std, size=int(n))
            interval = ConfidenceEngine.mean_z_interval(sample, population_std, level)
            # Count how often the constructed interval captures the known truth.
            hits += int(interval.contains(true_mean))
        return hits / float(trials)

    @staticmethod
    def generate_exam_sample(n: int = 40, seed: int = 43) -> np.ndarray:
        """Simulate exam scores centered near 72 with moderate spread."""
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer of at least 2.")
        rng = np.random.default_rng(seed)
        return rng.normal(72.0, 8.0, size=int(n))

    @staticmethod
    def generate_conversion_counts(seed: int = 43) -> tuple[int, int]:
        """Return (successes, n) for a click-through rate near 12%."""
        rng = np.random.default_rng(seed)
        trials = 800
        successes = int(rng.binomial(trials, 0.12))
        return successes, trials
