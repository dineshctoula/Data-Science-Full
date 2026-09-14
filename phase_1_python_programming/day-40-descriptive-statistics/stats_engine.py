"""Reusable descriptive-statistics tools for the Day 40 lesson.

The formulas are written out instead of hidden behind a single NumPy call so
center, spread, and outlier rules stay visible while they are being learned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiveNumberSummary:
    """Tukey's five-number summary plus the interquartile range derived from it."""

    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float

    @property
    def iqr(self) -> float:
        """Return the distance between the first and third quartiles."""
        return self.q3 - self.q1

    @property
    def lower_fence(self) -> float:
        """Return the lower 1.5-IQR outlier fence."""
        return self.q1 - 1.5 * self.iqr

    @property
    def upper_fence(self) -> float:
        """Return the upper 1.5-IQR outlier fence."""
        return self.q3 + 1.5 * self.iqr


@dataclass(frozen=True)
class DescriptiveReport:
    """A complete univariate summary produced from one numeric sample."""

    values: np.ndarray
    count: int
    mean: float
    median: float
    modes: np.ndarray
    population_variance: float
    sample_variance: float
    population_std: float
    sample_std: float
    five_number: FiveNumberSummary
    skewness: float
    z_scores: np.ndarray
    outlier_mask: np.ndarray

    @property
    def outlier_values(self) -> np.ndarray:
        """Return the observations that fall outside the 1.5-IQR fences."""
        return self.values[self.outlier_mask]

    @property
    def outlier_count(self) -> int:
        """Return how many observations were flagged as outliers."""
        return int(self.outlier_mask.sum())


class DescriptiveStats:
    """Compute center, spread, shape, and outlier diagnostics from a sample."""

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
    def mean(values: np.ndarray) -> float:
        """Return the arithmetic mean ``sum(x) / n``."""
        sample = DescriptiveStats._as_sample(values)
        return float(sample.sum() / sample.size)

    @staticmethod
    def median(values: np.ndarray) -> float:
        """Return the middle value after sorting, averaging the two centers if needed."""
        sample = np.sort(DescriptiveStats._as_sample(values))
        mid = sample.size // 2
        if sample.size % 2 == 1:
            return float(sample[mid])
        return float((sample[mid - 1] + sample[mid]) / 2.0)

    @staticmethod
    def modes(values: np.ndarray) -> np.ndarray:
        """Return every value that appears most often, sorted.

        Continuous data with all-unique values has no repeated mode, so the
        result is empty rather than claiming every observation is a mode.
        """
        sample = DescriptiveStats._as_sample(values)
        unique_values, counts = np.unique(sample, return_counts=True)
        highest = int(counts.max())
        if highest == 1:
            return np.array([], dtype=float)
        return unique_values[counts == highest]

    @staticmethod
    def percentile(values: np.ndarray, percent: float) -> float:
        """Return a linearly interpolated percentile.

        The rank is ``p / 100 * (n - 1)``.  That matches the common textbook
        linear method and keeps quartile examples easy to check by hand.
        """
        if not np.isfinite(percent) or percent < 0.0 or percent > 100.0:
            raise ValueError("Percent must be a finite number between 0 and 100.")
        sample = np.sort(DescriptiveStats._as_sample(values))
        rank = (percent / 100.0) * (sample.size - 1)
        lower = int(np.floor(rank))
        upper = int(np.ceil(rank))
        if lower == upper:
            return float(sample[lower])
        weight = rank - lower
        return float(sample[lower] + weight * (sample[upper] - sample[lower]))

    @staticmethod
    def five_number_summary(values: np.ndarray) -> FiveNumberSummary:
        """Return min, Q1, median, Q3, and max from one sample."""
        sample = DescriptiveStats._as_sample(values)
        return FiveNumberSummary(
            minimum=float(sample.min()),
            q1=DescriptiveStats.percentile(sample, 25.0),
            median=DescriptiveStats.median(sample),
            q3=DescriptiveStats.percentile(sample, 75.0),
            maximum=float(sample.max()),
        )

    @staticmethod
    def variance(values: np.ndarray, sample: bool = True) -> float:
        """Return variance using ``n - 1`` (sample) or ``n`` (population)."""
        data = DescriptiveStats._as_sample(values, minimum_size=2)
        centered = data - DescriptiveStats.mean(data)
        divisor = data.size - 1 if sample else data.size
        return float((centered**2).sum() / divisor)

    @staticmethod
    def std(values: np.ndarray, sample: bool = True) -> float:
        """Return the square root of the matching variance."""
        return float(np.sqrt(DescriptiveStats.variance(values, sample=sample)))

    @staticmethod
    def z_scores(values: np.ndarray) -> np.ndarray:
        """Standardize a sample with its sample mean and sample standard deviation."""
        data = DescriptiveStats._as_sample(values, minimum_size=2)
        scale = DescriptiveStats.std(data, sample=True)
        if np.isclose(scale, 0.0):
            raise ValueError("Z-scores are undefined when the sample standard deviation is zero.")
        return (data - DescriptiveStats.mean(data)) / scale

    @staticmethod
    def skewness(values: np.ndarray) -> float:
        """Return the Fisher–Pearson moment coefficient of skewness.

        ``m3 / m2^{3/2}`` is zero for a symmetric sample, positive when the
        right tail is longer, and negative when the left tail is longer.
        """
        data = DescriptiveStats._as_sample(values, minimum_size=3)
        centered = data - DescriptiveStats.mean(data)
        second_moment = float((centered**2).sum() / data.size)
        if np.isclose(second_moment, 0.0):
            raise ValueError("Skewness is undefined when every observation is the same.")
        third_moment = float((centered**3).sum() / data.size)
        return third_moment / second_moment**1.5

    @staticmethod
    def outlier_mask(values: np.ndarray) -> np.ndarray:
        """Flag observations outside Tukey's ``Q1 - 1.5 IQR`` and ``Q3 + 1.5 IQR`` fences."""
        data = DescriptiveStats._as_sample(values, minimum_size=2)
        summary = DescriptiveStats.five_number_summary(data)
        return (data < summary.lower_fence) | (data > summary.upper_fence)

    @staticmethod
    def summarize(values: np.ndarray) -> DescriptiveReport:
        """Compute the full Day 40 report used by the lesson pipeline."""
        data = DescriptiveStats._as_sample(values, minimum_size=3)
        five_number = DescriptiveStats.five_number_summary(data)
        return DescriptiveReport(
            values=data,
            count=int(data.size),
            mean=DescriptiveStats.mean(data),
            median=DescriptiveStats.median(data),
            modes=DescriptiveStats.modes(data),
            population_variance=DescriptiveStats.variance(data, sample=False),
            sample_variance=DescriptiveStats.variance(data, sample=True),
            population_std=DescriptiveStats.std(data, sample=False),
            sample_std=DescriptiveStats.std(data, sample=True),
            five_number=five_number,
            skewness=DescriptiveStats.skewness(data),
            z_scores=DescriptiveStats.z_scores(data),
            outlier_mask=DescriptiveStats.outlier_mask(data),
        )

    @staticmethod
    def generate_exam_scores(count: int = 80, seed: int = 40) -> np.ndarray:
        """Simulate slightly right-skewed exam scores with a few high outliers."""
        if not isinstance(count, (int, np.integer)) or count < 3:
            raise ValueError("Count must be an integer of at least 3.")
        rng = np.random.default_rng(seed)
        # A gamma draw is bounded below by zero and has a long right tail, so
        # the mean sits to the right of the median before scores are rescaled.
        body = 45.0 + rng.gamma(shape=4.0, scale=6.0, size=count - 3)
        outliers = np.array([99.0, 100.0, 18.0])
        scores = np.clip(np.concatenate((body, outliers)), 0.0, 100.0)
        return scores
