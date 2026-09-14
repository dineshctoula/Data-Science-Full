"""Runnable learning pipeline for Day 40: descriptive statistics."""

import numpy as np

from stats_engine import DescriptiveStats
from visualizer import StatsVisualizer


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 40: DESCRIPTIVE STATISTICS")
    print("=" * 80)

    textbook = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print("\n1. Center and spread on a symmetric sample")
    print(f"  Data: {textbook}")
    print(f"  Mean = {DescriptiveStats.mean(textbook):.1f}, median = {DescriptiveStats.median(textbook):.1f}")
    print(
        "  Variance: population "
        f"{DescriptiveStats.variance(textbook, sample=False):.1f} vs sample "
        f"{DescriptiveStats.variance(textbook, sample=True):.1f}"
    )
    print(f"  Skewness = {DescriptiveStats.skewness(textbook):.1f}  (symmetric, so the tails match)")

    scores = DescriptiveStats.generate_exam_scores()
    report = DescriptiveStats.summarize(scores)
    print("\n2. Simulated exam scores are slightly right-skewed")
    print(f"  n = {report.count}")
    print(f"  Mean = {report.mean:.3f}, median = {report.median:.3f}")
    print(f"  Sample std = {report.sample_std:.3f}, population std = {report.population_std:.3f}")
    print(f"  Skewness = {report.skewness:.3f}  (mean sits to the right of the median)")

    five = report.five_number
    print("\n3. Five-number summary and Tukey fences")
    print(
        f"  Min={five.minimum:.2f}, Q1={five.q1:.2f}, median={five.median:.2f}, "
        f"Q3={five.q3:.2f}, max={five.maximum:.2f}"
    )
    print(f"  IQR = {five.iqr:.2f}; fences [{five.lower_fence:.2f}, {five.upper_fence:.2f}]")
    print(f"  Outliers ({report.outlier_count}): {np.round(report.outlier_values, 2)}")

    print("\n4. Z-scores locate unusual observations on a common scale")
    print(f"  Mean of z-scores = {report.z_scores.mean():.2e}")
    print(f"  Sample std of z-scores = {report.z_scores.std(ddof=1):.2f}")
    extreme = report.values[np.argmax(np.abs(report.z_scores))]
    extreme_z = report.z_scores[np.argmax(np.abs(report.z_scores))]
    print(f"  Most extreme score {extreme:.1f} has z = {extreme_z:.2f}")

    visualizer = StatsVisualizer()
    histogram_image = visualizer.plot_histogram(
        report.values, report.mean, report.median, title="Exam-score distribution"
    )
    boxplot_image = visualizer.plot_boxplot(report.values, five, title="Exam-score five-number summary")
    zscore_image = visualizer.plot_zscore_strip(report, title="Standardized exam scores")
    print("\n5. Generated visual explanations")
    print(f"  Histogram:  {histogram_image}")
    print(f"  Box plot:   {boxplot_image}")
    print(f"  Z-scores:   {zscore_image}")
    print("\nDay 40 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
