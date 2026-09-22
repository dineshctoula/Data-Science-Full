"""Runnable learning pipeline for Day 48: Naive Bayes classification."""

import numpy as np

from naive_bayes_engine import NaiveBayesEngine
from visualizer import NaiveBayesVisualizer


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 48: NAIVE BAYES CLASSIFICATION")
    print("=" * 80)

    # ---- 1. Bayes building blocks ------------------------------------------
    values = np.array([0.0, 1.0, 2.0])
    log_density = NaiveBayesEngine.gaussian_log_pdf(values, mean=1.0, variance=1.0)
    print("\n1. Gaussian log-likelihood peaks at the class mean")
    for value, density in zip(values, np.exp(log_density)):
        print(f"  N({value:.0f} | μ=1, σ²=1) = {density:.4f}")

    # ---- 2. Three-class iris-like model ------------------------------------
    flowers_x, flowers_y, flower_names = NaiveBayesEngine.generate_iris_like_sample()
    flowers = NaiveBayesEngine.fit(flowers_x, flowers_y, flower_names)
    print("\n2. Three-class Gaussian Naive Bayes on petal features")
    print(f"  {flowers.summary()}")
    for stats in flowers.class_stats:
        print(f"  {stats.summary()}")

    # ---- 3. Confusion matrix -----------------------------------------------
    flower_cm = NaiveBayesEngine.confusion_matrix(
        flowers_y, flowers.predictions, flowers.classes
    )
    print("\n3. Confusion matrix for the flower model")
    print(f"  {flower_cm.astype(int).tolist()}")

    # ---- 4. Binary exam pass/fail model ------------------------------------
    exam_x, exam_y, exam_names = NaiveBayesEngine.generate_binary_exam_sample()
    exam = NaiveBayesEngine.fit(exam_x, exam_y, exam_names)
    print("\n4. Binary exam model (study hours + sleep hours)")
    print(f"  {exam.summary()}")
    for stats in exam.class_stats:
        label = "fail" if stats.label == 0 else "pass"
        print(
            f"  {label}: prior = {stats.prior:.3f}, "
            f"mean study = {stats.means[0]:.2f}, mean sleep = {stats.means[1]:.2f}"
        )

    # ---- 5. Posterior predictions ------------------------------------------
    new_students = np.array([[2.0, 5.0], [7.5, 7.0], [10.0, 8.0]])
    probs = NaiveBayesEngine.predict_proba(new_students, exam.class_stats)
    preds = NaiveBayesEngine.predict(new_students, exam.class_stats)
    print("\n5. Posterior predictions for three students")
    for row, probability, prediction in zip(new_students, probs, preds):
        decision = "pass" if prediction == 1.0 else "fail"
        print(
            f"  study={row[0]:.1f}, sleep={row[1]:.1f} → "
            f"P(fail)={probability[0]:.3f}, P(pass)={probability[1]:.3f} → {decision}"
        )

    # ---- 6. Charts ---------------------------------------------------------
    visualizer = NaiveBayesVisualizer()
    priors_image = visualizer.plot_class_priors(flowers, title="Flower-class priors")
    density_image = visualizer.plot_feature_likelihoods(
        flowers, feature_index=0, title="Petal-length class-conditional densities"
    )
    region_image = visualizer.plot_decision_regions(
        flowers_x, flowers_y, flowers, title="Petal-feature decision regions"
    )
    confusion_image = visualizer.plot_confusion_matrix(
        flower_cm, flowers.classes, title="Flower-class confusion matrix"
    )
    print("\n6. Generated visual explanations")
    print(f"  Class priors:         {priors_image}")
    print(f"  Feature likelihoods:  {density_image}")
    print(f"  Decision regions:     {region_image}")
    print(f"  Confusion matrix:     {confusion_image}")
    print("\nDay 48 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
